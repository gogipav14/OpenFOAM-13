/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     | Website:  https://openfoam.org
    \\  /    A nd           | Copyright (C) 2025 OpenFOAM Foundation
     \\/     M anipulation  |
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "FoamGinkgoLinOp.H"
#include "OGLExecutor.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<typename ValueType>
void Foam::OGL::FoamGinkgoLinOp<ValueType>::initGPUHaloExchange() const
{
    if (!gpuHaloExchange_ && useGPUHaloExchange_ && includeInterfaces_)
    {
        gpuHaloExchange_.reset(new GPUHaloExchange<ValueType>
        (
            this->get_executor(),
            *matrix_,
            *interfaceBouCoeffs_,
            *interfaces_,
            cmpt_
        ));
    }
}


template<typename ValueType>
void Foam::OGL::FoamGinkgoLinOp<ValueType>::updateMatrix() const
{
    try
    {
        // Check if we need to rebuild structure
        if (!structureValid_ || !csrConverter_.valid())
        {
            csrConverter_.reset(new lduToCSR(*matrix_));
            structureValid_ = cacheStructure_;
            valuesValid_ = false;
        }

        // Check if we need to update values
        if (!valuesValid_)
        {
            if constexpr (std::is_same<ValueType, float>::value)
            {
                csrConverter_->updateValuesF32();
                localMatrix_ = csrConverter_->createGinkgoMatrixF32(this->get_executor());
            }
            else
            {
                csrConverter_->updateValues();
                localMatrix_ = csrConverter_->createGinkgoMatrixF64(this->get_executor());
            }

            valuesValid_ = cacheValues_;
        }
    }
    catch (const std::exception& e)
    {
        FatalErrorInFunction
            << "Failed to update CSR matrix: " << e.what()
            << abort(FatalError);
    }
}


template<typename ValueType>
void Foam::OGL::FoamGinkgoLinOp<ValueType>::applyInterfaces
(
    const scalarField& x,
    scalarField& y
) const
{
    if (!includeInterfaces_)
    {
        return;
    }

    // Use OpenFOAM's existing interface update mechanism
    // This handles processor, cyclic, and NCC boundaries correctly
    //
    // Note: const_cast is required here because OpenFOAM's interface methods
    // are non-const (they use internal MPI buffers), but they do not modify
    // the matrix coefficients. This is a design limitation in OpenFOAM's
    // lduMatrix interface that we must work around.
    //
    // The operation is logically const: we're computing y += A_interface * x
    // where A_interface represents processor/cyclic/NCC boundary contributions.

    // Initialize interface matrix update (starts non-blocking MPI sends)
    matrix_->initMatrixInterfaces
    (
        *interfaceBouCoeffs_,
        *interfaces_,
        x,
        y,
        cmpt_
    );

    // Finalize interface matrix update (waits for MPI, applies contributions)
    matrix_->updateMatrixInterfaces
    (
        *interfaceBouCoeffs_,
        *interfaces_,
        x,
        y,
        cmpt_
    );
}


template<typename ValueType>
void Foam::OGL::FoamGinkgoLinOp<ValueType>::copyToHost
(
    const Vector* gkoVec,
    scalarField& foamField
) const
{
    const label n = foamField.size();

    // Get CPU executor for data access
    auto cpuExec = OGLExecutor::instance().cpuExecutor();

    // Create a view on CPU
    auto hostVec = Vector::create(cpuExec, gko::dim<2>(n, 1));
    hostVec->copy_from(gkoVec);

    // Copy data to OpenFOAM field
    const ValueType* data = hostVec->get_const_values();
    for (label i = 0; i < n; i++)
    {
        foamField[i] = static_cast<scalar>(data[i]);
    }
}


template<typename ValueType>
void Foam::OGL::FoamGinkgoLinOp<ValueType>::copyFromHost
(
    const scalarField& foamField,
    Vector* gkoVec
) const
{
    const label n = foamField.size();

    // Get CPU executor
    auto cpuExec = OGLExecutor::instance().cpuExecutor();

    // Create temporary host vector
    auto hostVec = Vector::create(cpuExec, gko::dim<2>(n, 1));
    ValueType* data = hostVec->get_values();

    // Copy data from OpenFOAM field
    for (label i = 0; i < n; i++)
    {
        data[i] = static_cast<ValueType>(foamField[i]);
    }

    // Copy to target executor
    gkoVec->copy_from(hostVec.get());
}


// * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * * //

template<typename ValueType>
void Foam::OGL::FoamGinkgoLinOp<ValueType>::apply_impl
(
    const gko::LinOp* b,
    gko::LinOp* x
) const
{
    try
    {
        // Ensure matrix is up to date
        updateMatrix();

        const label n = nRows();

        // Cast to dense vectors
        const auto bDense = gko::as<const Vector>(b);
        auto xDense = gko::as<Vector>(x);

        // Step 1: Apply local CSR matrix on GPU
        // y_local = A_local * b
        localMatrix_->apply(bDense, xDense);

        // Step 2: Apply interface contributions
        if (includeInterfaces_ && interfaces_ && interfaces_->size() > 0)
        {
            if (useGPUHaloExchange_)
            {
                // GPU-resident halo exchange path
                // Only transfers boundary values, not entire vectors
                initGPUHaloExchange();
                gpuHaloExchange_->apply(bDense, xDense);
            }
            else
            {
                // Legacy CPU-based path (3 full vector transfers)
                // Resize host buffers if needed
                if (xHost_.size() != n)
                {
                    xHost_.setSize(n);
                    yInterface_.setSize(n);
                }

                // Copy x (input) from GPU to CPU
                copyToHost(bDense, xHost_);

                // Initialize interface result to zero
                yInterface_ = 0.0;

                // Compute interface contribution: yInterface = A_interface * xHost
                applyInterfaces(xHost_, yInterface_);

                // Copy current y from GPU, add interface contribution, copy back
                scalarField yHost(n);
                copyToHost(xDense, yHost);

                // y = y_local + y_interface
                forAll(yHost, i)
                {
                    yHost[i] += yInterface_[i];
                }

                // Copy result back to GPU
                copyFromHost(yHost, xDense);
            }
        }
    }
    catch (const std::exception& e)
    {
        FatalErrorInFunction
            << "GPU operator apply failed: " << e.what()
            << abort(FatalError);
    }
}


template<typename ValueType>
void Foam::OGL::FoamGinkgoLinOp<ValueType>::apply_impl
(
    const gko::LinOp* alpha,
    const gko::LinOp* b,
    const gko::LinOp* beta,
    gko::LinOp* x
) const
{
    try
    {
        // For simplicity, implement using the basic apply
        // y = alpha * A * b + beta * y

        // Ensure matrix is up to date
        updateMatrix();

        const label n = nRows();

        // Cast inputs
        const auto alphaDense = gko::as<const gko::matrix::Dense<ValueType>>(alpha);
        const auto betaDense = gko::as<const gko::matrix::Dense<ValueType>>(beta);
        const auto bDense = gko::as<const Vector>(b);
        auto xDense = gko::as<Vector>(x);

        // Create temporary for A*b
        auto temp = Vector::create(this->get_executor(), gko::dim<2>(n, 1));

        // Compute A*b into temp
        this->apply(bDense, temp.get());

        // Compute y = alpha * temp + beta * y
        // Using Ginkgo's scaled add
        auto one = gko::initialize<gko::matrix::Dense<ValueType>>({1.0}, this->get_executor());
        xDense->scale(betaDense);
        temp->scale(alphaDense);
        xDense->add_scaled(one.get(), temp.get());
    }
    catch (const std::exception& e)
    {
        FatalErrorInFunction
            << "GPU scaled operator apply failed: " << e.what()
            << abort(FatalError);
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<typename ValueType>
Foam::OGL::FoamGinkgoLinOp<ValueType>::FoamGinkgoLinOp
(
    std::shared_ptr<const gko::Executor> exec
)
:
    gko::EnableLinOp<FoamGinkgoLinOp<ValueType>>(exec, gko::dim<2>(0, 0)),
    matrix_(nullptr),
    interfaceBouCoeffs_(nullptr),
    interfaceIntCoeffs_(nullptr),
    interfaces_(nullptr),
    cmpt_(0),
    csrConverter_(nullptr),
    localMatrix_(nullptr),
    includeInterfaces_(false),
    cacheStructure_(false),
    cacheValues_(false),
    structureValid_(false),
    valuesValid_(false),
    useGPUHaloExchange_(false),
    gpuHaloExchange_(nullptr)
{}


template<typename ValueType>
Foam::OGL::FoamGinkgoLinOp<ValueType>::FoamGinkgoLinOp
(
    std::shared_ptr<const gko::Executor> exec,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const direction cmpt,
    bool includeInterfaces,
    bool cacheStructure,
    bool cacheValues,
    bool useGPUHaloExchange
)
:
    gko::EnableLinOp<FoamGinkgoLinOp<ValueType>>(
        exec,
        gko::dim<2>(matrix.diag().size(), matrix.diag().size())
    ),
    matrix_(&matrix),
    interfaceBouCoeffs_(&interfaceBouCoeffs),
    interfaceIntCoeffs_(&interfaceIntCoeffs),
    interfaces_(&interfaces),
    cmpt_(cmpt),
    csrConverter_(nullptr),
    localMatrix_(nullptr),
    xHost_(matrix.diag().size()),
    yInterface_(matrix.diag().size()),
    includeInterfaces_(includeInterfaces),
    cacheStructure_(cacheStructure),
    cacheValues_(cacheValues),
    structureValid_(false),
    valuesValid_(false),
    useGPUHaloExchange_(useGPUHaloExchange),
    gpuHaloExchange_(nullptr)
{}


template<typename ValueType>
Foam::OGL::FoamGinkgoLinOp<ValueType>::FoamGinkgoLinOp
(
    const FoamGinkgoLinOp& other
)
:
    gko::EnableLinOp<FoamGinkgoLinOp<ValueType>>(other),
    matrix_(other.matrix_),
    interfaceBouCoeffs_(other.interfaceBouCoeffs_),
    interfaceIntCoeffs_(other.interfaceIntCoeffs_),
    interfaces_(other.interfaces_),
    cmpt_(other.cmpt_),
    csrConverter_(nullptr),
    localMatrix_(nullptr),
    xHost_(other.xHost_.size()),
    yInterface_(other.yInterface_.size()),
    includeInterfaces_(other.includeInterfaces_),
    cacheStructure_(other.cacheStructure_),
    cacheValues_(other.cacheValues_),
    structureValid_(false),
    valuesValid_(false),
    useGPUHaloExchange_(other.useGPUHaloExchange_),
    gpuHaloExchange_(nullptr)
{}


template<typename ValueType>
Foam::OGL::FoamGinkgoLinOp<ValueType>&
Foam::OGL::FoamGinkgoLinOp<ValueType>::operator=
(
    const FoamGinkgoLinOp& other
)
{
    if (this != &other)
    {
        gko::EnableLinOp<FoamGinkgoLinOp<ValueType>>::operator=(other);
        matrix_ = other.matrix_;
        interfaceBouCoeffs_ = other.interfaceBouCoeffs_;
        interfaceIntCoeffs_ = other.interfaceIntCoeffs_;
        interfaces_ = other.interfaces_;
        cmpt_ = other.cmpt_;
        csrConverter_.reset();
        localMatrix_.reset();
        xHost_.setSize(other.xHost_.size());
        yInterface_.setSize(other.yInterface_.size());
        includeInterfaces_ = other.includeInterfaces_;
        cacheStructure_ = other.cacheStructure_;
        cacheValues_ = other.cacheValues_;
        structureValid_ = false;
        valuesValid_ = false;
        useGPUHaloExchange_ = other.useGPUHaloExchange_;
        gpuHaloExchange_.reset();
    }
    return *this;
}


template<typename ValueType>
Foam::OGL::FoamGinkgoLinOp<ValueType>&
Foam::OGL::FoamGinkgoLinOp<ValueType>::operator=
(
    FoamGinkgoLinOp&& other
)
{
    if (this != &other)
    {
        gko::EnableLinOp<FoamGinkgoLinOp<ValueType>>::operator=(std::move(other));
        matrix_ = other.matrix_;
        interfaceBouCoeffs_ = other.interfaceBouCoeffs_;
        interfaceIntCoeffs_ = other.interfaceIntCoeffs_;
        interfaces_ = other.interfaces_;
        cmpt_ = other.cmpt_;
        csrConverter_.reset();
        other.csrConverter_.reset();
        localMatrix_ = std::move(other.localMatrix_);
        xHost_.transfer(other.xHost_);
        yInterface_.transfer(other.yInterface_);
        includeInterfaces_ = other.includeInterfaces_;
        cacheStructure_ = other.cacheStructure_;
        cacheValues_ = other.cacheValues_;
        structureValid_ = other.structureValid_;
        valuesValid_ = other.valuesValid_;
        useGPUHaloExchange_ = other.useGPUHaloExchange_;
        gpuHaloExchange_ = std::move(other.gpuHaloExchange_);
    }
    return *this;
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

template<typename ValueType>
std::shared_ptr<typename Foam::OGL::FoamGinkgoLinOp<ValueType>::Vector>
Foam::OGL::FoamGinkgoLinOp<ValueType>::createVector() const
{
    return Vector::create(this->get_executor(), gko::dim<2>(nRows(), 1));
}


template<typename ValueType>
std::shared_ptr<typename Foam::OGL::FoamGinkgoLinOp<ValueType>::Vector>
Foam::OGL::FoamGinkgoLinOp<ValueType>::createVector
(
    const scalarField& field
) const
{
    auto vec = createVector();
    copyFromHost(field, vec.get());
    return vec;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Explicit instantiation
template class Foam::OGL::FoamGinkgoLinOp<float>;
template class Foam::OGL::FoamGinkgoLinOp<double>;


// ************************************************************************* //
