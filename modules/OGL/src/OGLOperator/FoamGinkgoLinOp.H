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

Class
    Foam::OGL::FoamGinkgoLinOp

Description
    Ginkgo LinearOperator that wraps OpenFOAM's lduMatrix and correctly
    applies A*x including interface contributions.

    The operator application is:
        y = A_local * x + A_interface * x

    Where:
    - A_local is the CSR matrix on GPU for internal cell-cell couplings
    - A_interface handles processor/cyclic/NCC boundary contributions

    This design ensures CG/BiCGStab see a consistent A*x every iteration,
    which is critical for convergence and correctness.

SourceFiles
    FoamGinkgoLinOp.C

\*---------------------------------------------------------------------------*/

#ifndef FoamGinkgoLinOp_H
#define FoamGinkgoLinOp_H

#include "lduMatrix.H"
#include "FieldField.H"
#include "lduInterfaceFieldPtrsList.H"
#include "lduToCSR.H"
#include "GPUHaloExchange.H"

#include "GinkgoCompat.H"
#include <memory>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                     Class FoamGinkgoLinOp Declaration
\*---------------------------------------------------------------------------*/

template<typename ValueType>
class FoamGinkgoLinOp
:
    public gko::EnableLinOp<FoamGinkgoLinOp<ValueType>>,
    public gko::EnableCreateMethod<FoamGinkgoLinOp<ValueType>>
{
    friend class gko::EnablePolymorphicObject<FoamGinkgoLinOp<ValueType>, gko::LinOp>;
    friend class gko::EnableCreateMethod<FoamGinkgoLinOp<ValueType>>;

public:

    // Public Types

        using CsrMatrix = gko::matrix::Csr<ValueType, int>;
        using Vector = gko::matrix::Dense<ValueType>;


private:

    // Private Data

        //- Pointer to the OpenFOAM lduMatrix (non-owning)
        const lduMatrix* matrix_;

        //- Interface boundary coefficients (non-owning)
        const FieldField<Field, scalar>* interfaceBouCoeffs_;

        //- Interface internal coefficients (non-owning)
        const FieldField<Field, scalar>* interfaceIntCoeffs_;

        //- Interface field pointers (non-owning)
        const lduInterfaceFieldPtrsList* interfaces_;

        //- Direction component for vector fields
        direction cmpt_;

        //- LDU to CSR converter (owns structure and value caching)
        mutable autoPtr<lduToCSR> csrConverter_;

        //- Ginkgo CSR matrix for local part (on GPU)
        mutable std::shared_ptr<CsrMatrix> localMatrix_;

        //- Host-side buffer for x (for interface computation)
        mutable scalarField xHost_;

        //- Host-side buffer for y (interface contribution)
        mutable scalarField yInterface_;

        //- Whether to include interface contributions
        bool includeInterfaces_;

        //- Cache structure (rowPtr/colIdx) between solves
        bool cacheStructure_;

        //- Cache values between solves (disable for dynamic meshes)
        bool cacheValues_;

        //- Structure is valid
        mutable bool structureValid_;

        //- Values are valid
        mutable bool valuesValid_;

        //- Use GPU-resident halo exchange (faster but requires more GPU memory)
        bool useGPUHaloExchange_;

        //- GPU halo exchange (lazy initialized)
        mutable std::unique_ptr<GPUHaloExchange<ValueType>> gpuHaloExchange_;


    // Private Member Functions

        //- Initialize GPU halo exchange if needed
        void initGPUHaloExchange() const;

        //- Build/update the CSR matrix
        void updateMatrix() const;

        //- Compute interface contribution: y += A_interface * x
        void applyInterfaces
        (
            const scalarField& x,
            scalarField& y
        ) const;

        //- Copy Ginkgo dense vector to OpenFOAM scalarField
        void copyToHost
        (
            const Vector* gkoVec,
            scalarField& foamField
        ) const;

        //- Copy OpenFOAM scalarField to Ginkgo dense vector
        void copyFromHost
        (
            const scalarField& foamField,
            Vector* gkoVec
        ) const;


protected:

    // Protected Constructors

        //- Construct with only executor (required by Ginkgo CRTP pattern)
        explicit FoamGinkgoLinOp(std::shared_ptr<const gko::Executor> exec);


    // Protected Member Functions

        //- Apply the operator: y = A * x
        void apply_impl(const gko::LinOp* b, gko::LinOp* x) const override;

        //- Apply the operator with scaling: y = alpha * A * x + beta * y
        void apply_impl
        (
            const gko::LinOp* alpha,
            const gko::LinOp* b,
            const gko::LinOp* beta,
            gko::LinOp* x
        ) const override;


public:

    // Constructors

        //- Construct from lduMatrix and interfaces
        FoamGinkgoLinOp
        (
            std::shared_ptr<const gko::Executor> exec,
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const direction cmpt = 0,
            bool includeInterfaces = true,
            bool cacheStructure = true,
            bool cacheValues = false,
            bool useGPUHaloExchange = true  //- Use GPU-resident halo exchange
        );

        //- Copy constructor
        FoamGinkgoLinOp(const FoamGinkgoLinOp& other);

        //- Copy assignment (required by Ginkgo's EnablePolymorphicAssignment)
        FoamGinkgoLinOp& operator=(const FoamGinkgoLinOp& other);

        //- Move assignment (required by Ginkgo's EnablePolymorphicObject)
        FoamGinkgoLinOp& operator=(FoamGinkgoLinOp&& other);


    //- Destructor
    virtual ~FoamGinkgoLinOp() = default;


    // Member Functions

        //- Invalidate cached structure (call when mesh topology changes)
        void invalidateStructure()
        {
            structureValid_ = false;
            valuesValid_ = false;
        }

        //- Invalidate cached values (call when coefficients change)
        void invalidateValues()
        {
            valuesValid_ = false;
        }

        //- Update matrix and interface pointers (for solver caching
        //  across OpenFOAM re-instantiations where the same matrix
        //  object is reused but the solver is recreated)
        void updatePointers
        (
            const lduMatrix& matrix,
            const FieldField<Field, scalar>& interfaceBouCoeffs,
            const FieldField<Field, scalar>& interfaceIntCoeffs,
            const lduInterfaceFieldPtrsList& interfaces,
            const direction cmpt
        )
        {
            matrix_ = &matrix;
            interfaceBouCoeffs_ = &interfaceBouCoeffs;
            interfaceIntCoeffs_ = &interfaceIntCoeffs;
            interfaces_ = &interfaces;
            cmpt_ = cmpt;
        }

        //- Access the local CSR matrix
        std::shared_ptr<CsrMatrix> localMatrix() const
        {
            updateMatrix();
            return localMatrix_;
        }

        //- Number of rows
        label nRows() const
        {
            return matrix_->diag().size();
        }

        //- Create a compatible Ginkgo dense vector
        std::shared_ptr<Vector> createVector() const;

        //- Create a Ginkgo dense vector from OpenFOAM field
        std::shared_ptr<Vector> createVector(const scalarField& field) const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
