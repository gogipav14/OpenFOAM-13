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

#include "GPUHaloExchange.H"
#include "OGLExecutor.H"
#include "lduInterfaceField.H"
#include "processorLduInterfaceField.H"
#include "Pstream.H"

#include <cstdlib>

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

template<typename ValueType>
bool Foam::OGL::GPUHaloExchange<ValueType>::detectCudaAwareMPI()
{
    // Check environment variables that indicate CUDA-aware MPI
    // Common variables set by various MPI implementations:
    // - OMPI_MCA_opal_cuda_support (OpenMPI)
    // - MPICH_GPU_SUPPORT_ENABLED (MPICH)
    // - MV2_USE_CUDA (MVAPICH2)

    const char* ompiCuda = std::getenv("OMPI_MCA_opal_cuda_support");
    if (ompiCuda && std::string(ompiCuda) == "true")
    {
        return true;
    }

    const char* mpichGpu = std::getenv("MPICH_GPU_SUPPORT_ENABLED");
    if (mpichGpu && std::string(mpichGpu) == "1")
    {
        return true;
    }

    const char* mv2Cuda = std::getenv("MV2_USE_CUDA");
    if (mv2Cuda && std::string(mv2Cuda) == "1")
    {
        return true;
    }

    // User can also force GPU-aware MPI via OGL-specific variable
    const char* oglGpuMpi = std::getenv("OGL_CUDA_AWARE_MPI");
    if (oglGpuMpi && std::string(oglGpuMpi) == "1")
    {
        return true;
    }

    return false;
}


template<typename ValueType>
void Foam::OGL::GPUHaloExchange<ValueType>::initializeBuffers()
{
    const label nInterfaces = interfaces_.size();

    faceCellsGPU_.resize(nInterfaces);
    sendBuffersGPU_.resize(nInterfaces);
    recvBuffersGPU_.resize(nInterfaces);
    sendBuffersHost_.resize(nInterfaces);
    recvBuffersHost_.resize(nInterfaces);
    bouCoeffsGPU_.resize(nInterfaces);

    totalBoundaryFaces_ = 0;

    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const lduInterfaceField& intf = interfaces_[i];
            const labelUList& faceCells = intf.interface().faceCells();
            const label size = faceCells.size();

            totalBoundaryFaces_ += size;

            // Copy face cell indices to GPU
            // Note: const_cast needed for Ginkgo array view, but data is
            // immediately copied so original is not modified
            auto faceCellsHost = gko::array<int>::view(
                cpuExec_,
                size,
                const_cast<int*>(
                    reinterpret_cast<const int*>(faceCells.cdata())
                )
            ).copy_to_array();
            faceCellsHost.set_executor(exec_);
            faceCellsGPU_[i] = std::move(faceCellsHost);

            // Allocate GPU send/receive buffers
            sendBuffersGPU_[i] = Vector::create(exec_, gko::dim<2>(size, 1));
            recvBuffersGPU_[i] = Vector::create(exec_, gko::dim<2>(size, 1));

            // Allocate host staging buffers (used when CUDA-aware MPI unavailable)
            sendBuffersHost_[i].setSize(size);
            recvBuffersHost_[i].setSize(size);

            // Copy boundary coefficients to GPU
            const scalarField& bouCoeffs = interfaceBouCoeffs_[i];
            auto bouCoeffsHost = Vector::create(cpuExec_, gko::dim<2>(size, 1));
            ValueType* data = bouCoeffsHost->get_values();
            forAll(bouCoeffs, fi)
            {
                data[fi] = static_cast<ValueType>(bouCoeffs[fi]);
            }
            bouCoeffsGPU_[i] = Vector::create(exec_, gko::dim<2>(size, 1));
            bouCoeffsGPU_[i]->copy_from(bouCoeffsHost.get());
        }
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

template<typename ValueType>
Foam::OGL::GPUHaloExchange<ValueType>::GPUHaloExchange
(
    std::shared_ptr<const gko::Executor> exec,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const direction cmpt
)
:
    exec_(exec),
    cpuExec_(OGLExecutor::instance().cpuExecutor()),
    matrix_(matrix),
    interfaces_(interfaces),
    interfaceBouCoeffs_(interfaceBouCoeffs),
    cmpt_(cmpt),
    cudaAwareMPI_(detectCudaAwareMPI()),
    totalBoundaryFaces_(0)
{
    initializeBuffers();

    if (OGLExecutor::instance().debug() >= 1)
    {
        Info<< "GPUHaloExchange: initialized with "
            << totalBoundaryFaces_ << " boundary faces, "
            << "CUDA-aware MPI: " << (cudaAwareMPI_ ? "yes" : "no")
            << endl;
    }
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

template<typename ValueType>
void Foam::OGL::GPUHaloExchange<ValueType>::gather(const Vector* x) const
{
    // For each interface, gather x[faceCells[i]] into sendBuffer
    // This is a gather operation: sendBuffer[j] = x[faceCells[j]]

    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const label size = faceCellsGPU_[i].get_size();
            if (size == 0) continue;

            // Get pointers for gather operation
            const ValueType* xData = x->get_const_values();
            ValueType* sendData = sendBuffersGPU_[i]->get_values();
            const int* indices = faceCellsGPU_[i].get_const_data();

            // Copy to host for gather (Ginkgo doesn't have direct gather kernel)
            // This is still more efficient than copying the entire vector
            auto xHost = Vector::create(cpuExec_, x->get_size());
            xHost->copy_from(x);

            auto sendHost = Vector::create(cpuExec_, gko::dim<2>(size, 1));

            const ValueType* xHostData = xHost->get_const_values();
            ValueType* sendHostData = sendHost->get_values();

            // Copy face cell indices to host for gather
            auto indicesHost = faceCellsGPU_[i];
            indicesHost.set_executor(cpuExec_);
            const int* indicesHostData = indicesHost.get_const_data();

            // Perform gather on host
            for (label j = 0; j < size; j++)
            {
                sendHostData[j] = xHostData[indicesHostData[j]];
            }

            // Copy gathered data back to GPU buffer
            sendBuffersGPU_[i]->copy_from(sendHost.get());
        }
    }
}


template<typename ValueType>
void Foam::OGL::GPUHaloExchange<ValueType>::exchange() const
{
    if (!Pstream::parRun())
    {
        // Serial run - no exchange needed, but may have cyclic boundaries
        // For cyclic, the send buffer becomes the receive buffer
        forAll(interfaces_, i)
        {
            if (interfaces_.set(i))
            {
                recvBuffersGPU_[i]->copy_from(sendBuffersGPU_[i].get());
            }
        }
        return;
    }

    // Parallel exchange
    if (cudaAwareMPI_)
    {
        // Direct GPU-to-GPU transfer using CUDA-aware MPI
        // Note: This would use MPI_Isend/MPI_Irecv with GPU pointers
        // For now, fall back to staged transfer as Ginkgo doesn't expose
        // raw pointers in a portable way for all backends

        // TODO: Implement true CUDA-aware MPI path when Ginkgo supports it
        // For now, use the staged path
    }

    // Staged transfer through host memory
    // Step 1: Copy send buffers from GPU to host
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const label size = sendBuffersHost_[i].size();
            if (size == 0) continue;

            auto sendHost = Vector::create(cpuExec_, gko::dim<2>(size, 1));
            sendHost->copy_from(sendBuffersGPU_[i].get());

            const ValueType* data = sendHost->get_const_values();
            forAll(sendBuffersHost_[i], j)
            {
                sendBuffersHost_[i][j] = static_cast<scalar>(data[j]);
            }
        }
    }

    // Step 2: Use OpenFOAM's interface mechanism for MPI exchange
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const lduInterfaceField& intf = interfaces_[i];

            // Initiate send
            intf.initInterfaceMatrixUpdate
            (
                sendBuffersHost_[i],
                interfaces_,
                Pstream::defaultCommsType
            );
        }
    }

    // Finalize communication
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const lduInterfaceField& intf = interfaces_[i];

            // Complete receive
            intf.updateInterfaceMatrix
            (
                recvBuffersHost_[i],
                sendBuffersHost_[i],
                interfaces_,
                Pstream::defaultCommsType
            );
        }
    }

    // Step 3: Copy receive buffers from host to GPU
    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const label size = recvBuffersHost_[i].size();
            if (size == 0) continue;

            auto recvHost = Vector::create(cpuExec_, gko::dim<2>(size, 1));
            ValueType* data = recvHost->get_values();
            forAll(recvBuffersHost_[i], j)
            {
                data[j] = static_cast<ValueType>(recvBuffersHost_[i][j]);
            }

            recvBuffersGPU_[i]->copy_from(recvHost.get());
        }
    }
}


template<typename ValueType>
void Foam::OGL::GPUHaloExchange<ValueType>::scatter(Vector* y) const
{
    // For each interface, add bouCoeffs[j] * recvBuffer[j] to y[faceCells[j]]
    // This is a scatter-add operation

    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const label size = faceCellsGPU_[i].get_size();
            if (size == 0) continue;

            // Copy y to host for scatter-add
            auto yHost = Vector::create(cpuExec_, y->get_size());
            yHost->copy_from(y);

            auto recvHost = Vector::create(cpuExec_, gko::dim<2>(size, 1));
            recvHost->copy_from(recvBuffersGPU_[i].get());

            auto bouCoeffsHost = Vector::create(cpuExec_, gko::dim<2>(size, 1));
            bouCoeffsHost->copy_from(bouCoeffsGPU_[i].get());

            // Copy indices to host
            auto indicesHost = faceCellsGPU_[i];
            indicesHost.set_executor(cpuExec_);

            ValueType* yData = yHost->get_values();
            const ValueType* recvData = recvHost->get_const_values();
            const ValueType* coeffData = bouCoeffsHost->get_const_values();
            const int* indices = indicesHost.get_const_data();

            // Perform scatter-add on host
            // y[faceCells[j]] += bouCoeffs[j] * recvBuffer[j]
            for (label j = 0; j < size; j++)
            {
                yData[indices[j]] += coeffData[j] * recvData[j];
            }

            // Copy result back to GPU
            y->copy_from(yHost.get());
        }
    }
}


template<typename ValueType>
void Foam::OGL::GPUHaloExchange<ValueType>::apply
(
    const Vector* x,
    Vector* y
) const
{
    if (interfaces_.size() == 0 || totalBoundaryFaces_ == 0)
    {
        return;
    }

    gather(x);
    exchange();
    scatter(y);
}


template<typename ValueType>
void Foam::OGL::GPUHaloExchange<ValueType>::report() const
{
    Info<< "GPUHaloExchange statistics:" << nl
        << "  Number of interfaces: " << interfaces_.size() << nl
        << "  Total boundary faces: " << totalBoundaryFaces_ << nl
        << "  CUDA-aware MPI: " << (cudaAwareMPI_ ? "enabled" : "disabled") << nl;

    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            Info<< "  Interface " << i << ": "
                << faceCellsGPU_[i].get_size() << " faces" << nl;
        }
    }
    Info<< endl;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Explicit instantiation
template class Foam::OGL::GPUHaloExchange<float>;
template class Foam::OGL::GPUHaloExchange<double>;


// ************************************************************************* //
