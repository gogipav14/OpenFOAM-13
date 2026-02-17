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
#include "IPstream.H"
#include "OPstream.H"
#include "HaloKernels.h"

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
            auto faceCellsView = gko::array<int>::view(
                cpuExec_,
                size,
                const_cast<int*>(
                    reinterpret_cast<const int*>(faceCells.cdata())
                )
            );
            faceCellsGPU_[i] = gko::array<int>(exec_, faceCellsView);

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

            // Track GPU memory for this interface:
            // faceCells(int) + send+recv buffers + bouCoeffs (ValueType each)
            OGLExecutor::instance().trackAllocation(
                size * sizeof(int)
              + 3 * size * sizeof(ValueType)
            );
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
    // Runs entirely on GPU via CUDA kernel — no host round-trip needed

    // Ensure any prior Ginkgo operations on x are complete
    exec_->synchronize();

    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const label size = faceCellsGPU_[i].get_size();
            if (size == 0) continue;

            const ValueType* xData = x->get_const_values();
            ValueType* sendData = sendBuffersGPU_[i]->get_values();
            const int* indices = faceCellsGPU_[i].get_const_data();

            if constexpr (std::is_same<ValueType, float>::value)
            {
                haloGatherFloat(xData, sendData, indices, size);
            }
            else
            {
                haloGatherDouble(xData, sendData, indices, size);
            }
        }
    }

    // Ensure all gather kernels complete before exchange
    haloCudaSync();
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

    // Parallel exchange: choose between CUDA-aware MPI (direct GPU pointers)
    // and staged host transfer based on runtime detection
    if (cudaAwareMPI_)
    {
        // Direct GPU-to-GPU transfer via CUDA-aware MPI.
        // GPU send buffers are already populated by gather().
        // Pass device pointers directly to MPI — the CUDA-aware MPI
        // implementation handles DMA/GPUDirect RDMA internally.
        label nReqsBefore = UPstream::nRequests();

        forAll(interfaces_, i)
        {
            if (interfaces_.set(i))
            {
                const label size = faceCellsGPU_[i].get_size();
                if (size == 0) continue;

                if (isA<processorLduInterfaceField>(interfaces_[i]))
                {
                    const processorLduInterfaceField& procIntf =
                        refCast<const processorLduInterfaceField>
                        (interfaces_[i]);

                    const std::streamsize byteSize =
                        size * sizeof(ValueType);

                    // Post non-blocking receive into GPU buffer
                    UIPstream::read
                    (
                        Pstream::commsTypes::nonBlocking,
                        procIntf.neighbProcNo(),
                        reinterpret_cast<char*>
                        (
                            recvBuffersGPU_[i]->get_values()
                        ),
                        byteSize,
                        UPstream::msgType(),
                        procIntf.comm()
                    );

                    // Post non-blocking send from GPU buffer
                    UOPstream::write
                    (
                        Pstream::commsTypes::nonBlocking,
                        procIntf.neighbProcNo(),
                        reinterpret_cast<const char*>
                        (
                            sendBuffersGPU_[i]->get_const_values()
                        ),
                        byteSize,
                        UPstream::msgType(),
                        procIntf.comm()
                    );
                }
                else
                {
                    // Non-processor interface (e.g. cyclic within same proc)
                    recvBuffersGPU_[i]->copy_from(sendBuffersGPU_[i].get());
                }
            }
        }

        UPstream::waitRequests(nReqsBefore);
    }
    else
    {
        // Staged transfer through host memory
        // Step 1: Copy send buffers from GPU to host
        forAll(interfaces_, i)
        {
            if (interfaces_.set(i))
            {
                const label size = sendBuffersHost_[i].size();
                if (size == 0) continue;

                auto sendHost =
                    Vector::create(cpuExec_, gko::dim<2>(size, 1));
                sendHost->copy_from(sendBuffersGPU_[i].get());

                const ValueType* data = sendHost->get_const_values();
                forAll(sendBuffersHost_[i], j)
                {
                    sendBuffersHost_[i][j] =
                        static_cast<scalar>(data[j]);
                }
            }
        }

        // Step 2: Exchange via MPI using host buffers
        {
            label nReqsBefore = UPstream::nRequests();

            forAll(interfaces_, i)
            {
                if (interfaces_.set(i))
                {
                    if (isA<processorLduInterfaceField>(interfaces_[i]))
                    {
                        const processorLduInterfaceField& procIntf =
                            refCast<const processorLduInterfaceField>
                            (interfaces_[i]);

                        UIPstream::read
                        (
                            Pstream::commsTypes::nonBlocking,
                            procIntf.neighbProcNo(),
                            reinterpret_cast<char*>
                            (
                                recvBuffersHost_[i].begin()
                            ),
                            recvBuffersHost_[i].byteSize(),
                            UPstream::msgType(),
                            procIntf.comm()
                        );

                        UOPstream::write
                        (
                            Pstream::commsTypes::nonBlocking,
                            procIntf.neighbProcNo(),
                            reinterpret_cast<const char*>
                            (
                                sendBuffersHost_[i].begin()
                            ),
                            sendBuffersHost_[i].byteSize(),
                            UPstream::msgType(),
                            procIntf.comm()
                        );
                    }
                    else
                    {
                        recvBuffersHost_[i] = sendBuffersHost_[i];
                    }
                }
            }

            UPstream::waitRequests(nReqsBefore);
        }

        // Step 3: Copy receive buffers from host to GPU
        forAll(interfaces_, i)
        {
            if (interfaces_.set(i))
            {
                const label size = recvBuffersHost_[i].size();
                if (size == 0) continue;

                auto recvHost =
                    Vector::create(cpuExec_, gko::dim<2>(size, 1));
                ValueType* data = recvHost->get_values();
                forAll(recvBuffersHost_[i], j)
                {
                    data[j] =
                        static_cast<ValueType>(recvBuffersHost_[i][j]);
                }

                recvBuffersGPU_[i]->copy_from(recvHost.get());
            }
        }
    }
}


template<typename ValueType>
void Foam::OGL::GPUHaloExchange<ValueType>::scatter(Vector* y) const
{
    // For each interface, add bouCoeffs[j] * recvBuffer[j] to y[faceCells[j]]
    // This is a scatter-add operation
    // Runs entirely on GPU via CUDA kernel — no host round-trip needed

    forAll(interfaces_, i)
    {
        if (interfaces_.set(i))
        {
            const label size = faceCellsGPU_[i].get_size();
            if (size == 0) continue;

            ValueType* yData = y->get_values();
            const ValueType* recvData =
                recvBuffersGPU_[i]->get_const_values();
            const ValueType* coeffData =
                bouCoeffsGPU_[i]->get_const_values();
            const int* indices = faceCellsGPU_[i].get_const_data();

            if constexpr (std::is_same<ValueType, float>::value)
            {
                haloScatterAddFloat
                (
                    yData, recvData, coeffData, indices, size
                );
            }
            else
            {
                haloScatterAddDouble
                (
                    yData, recvData, coeffData, indices, size
                );
            }
        }
    }

    // Ensure all scatter-add kernels complete before Ginkgo uses y
    haloCudaSync();
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
