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

Description
    CUDA kernels for GPU-resident halo exchange gather/scatter operations.

    These replace the previous host-roundtrip implementation where the
    entire solution vector was copied GPU->CPU for a simple indexed
    gather/scatter, then copied back. Now only O(boundary) data moves
    through the kernels, and the full vector stays on GPU.

\*---------------------------------------------------------------------------*/

#include "HaloKernels.h"
#include <cuda_runtime.h>

// * * * * * * * * * * * * * * CUDA Kernels * * * * * * * * * * * * * * * * * //

static const int BLOCK_SIZE = 256;

template<typename T>
__global__ void gatherKernel
(
    const T* __restrict__ x,
    T* __restrict__ sendBuf,
    const int* __restrict__ indices,
    int n
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n)
    {
        sendBuf[j] = x[indices[j]];
    }
}


template<typename T>
__global__ void scatterAddKernel
(
    T* y,
    const T* __restrict__ recvBuf,
    const T* __restrict__ coeffs,
    const int* __restrict__ indices,
    int n
)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < n)
    {
        // atomicAdd is needed because cells at processor boundary corners
        // could appear in multiple interfaces
        atomicAdd(&y[indices[j]], coeffs[j] * recvBuf[j]);
    }
}


// * * * * * * * * * * * * * C API Functions  * * * * * * * * * * * * * * * * //

extern "C" {

void haloGatherFloat
(
    const float* x,
    float* sendBuf,
    const int* indices,
    int n
)
{
    if (n <= 0) return;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gatherKernel<<<grid, BLOCK_SIZE>>>(x, sendBuf, indices, n);
}


void haloGatherDouble
(
    const double* x,
    double* sendBuf,
    const int* indices,
    int n
)
{
    if (n <= 0) return;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    gatherKernel<<<grid, BLOCK_SIZE>>>(x, sendBuf, indices, n);
}


void haloScatterAddFloat
(
    float* y,
    const float* recvBuf,
    const float* coeffs,
    const int* indices,
    int n
)
{
    if (n <= 0) return;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scatterAddKernel<<<grid, BLOCK_SIZE>>>(y, recvBuf, coeffs, indices, n);
}


void haloScatterAddDouble
(
    double* y,
    const double* recvBuf,
    const double* coeffs,
    const int* indices,
    int n
)
{
    if (n <= 0) return;
    int grid = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    scatterAddKernel<<<grid, BLOCK_SIZE>>>(y, recvBuf, coeffs, indices, n);
}


void haloCudaSync()
{
    cudaDeviceSynchronize();
}


void haloCopyHostToDevice(void* dst, const void* src, size_t bytes)
{
    if (bytes > 0)
    {
        cudaMemcpy(dst, src, bytes, cudaMemcpyHostToDevice);
    }
}


void haloGetGpuMemInfo(size_t* gpuFree, size_t* gpuTotal)
{
    cudaMemGetInfo(gpuFree, gpuTotal);
}


} // extern "C"

// ************************************************************************* //
