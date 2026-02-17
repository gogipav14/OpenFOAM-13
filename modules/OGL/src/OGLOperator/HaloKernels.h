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

    - Gather:      sendBuf[j] = x[indices[j]]
    - Scatter-add: y[indices[j]] += coeffs[j] * recvBuf[j]

    Eliminates the full-vector GPU<->CPU round-trips that were previously
    required because Ginkgo lacks native gather/scatter-add kernels.

\*---------------------------------------------------------------------------*/

#ifndef HaloKernels_H
#define HaloKernels_H

#ifdef __cplusplus
extern "C" {
#endif

// Gather: sendBuf[j] = x[indices[j]] for j in [0, n)
void haloGatherFloat
(
    const float* x,
    float* sendBuf,
    const int* indices,
    int n
);

void haloGatherDouble
(
    const double* x,
    double* sendBuf,
    const int* indices,
    int n
);

// Scatter-add: y[indices[j]] += coeffs[j] * recvBuf[j] for j in [0, n)
void haloScatterAddFloat
(
    float* y,
    const float* recvBuf,
    const float* coeffs,
    const int* indices,
    int n
);

void haloScatterAddDouble
(
    double* y,
    const double* recvBuf,
    const double* coeffs,
    const int* indices,
    int n
);

// Synchronize all CUDA streams (call before MPI with GPU pointers)
void haloCudaSync();

// Host-to-device memcpy (for in-place GPU CSR value update)
void haloCopyHostToDevice(void* dst, const void* src, size_t bytes);

// Query GPU memory: sets *free and *total in bytes
void haloGetGpuMemInfo(size_t* gpuFree, size_t* gpuTotal);

#ifdef __cplusplus
}
#endif

#endif

// ************************************************************************* //
