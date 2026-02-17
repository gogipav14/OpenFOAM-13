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
    CUDA implementation of DCT-based Laplacian preconditioner for Neumann BCs.

    Uses even extension + cuFFT to implement DCT-II diagonalization of the
    cell-centered FV Laplacian with Neumann (zeroGradient) boundary conditions.

    For Neumann BCs on cell-centered unknowns, the eigenvectors are cosines
    (DCT-II basis), NOT complex exponentials (DFT basis). Using DFT for a
    Neumann problem applies the inverse in the wrong eigenbasis.

    Implementation: even-extend the input in each direction (N -> 2N),
    apply FFT on the extended grid, scale by Neumann eigenvalues, IFFT,
    extract the original block. This emulates DCT using cuFFT.

    Neumann eigenvalues (cell-centered FV):
        lambda(i,j,k) = -2*coeffX*(1 - cos(pi*i/nx))
                       - 2*coeffY*(1 - cos(pi*j/ny))
                       - 2*coeffZ*(1 - cos(pi*k/nz))

    Note: pi*k/N (Neumann/DCT) vs 2*pi*k/N (periodic/DFT).

    Memory layout:
        OpenFOAM: cellId = ix + iy*nx + iz*nx*ny (ix fastest)
        Extended grid: 2nz * 2ny * 2nx (or nz if nz==1)
        cuFFT: n = {ez, ey, ex} where ex = 2*nx, etc.

    Status: Eigenmode self-test passes (100% match), but the preconditioner
    does not yet improve CG convergence on real pressure systems. The likely
    cause is mismatch between the idealized Neumann Laplacian and the actual
    FV pressure operator (variable rAU, boundary stencils). Parked for now;
    Block Jacobi is used as the production preconditioner.

\*---------------------------------------------------------------------------*/

#include <cufft.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

#include "FFTKernels.h"

// -------------------------------------------------------------------------
// Internal state structure
// -------------------------------------------------------------------------

struct FFTPrecondState
{
    int nx, ny, nz;
    int totalCells;          // nx * ny * nz

    // Extended dimensions (2*nx, 2*ny, 2*nz; nz stays 1 if nz==1)
    int ex, ey, ez;
    int extendedCells;       // ex * ey * ez
    int complexSize;         // ez * ey * (ex/2 + 1)

    int useFloat;

    // cuFFT plans for the EXTENDED grid
    cufftHandle planR2C_f;
    cufftHandle planC2R_f;
    cufftHandle planD2Z;
    cufftHandle planZ2D;

    // Inverse eigenvalues (Neumann/DCT) on GPU, size complexSize
    float*  invEigFloat;
    double* invEigDouble;

    // Work buffers for the extended grid
    void* complexBuf;        // size complexSize * sizeof(complex)
    void* realBuf;           // size extendedCells * sizeof(real)
};


// -------------------------------------------------------------------------
// CUDA kernels
// -------------------------------------------------------------------------

// Even-extend a 3D array: [nz][ny][nx] -> [ez][ey][ex]
// Half-sample symmetric (HSS) extension for DCT-II:
//   ext[i] = orig[i]         for i < N
//   ext[2N-1-i] = orig[i]   for i < N
template<typename T>
__global__ void evenExtend3D(
    const T* __restrict__ input,
    T* __restrict__ output,
    int nx, int ny, int nz,
    int ex, int ey, int ez
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int extTotal = ex * ey * ez;
    if (idx >= extTotal) return;

    // Decompose extended index
    int eix = idx % ex;
    int eiy = (idx / ex) % ey;
    int eiz = idx / (ex * ey);

    // Map extended coordinate to original via reflection
    int oix = (eix < nx) ? eix : (2 * nx - 1 - eix);
    int oiy = (eiy < ny) ? eiy : (2 * ny - 1 - eiy);
    int oiz;
    if (nz == 1)
        oiz = 0;  // No extension in z for 2D
    else
        oiz = (eiz < nz) ? eiz : (2 * nz - 1 - eiz);

    output[idx] = input[oiz * nx * ny + oiy * nx + oix];
}

// Extract the original block from extended IFFT output
template<typename T>
__global__ void extractOriginal3D(
    const T* __restrict__ extended,
    T* __restrict__ output,
    int nx, int ny, int nz,
    int ex, int ey
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nz;
    if (idx >= total) return;

    int oix = idx % nx;
    int oiy = (idx / nx) % ny;
    int oiz = idx / (nx * ny);

    output[idx] = extended[oiz * ex * ey + oiy * ex + oix];
}

// Compute inverse DCT/Neumann eigenvalues (float)
// These use 2*pi*k/(2N) = pi*k/N because the even extension
// maps DFT frequencies on [0, 2N) to DCT-II frequencies on [0, N).
__global__ void computeInvEigenvaluesF(
    float* invEig,
    int nx, int ny, int nz,
    int ex, int ey, int ez, int exHalf,
    float coeffX, float coeffY, float coeffZ
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ez * ey * exHalf;
    if (idx >= total) return;

    int kx = idx % exHalf;
    int ky = (idx / exHalf) % ey;
    int kz = idx / (exHalf * ey);

    // Extended grid eigenvalues: 2*pi*k/(2N) = pi*k/N
    float pi2 = 6.283185307179586f;
    float lamX = -2.0f * coeffX * (1.0f - cosf(pi2 * (float)kx / (float)ex));
    float lamY = -2.0f * coeffY * (1.0f - cosf(pi2 * (float)ky / (float)ey));
    float lamZ = (nz > 1)
        ? (-2.0f * coeffZ * (1.0f - cosf(pi2 * (float)kz / (float)ez)))
        : 0.0f;

    float lam = lamX + lamY + lamZ;

    invEig[idx] = (fabsf(lam) > 1.0e-12f) ? (1.0f / lam) : 0.0f;
}

// Compute inverse DCT/Neumann eigenvalues (double)
__global__ void computeInvEigenvaluesD(
    double* invEig,
    int nx, int ny, int nz,
    int ex, int ey, int ez, int exHalf,
    double coeffX, double coeffY, double coeffZ
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ez * ey * exHalf;
    if (idx >= total) return;

    int kx = idx % exHalf;
    int ky = (idx / exHalf) % ey;
    int kz = idx / (exHalf * ey);

    double pi2 = 6.283185307179586;
    double lamX = -2.0 * coeffX * (1.0 - cos(pi2 * (double)kx / (double)ex));
    double lamY = -2.0 * coeffY * (1.0 - cos(pi2 * (double)ky / (double)ey));
    double lamZ = (nz > 1)
        ? (-2.0 * coeffZ * (1.0 - cos(pi2 * (double)kz / (double)ez)))
        : 0.0;

    double lam = lamX + lamY + lamZ;

    invEig[idx] = (fabs(lam) > 1.0e-15) ? (1.0 / lam) : 0.0;
}

// Scale complex array by real inverse eigenvalues (float)
__global__ void scaleComplexF(
    cufftComplex* data,
    const float* invEig,
    int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    data[idx].x *= invEig[idx];
    data[idx].y *= invEig[idx];
}

// Scale complex array by real inverse eigenvalues (double)
__global__ void scaleComplexD(
    cufftDoubleComplex* data,
    const double* invEig,
    int n
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    data[idx].x *= invEig[idx];
    data[idx].y *= invEig[idx];
}

// Scale real array (float)
__global__ void scaleRealF(float* data, float scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] *= scale;
}

// Scale real array (double)
__global__ void scaleRealD(double* data, double scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] *= scale;
}


// -------------------------------------------------------------------------
// API implementation
// -------------------------------------------------------------------------

extern "C"
FFTPrecondHandle fftPrecondCreate(
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    int useFloat
)
{
    FFTPrecondState* s = new FFTPrecondState();
    s->nx = nx;
    s->ny = ny;
    s->nz = nz;
    s->totalCells = nx * ny * nz;

    // Even extension dimensions: double each active dimension
    s->ex = 2 * nx;
    s->ey = 2 * ny;
    s->ez = (nz > 1) ? (2 * nz) : 1;  // Don't extend z for 2D (nz=1)
    s->extendedCells = s->ex * s->ey * s->ez;

    // R2C halves the last (fastest) dimension
    int exHalf = s->ex / 2 + 1;  // = nx + 1
    s->complexSize = s->ez * s->ey * exHalf;

    s->useFloat = useFloat;
    s->invEigFloat = nullptr;
    s->invEigDouble = nullptr;
    s->complexBuf = nullptr;
    s->realBuf = nullptr;
    s->planR2C_f = 0;
    s->planC2R_f = 0;
    s->planD2Z = 0;
    s->planZ2D = 0;

    // cuFFT dimensions for the extended grid
    int n[3] = {s->ez, s->ey, s->ex};

    int threads = 256;

    fprintf(stderr, "FFTPrecond DCT: nx=%d ny=%d nz=%d -> extended %dx%dx%d"
            " (complexSize=%d)\n",
            nx, ny, nz, s->ex, s->ey, s->ez, s->complexSize);

    if (useFloat)
    {
        cufftResult res;

        res = cufftPlanMany(
            &s->planR2C_f, 3, n,
            nullptr, 1, 0,
            nullptr, 1, 0,
            CUFFT_R2C, 1
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "FFTPrecond: cufftPlan R2C failed: %d\n", res);
            delete s;
            return nullptr;
        }

        res = cufftPlanMany(
            &s->planC2R_f, 3, n,
            nullptr, 1, 0,
            nullptr, 1, 0,
            CUFFT_C2R, 1
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "FFTPrecond: cufftPlan C2R failed: %d\n", res);
            cufftDestroy(s->planR2C_f);
            delete s;
            return nullptr;
        }

        // Allocate GPU buffers for extended grid
        cudaMalloc(&s->invEigFloat, s->complexSize * sizeof(float));
        cudaMalloc(&s->complexBuf, s->complexSize * sizeof(cufftComplex));
        cudaMalloc(&s->realBuf, s->extendedCells * sizeof(float));

        // Compute initial eigenvalues using geometric coefficients
        float geoCoeffX = (float)(dy * dz / dx);
        float geoCoeffY = (float)(dx * dz / dy);
        float geoCoeffZ = (float)(dx * dy / dz);
        int blocks = (s->complexSize + threads - 1) / threads;
        computeInvEigenvaluesF<<<blocks, threads>>>(
            s->invEigFloat, nx, ny, nz,
            s->ex, s->ey, s->ez, exHalf,
            geoCoeffX, geoCoeffY, geoCoeffZ
        );
        cudaDeviceSynchronize();
    }
    else
    {
        cufftResult res;
        res = cufftPlanMany(
            &s->planD2Z, 3, n,
            nullptr, 1, 0,
            nullptr, 1, 0,
            CUFFT_D2Z, 1
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "FFTPrecond: cufftPlan D2Z failed: %d\n", res);
            delete s;
            return nullptr;
        }

        res = cufftPlanMany(
            &s->planZ2D, 3, n,
            nullptr, 1, 0,
            nullptr, 1, 0,
            CUFFT_Z2D, 1
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "FFTPrecond: cufftPlan Z2D failed: %d\n", res);
            cufftDestroy(s->planD2Z);
            delete s;
            return nullptr;
        }

        cudaMalloc(&s->invEigDouble, s->complexSize * sizeof(double));
        cudaMalloc(&s->complexBuf,
            s->complexSize * sizeof(cufftDoubleComplex));
        cudaMalloc(&s->realBuf, s->extendedCells * sizeof(double));

        double geoCoeffX = dy * dz / dx;
        double geoCoeffY = dx * dz / dy;
        double geoCoeffZ = dx * dy / dz;
        int blocks = (s->complexSize + threads - 1) / threads;
        computeInvEigenvaluesD<<<blocks, threads>>>(
            s->invEigDouble, nx, ny, nz,
            s->ex, s->ey, s->ez, s->ex / 2 + 1,
            geoCoeffX, geoCoeffY, geoCoeffZ
        );
        cudaDeviceSynchronize();
    }

    return s;
}


extern "C"
void fftPrecondDestroy(FFTPrecondHandle h)
{
    if (!h) return;

    if (h->useFloat)
    {
        if (h->planR2C_f) cufftDestroy(h->planR2C_f);
        if (h->planC2R_f) cufftDestroy(h->planC2R_f);
        if (h->invEigFloat) cudaFree(h->invEigFloat);
    }
    else
    {
        if (h->planD2Z) cufftDestroy(h->planD2Z);
        if (h->planZ2D) cufftDestroy(h->planZ2D);
        if (h->invEigDouble) cudaFree(h->invEigDouble);
    }

    if (h->complexBuf) cudaFree(h->complexBuf);
    if (h->realBuf) cudaFree(h->realBuf);

    delete h;
}


extern "C"
void fftPrecondUpdateCoeffs(
    FFTPrecondHandle h,
    double coeffX, double coeffY, double coeffZ
)
{
    if (!h) return;

    int exHalf = h->ex / 2 + 1;
    int threads = 256;
    int blocks = (h->complexSize + threads - 1) / threads;

    fprintf(stderr, "FFTPrecond DCT: updating coeffs: coeffX=%.6e"
            " coeffY=%.6e coeffZ=%.6e\n", coeffX, coeffY, coeffZ);

    if (h->useFloat)
    {
        computeInvEigenvaluesF<<<blocks, threads>>>(
            h->invEigFloat, h->nx, h->ny, h->nz,
            h->ex, h->ey, h->ez, exHalf,
            (float)coeffX, (float)coeffY, (float)coeffZ
        );
    }
    else
    {
        computeInvEigenvaluesD<<<blocks, threads>>>(
            h->invEigDouble, h->nx, h->ny, h->nz,
            h->ex, h->ey, h->ez, exHalf,
            coeffX, coeffY, coeffZ
        );
    }
    cudaDeviceSynchronize();
}


extern "C"
void fftPrecondApplyFloat(
    FFTPrecondHandle h,
    const float* b_ptr,
    float* x_ptr,
    int n
)
{
    if (!h || n != h->totalCells) return;

    int threads = 256;
    float* ext = (float*)h->realBuf;
    cufftComplex* freq = (cufftComplex*)h->complexBuf;

    // 1. Even-extend input: [nz][ny][nx] -> [ez][ey][ex]
    int extBlocks = (h->extendedCells + threads - 1) / threads;
    evenExtend3D<float><<<extBlocks, threads>>>(
        b_ptr, ext,
        h->nx, h->ny, h->nz,
        h->ex, h->ey, h->ez
    );

    // 2. Forward R2C FFT on extended grid
    cufftExecR2C(h->planR2C_f, ext, freq);

    // 3. Scale by inverse Neumann eigenvalues
    int cblocks = (h->complexSize + threads - 1) / threads;
    scaleComplexF<<<cblocks, threads>>>(freq, h->invEigFloat, h->complexSize);

    // 4. Inverse C2R FFT -> extended real buffer
    cufftExecC2R(h->planC2R_f, freq, ext);

    // 5. Extract original block and normalize by 1/extendedCells
    int origBlocks = (n + threads - 1) / threads;
    extractOriginal3D<float><<<origBlocks, threads>>>(
        ext, x_ptr,
        h->nx, h->ny, h->nz,
        h->ex, h->ey
    );

    float scale = 1.0f / (float)h->extendedCells;
    scaleRealF<<<origBlocks, threads>>>(x_ptr, scale, n);
}


extern "C"
void fftPrecondApplyDouble(
    FFTPrecondHandle h,
    const double* b_ptr,
    double* x_ptr,
    int n
)
{
    if (!h || n != h->totalCells) return;

    int threads = 256;
    double* ext = (double*)h->realBuf;
    cufftDoubleComplex* freq = (cufftDoubleComplex*)h->complexBuf;

    // 1. Even-extend input
    int extBlocks = (h->extendedCells + threads - 1) / threads;
    evenExtend3D<double><<<extBlocks, threads>>>(
        b_ptr, ext,
        h->nx, h->ny, h->nz,
        h->ex, h->ey, h->ez
    );

    // 2. Forward D2Z FFT on extended grid
    cufftExecD2Z(h->planD2Z, ext, freq);

    // 3. Scale by inverse Neumann eigenvalues
    int cblocks = (h->complexSize + threads - 1) / threads;
    scaleComplexD<<<cblocks, threads>>>(freq, h->invEigDouble, h->complexSize);

    // 4. Inverse Z2D FFT
    cufftExecZ2D(h->planZ2D, freq, ext);

    // 5. Extract original block and normalize
    int origBlocks = (n + threads - 1) / threads;
    extractOriginal3D<double><<<origBlocks, threads>>>(
        ext, x_ptr,
        h->nx, h->ny, h->nz,
        h->ex, h->ey
    );

    double scale = 1.0 / (double)h->extendedCells;
    scaleRealD<<<origBlocks, threads>>>(x_ptr, scale, n);
}
