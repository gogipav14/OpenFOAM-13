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
    CUDA implementation of FFT-based Laplacian preconditioner.

    Uses cuFFT R2C/C2R transforms for O(N log N) approximate inverse
    of the discrete Laplacian on uniform Cartesian grids.

    Mathematical basis (periodic discrete Laplacian eigenvalues):
        lambda(i,j,k) = 2/dx^2 * (cos(2*pi*i/nx) - 1)
                       + 2/dy^2 * (cos(2*pi*j/ny) - 1)
                       + 2/dz^2 * (cos(2*pi*k/nz) - 1)

    Preconditioner application:
        x = IFFT( FFT(b) ./ lambda ) * (1/N)

    Reference: moljax paper, Section 3.4 (FFT-Diagonalized Operators)

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
    int complexSize;         // nx * ny * (nz/2 + 1)
    int useFloat;

    // cuFFT plans
    cufftHandle planR2C_f;   // float:  R2C forward
    cufftHandle planC2R_f;   // float:  C2R inverse
    cufftHandle planD2Z;     // double: D2Z forward
    cufftHandle planZ2D;     // double: Z2D inverse

    // Inverse eigenvalues on GPU (1/lambda, with zero mode set to 0)
    float*  invEigFloat;
    double* invEigDouble;

    // Work buffers on GPU
    // For float:  cufftComplex (float2)  of size complexSize
    // For double: cufftDoubleComplex (double2) of size complexSize
    void* complexBuf;

    // Padded real buffer for R2C (needs nx * ny * 2*(nz/2+1) reals)
    void* paddedRealBuf;
};


// -------------------------------------------------------------------------
// CUDA kernels
// -------------------------------------------------------------------------

// Compute inverse eigenvalues for periodic Laplacian (float)
__global__ void computeInvEigenvaluesF(
    float* invEig,
    int nx, int ny, int nzHalf,  // nzHalf = nz/2 + 1
    int nz,
    float dx, float dy, float dz
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nzHalf;
    if (idx >= total) return;

    // Decompose to frequency indices (cuFFT R2C output layout)
    // Data layout: [ix][iy][iz] with iz varying fastest, size nzHalf
    int iz = idx % nzHalf;
    int iy = (idx / nzHalf) % ny;
    int ix = idx / (nzHalf * ny);

    float pi2 = 6.283185307179586f;  // 2*pi

    float lamX = 2.0f / (dx * dx) * (cosf(pi2 * (float)ix / (float)nx) - 1.0f);
    float lamY = 2.0f / (dy * dy) * (cosf(pi2 * (float)iy / (float)ny) - 1.0f);
    float lamZ = 2.0f / (dz * dz) * (cosf(pi2 * (float)iz / (float)nz) - 1.0f);

    float lam = lamX + lamY + lamZ;

    // Inverse eigenvalue; zero mode (lam=0) maps to 0 (projects out constant)
    invEig[idx] = (fabsf(lam) > 1.0e-12f) ? (1.0f / lam) : 0.0f;
}

// Compute inverse eigenvalues for periodic Laplacian (double)
__global__ void computeInvEigenvaluesD(
    double* invEig,
    int nx, int ny, int nzHalf,
    int nz,
    double dx, double dy, double dz
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nx * ny * nzHalf;
    if (idx >= total) return;

    int iz = idx % nzHalf;
    int iy = (idx / nzHalf) % ny;
    int ix = idx / (nzHalf * ny);

    double pi2 = 6.283185307179586;

    double lamX = 2.0 / (dx * dx) * (cos(pi2 * (double)ix / (double)nx) - 1.0);
    double lamY = 2.0 / (dy * dy) * (cos(pi2 * (double)iy / (double)ny) - 1.0);
    double lamZ = 2.0 / (dz * dz) * (cos(pi2 * (double)iz / (double)nz) - 1.0);

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

// Scale real array by 1/N for cuFFT normalization (float)
__global__ void scaleRealF(float* data, float scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] *= scale;
}

// Scale real array by 1/N for cuFFT normalization (double)
__global__ void scaleRealD(double* data, double scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    data[idx] *= scale;
}

// Copy with padding for R2C (float): from (nx*ny*nz) to (nx*ny*2*(nz/2+1))
// cuFFT R2C in-place requires the real array to be padded in the last dim
__global__ void copyToPaddedF(
    float* padded,
    const float* src,
    int nx, int ny, int nz, int nzPadded  // nzPadded = 2*(nz/2+1)
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int iz = idx % nz;
    int iy = (idx / nz) % ny;
    int ix = idx / (nz * ny);

    padded[ix * ny * nzPadded + iy * nzPadded + iz] = src[idx];
}

// Copy from padded back to compact (float)
__global__ void copyFromPaddedF(
    float* dst,
    const float* padded,
    int nx, int ny, int nz, int nzPadded
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int iz = idx % nz;
    int iy = (idx / nz) % ny;
    int ix = idx / (nz * ny);

    dst[idx] = padded[ix * ny * nzPadded + iy * nzPadded + iz];
}

// Copy with padding for R2C (double)
__global__ void copyToPaddedD(
    double* padded,
    const double* src,
    int nx, int ny, int nz, int nzPadded
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int iz = idx % nz;
    int iy = (idx / nz) % ny;
    int ix = idx / (nz * ny);

    padded[ix * ny * nzPadded + iy * nzPadded + iz] = src[idx];
}

// Copy from padded back to compact (double)
__global__ void copyFromPaddedD(
    double* dst,
    const double* padded,
    int nx, int ny, int nz, int nzPadded
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;

    int iz = idx % nz;
    int iy = (idx / nz) % ny;
    int ix = idx / (nz * ny);

    dst[idx] = padded[ix * ny * nzPadded + iy * nzPadded + iz];
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
    s->complexSize = nx * ny * (nz / 2 + 1);
    s->useFloat = useFloat;
    s->invEigFloat = nullptr;
    s->invEigDouble = nullptr;
    s->complexBuf = nullptr;
    s->paddedRealBuf = nullptr;
    s->planR2C_f = 0;
    s->planC2R_f = 0;
    s->planD2Z = 0;
    s->planZ2D = 0;

    int nzHalf = nz / 2 + 1;
    int nzPadded = 2 * nzHalf;
    int paddedSize = nx * ny * nzPadded;

    // cuFFT dimensions: transform of size (nx, ny, nz)
    // Data layout: ix varies slowest, iz varies fastest
    // This matches OpenFOAM's blockMesh cell ordering for a single block:
    //   cellId = ix + iy*nx + iz*nx*ny  (ix fastest)
    // BUT cuFFT expects last dim to vary fastest.
    //
    // OpenFOAM ordering: ix fastest → data[iz][iy][ix]
    // cuFFT 3D(n0,n1,n2): n2 fastest → we pass n = {nz, ny, nx}
    // so that cuFFT treats ix (the fastest-varying in memory) as the
    // last transform dimension.
    int n[3] = {nz, ny, nx};

    int threads = 256;

    if (useFloat)
    {
        // Create cuFFT plans
        // For out-of-place R2C: input is padded real, output is complex
        cufftResult res;
        res = cufftPlanMany(
            &s->planR2C_f, 3, n,
            nullptr, 1, 0,  // inembed, istride, idist (auto)
            nullptr, 1, 0,  // onembed, ostride, odist (auto)
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

        // Allocate GPU buffers
        cudaMalloc(&s->invEigFloat, s->complexSize * sizeof(float));
        cudaMalloc(&s->complexBuf, s->complexSize * sizeof(cufftComplex));
        cudaMalloc(&s->paddedRealBuf, paddedSize * sizeof(float));

        // Compute inverse eigenvalues
        int blocks = (s->complexSize + threads - 1) / threads;
        computeInvEigenvaluesF<<<blocks, threads>>>(
            s->invEigFloat, nx, ny, nzHalf, nz,
            (float)dx, (float)dy, (float)dz
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
        cudaMalloc(&s->paddedRealBuf, paddedSize * sizeof(double));

        int blocks = (s->complexSize + threads - 1) / threads;
        computeInvEigenvaluesD<<<blocks, threads>>>(
            s->invEigDouble, nx, ny, nzHalf, nz,
            dx, dy, dz
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
    if (h->paddedRealBuf) cudaFree(h->paddedRealBuf);

    delete h;
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
    int nx = h->nx, ny = h->ny, nz = h->nz;
    int nzHalf = nz / 2 + 1;
    int nzPadded = 2 * nzHalf;

    float* padded = (float*)h->paddedRealBuf;
    cufftComplex* freq = (cufftComplex*)h->complexBuf;

    // Zero the padded buffer (handles padding bytes)
    cudaMemset(padded, 0, nx * ny * nzPadded * sizeof(float));

    // 1. Copy input to padded buffer
    int blocks = (n + threads - 1) / threads;
    copyToPaddedF<<<blocks, threads>>>(padded, b_ptr, nx, ny, nz, nzPadded);

    // 2. Forward R2C FFT (in-place on padded buffer → complex output)
    cufftExecR2C(h->planR2C_f, padded, freq);

    // 3. Scale by inverse eigenvalues
    int cblocks = (h->complexSize + threads - 1) / threads;
    scaleComplexF<<<cblocks, threads>>>(freq, h->invEigFloat, h->complexSize);

    // 4. Inverse C2R FFT
    cufftExecC2R(h->planC2R_f, freq, padded);

    // 5. Copy from padded to output with 1/N normalization
    copyFromPaddedF<<<blocks, threads>>>(x_ptr, padded, nx, ny, nz, nzPadded);

    float scale = 1.0f / (float)n;
    scaleRealF<<<blocks, threads>>>(x_ptr, scale, n);
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
    int nx = h->nx, ny = h->ny, nz = h->nz;
    int nzHalf = nz / 2 + 1;
    int nzPadded = 2 * nzHalf;

    double* padded = (double*)h->paddedRealBuf;
    cufftDoubleComplex* freq = (cufftDoubleComplex*)h->complexBuf;

    cudaMemset(padded, 0, nx * ny * nzPadded * sizeof(double));

    int blocks = (n + threads - 1) / threads;
    copyToPaddedD<<<blocks, threads>>>(padded, b_ptr, nx, ny, nz, nzPadded);

    cufftExecD2Z(h->planD2Z, padded, freq);

    int cblocks = (h->complexSize + threads - 1) / threads;
    scaleComplexD<<<cblocks, threads>>>(freq, h->invEigDouble, h->complexSize);

    cufftExecZ2D(h->planZ2D, freq, padded);

    copyFromPaddedD<<<blocks, threads>>>(x_ptr, padded, nx, ny, nz, nzPadded);

    double scale = 1.0 / (double)n;
    scaleRealD<<<blocks, threads>>>(x_ptr, scale, n);
}
