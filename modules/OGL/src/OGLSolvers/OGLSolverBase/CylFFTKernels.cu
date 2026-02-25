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
    CUDA implementation of cylindrical FFT+Thomas preconditioner.

    For structured (r, theta) grids, the FV Laplacian decouples in theta
    after DFT. Each Fourier mode leaves a tridiagonal system in r with
    r-dependent coefficients. The Thomas algorithm is pre-factored during
    setup and applied in O(nr) per mode at solve time.

\*---------------------------------------------------------------------------*/

#include "CylFFTKernels.h"
#include <cufft.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---- State structure -------------------------------------------------------

struct CylFFTPrecondState
{
    int nr;
    int ntheta;
    int nModes;      // ntheta/2 + 1 (R2C output)
    int totalCells;  // nr * ntheta
    int useFloat;

    // cuFFT plans: batched 1D R2C/C2R of length ntheta, batch = nr
    cufftHandle planR2C_f;  // float: R2C
    cufftHandle planC2R_f;  // float: C2R
    cufftHandle planD2Z;    // double: D2Z (R2C equivalent)
    cufftHandle planZ2D;    // double: Z2D (C2R equivalent)

    // Pre-factored Thomas coefficients on GPU.
    // For each Fourier mode m, the Thomas LU factors are:
    //   thomasL[m * (nr-1) + i]  = lower factor l[i] for mode m
    //   thomasD[m * nr + i]      = modified diagonal d'[i] for mode m
    // Upper diagonal is mode-independent:
    //   thomasU[i] = upper[i]
    float*  thomasL_f;
    float*  thomasD_f;
    float*  thomasU_f;
    double* thomasL_d;
    double* thomasD_d;
    double* thomasU_d;

    // Work buffer: complex data after forward FFT
    // Size: nr * nModes * sizeof(complex)
    void* complexBuf;

    // Work buffer: real data (copy of input for in-place FFT)
    // Size: nr * ntheta * sizeof(real)
    void* realBuf;

    // --- Per-sector DCT mode ---
    // When sectorMode=true, the FFT operates on mirrored (2N) data instead
    // of the raw (N=nthetaSector) data.  The cuFFT plans use length 2N with
    // batch = nr*nSectors.  The mirror symmetry ensures the FFT output
    // encodes DCT-II coefficients (Neumann BCs).
    bool sectorMode;
    int nSectors;       // number of angular sectors
    int nthetaSector;   // angular cells per sector (= N)
    // For sector mode: nModes = nthetaSector + 1, FFT length = 2*nthetaSector
    // realBuf sized: batch * 2*nthetaSector (mirror data)
    // complexBuf sized: batch * nModes
};


// ---- Thomas solve kernel ---------------------------------------------------

// Each thread solves one Fourier mode's tridiagonal system in r.
// Data is complex (from R2C FFT), coefficients are real (pre-factored).
// Access pattern: data[i_r * nModes + mode] with stride = nModes.

template<typename Real, typename Complex>
__global__ void thomasSolveKernel(
    Complex* __restrict__ data,   // [nr * nModes] complex values, in-place
    const Real* __restrict__ L,   // [nModes * (nr-1)] lower factors
    const Real* __restrict__ D,   // [nModes * nr] modified diagonals
    const Real* __restrict__ U,   // [nr] upper diagonal (mode-independent)
    int nr,
    int nModes
)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= nModes) return;

    int stride = nModes;

    // Note on m=0 (DC mode):  The 1D radial Laplacian with Neumann BCs is
    // singular (constant in null space).  Detrending the RHS (removing the
    // mean to project onto the range) is correct for a direct solve, but
    // HARMFUL inside a PCG preconditioner: the CG search direction needs
    // the full signal including the DC mean, which is ultimately absorbed
    // by the pressure reference cell.  Removing and re-centering distorts
    // the descent direction and causes sporadic PCG divergence.
    //
    // Instead, the factorisation uses a small diagonal regularisation
    // (regFrac = 1e-3) which approximates the reference cell's effect on
    // the DC mode and keeps the Thomas factorisation non-singular.

    // Forward substitution: data[i] -= L[i] * data[i-1]
    for (int i = 1; i < nr; i++)
    {
        Real l = L[m * (nr - 1) + (i - 1)];
        Complex prev = data[(i - 1) * stride + m];
        Complex& cur = data[i * stride + m];
        cur.x -= l * prev.x;
        cur.y -= l * prev.y;
    }

    // Back substitution: data[i] = (data[i] - U[i] * data[i+1]) / D[i]
    {
        int i = nr - 1;
        Real d = D[m * nr + i];
        Complex& val = data[i * stride + m];
        val.x /= d;
        val.y /= d;
    }
    for (int i = nr - 2; i >= 0; i--)
    {
        Real u = U[i];
        Real d = D[m * nr + i];
        Complex next = data[(i + 1) * stride + m];
        Complex& val = data[i * stride + m];
        val.x = (val.x - u * next.x) / d;
        val.y = (val.y - u * next.y) / d;
    }
}


// ---- Scale kernel (normalize after inverse FFT) ----------------------------

template<typename T>
__global__ void scaleKernel(T* data, T scale, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) data[idx] *= scale;
}


// ---- DCT mirror/extract kernels -------------------------------------------

// Mirror input for DCT-II via FFT:
// Given x[0..N-1] per batch, produce y[0..2N-1] where y[k]=x[k], y[2N-1-k]=x[k].
// This even-symmetric extension makes the DFT encode DCT-II coefficients.
template<typename T>
__global__ void mirrorForDCTKernel(
    const T* __restrict__ input,   // [totalBatch * N]
    T*       __restrict__ output,  // [totalBatch * 2N]
    int N, int totalElements       // totalElements = totalBatch * N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;
    int b = idx / N;
    int n = idx % N;
    T val = input[idx];
    output[b * 2 * N + n] = val;
    output[b * 2 * N + (2 * N - 1 - n)] = val;
}


// Extract first N elements from each 2N-length batch and apply scale.
template<typename T>
__global__ void extractScaleKernel(
    const T* __restrict__ input,   // [totalBatch * 2N]
    T*       __restrict__ output,  // [totalBatch * N]
    T scale,
    int N, int totalElements       // totalElements = totalBatch * N
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalElements) return;
    int b = idx / N;
    int n = idx % N;
    output[idx] = input[b * 2 * N + n] * scale;
}


// ---- C API implementation --------------------------------------------------

extern "C"
CylFFTHandle cylFFTPrecondCreate(
    int nr, int ntheta,
    int useFloat
)
{
    auto* s = new CylFFTPrecondState;
    s->nr = nr;
    s->ntheta = ntheta;
    s->nModes = ntheta / 2 + 1;
    s->totalCells = nr * ntheta;
    s->useFloat = useFloat;
    s->sectorMode = false;
    s->nSectors = 0;
    s->nthetaSector = 0;

    // Initialize pointers
    s->thomasL_f = nullptr;
    s->thomasD_f = nullptr;
    s->thomasU_f = nullptr;
    s->thomasL_d = nullptr;
    s->thomasD_d = nullptr;
    s->thomasU_d = nullptr;
    s->complexBuf = nullptr;
    s->realBuf = nullptr;

    // Create cuFFT plans: batched 1D FFT of length ntheta, batch = nr.
    // Input layout: data[i_r * ntheta + i_theta]
    // R2C output: data[i_r * nModes + m] (complex)
    int n[] = {ntheta};
    int inembed[] = {ntheta};
    int onembed[] = {s->nModes};

    if (useFloat)
    {
        cufftResult res = cufftPlanMany(
            &s->planR2C_f, 1, n,
            inembed, 1, ntheta,     // input: stride=1, dist=ntheta
            onembed, 1, s->nModes,  // output: stride=1, dist=nModes
            CUFFT_R2C, nr
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "CylFFT: R2C plan failed: %d\n", res);
            delete s;
            return nullptr;
        }

        res = cufftPlanMany(
            &s->planC2R_f, 1, n,
            onembed, 1, s->nModes,  // input: complex
            inembed, 1, ntheta,     // output: real
            CUFFT_C2R, nr
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "CylFFT: C2R plan failed: %d\n", res);
            cufftDestroy(s->planR2C_f);
            delete s;
            return nullptr;
        }

        // Allocate work buffers
        cudaMalloc(&s->complexBuf, nr * s->nModes * sizeof(cufftComplex));
        cudaMalloc(&s->realBuf, nr * ntheta * sizeof(float));
    }
    else
    {
        cufftResult res = cufftPlanMany(
            &s->planD2Z, 1, n,
            inembed, 1, ntheta,
            onembed, 1, s->nModes,
            CUFFT_D2Z, nr
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "CylFFT: D2Z plan failed: %d\n", res);
            delete s;
            return nullptr;
        }

        res = cufftPlanMany(
            &s->planZ2D, 1, n,
            onembed, 1, s->nModes,
            inembed, 1, ntheta,
            CUFFT_Z2D, nr
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "CylFFT: Z2D plan failed: %d\n", res);
            cufftDestroy(s->planD2Z);
            delete s;
            return nullptr;
        }

        cudaMalloc(&s->complexBuf, nr * s->nModes * sizeof(cufftDoubleComplex));
        cudaMalloc(&s->realBuf, nr * ntheta * sizeof(double));
    }

    return s;
}


extern "C"
void cylFFTPrecondDestroy(CylFFTHandle h)
{
    if (!h) return;

    if (h->useFloat)
    {
        cufftDestroy(h->planR2C_f);
        cufftDestroy(h->planC2R_f);
        if (h->thomasL_f) cudaFree(h->thomasL_f);
        if (h->thomasD_f) cudaFree(h->thomasD_f);
        if (h->thomasU_f) cudaFree(h->thomasU_f);
    }
    else
    {
        cufftDestroy(h->planD2Z);
        cufftDestroy(h->planZ2D);
        if (h->thomasL_d) cudaFree(h->thomasL_d);
        if (h->thomasD_d) cudaFree(h->thomasD_d);
        if (h->thomasU_d) cudaFree(h->thomasU_d);
    }

    if (h->complexBuf) cudaFree(h->complexBuf);
    if (h->realBuf) cudaFree(h->realBuf);

    delete h;
}


extern "C"
void cylFFTPrecondSetCoeffs(
    CylFFTHandle h,
    const double* lower,
    const double* upper,
    const double* thetaCoeff
)
{
    if (!h) return;

    int nr = h->nr;
    int ntheta = h->ntheta;
    int nModes = h->nModes;

    // Compute Thomas LU factorization for each Fourier mode on the host.
    // diagBase[i] = -(lower[i] + upper[i])  (sum of off-diagonals, negated)
    // diag[i,m] = diagBase[i] - 2*thetaCoeff[i]*(1 - cos(2*pi*m/ntheta))
    //
    // Thomas forward sweep:
    //   l[0] = 0 (no lower for first row)
    //   d'[0] = diag[0,m]
    //   For i=1..nr-1:
    //     l[i] = lower[i] / d'[i-1]
    //     d'[i] = diag[i,m] - l[i] * upper[i-1]

    // Allocate host arrays
    double* hL = new double[nModes * (nr - 1)];
    double* hD = new double[nModes * nr];
    double* hU = new double[nr];

    // Copy upper diagonal (mode-independent)
    memcpy(hU, upper, nr * sizeof(double));

    // Factor each mode
    for (int m = 0; m < nModes; m++)
    {
        double eigenTheta = 2.0 * (1.0 - cos(2.0 * M_PI * m / ntheta));

        // For m=0, the 1D radial Laplacian with Neumann BCs is singular
        // (constant in null space).  The kernel detrends the RHS before
        // solving, so the system is compatible to floating-point precision.
        // The diagonal regularisation keeps the Thomas factorisation
        // numerically stable (condition number ~ 1/regFrac must stay
        // within FP32 range, i.e. regFrac >= ~1e-4).
        double regFrac = (m == 0) ? 1e-3 : 0.0;

        // First row
        double diagBase0 = -(lower[0] + upper[0]);
        double diag0 = diagBase0 * (1.0 + regFrac)
                      - thetaCoeff[0] * eigenTheta;
        hD[m * nr + 0] = diag0;

        // Forward sweep
        for (int i = 1; i < nr; i++)
        {
            double diagBaseI = -(lower[i] + upper[i]);
            double diagI = diagBaseI * (1.0 + regFrac)
                          - thetaCoeff[i] * eigenTheta;

            double l = lower[i] / hD[m * nr + (i - 1)];
            hL[m * (nr - 1) + (i - 1)] = l;
            hD[m * nr + i] = diagI - l * upper[i - 1];
        }
    }

    // Transfer to GPU
    if (h->useFloat)
    {
        // Convert double to float
        float* hLf = new float[nModes * (nr - 1)];
        float* hDf = new float[nModes * nr];
        float* hUf = new float[nr];

        for (int i = 0; i < nModes * (nr - 1); i++) hLf[i] = (float)hL[i];
        for (int i = 0; i < nModes * nr; i++) hDf[i] = (float)hD[i];
        for (int i = 0; i < nr; i++) hUf[i] = (float)hU[i];

        if (h->thomasL_f) cudaFree(h->thomasL_f);
        if (h->thomasD_f) cudaFree(h->thomasD_f);
        if (h->thomasU_f) cudaFree(h->thomasU_f);

        cudaMalloc(&h->thomasL_f, nModes * (nr - 1) * sizeof(float));
        cudaMalloc(&h->thomasD_f, nModes * nr * sizeof(float));
        cudaMalloc(&h->thomasU_f, nr * sizeof(float));

        cudaMemcpy(h->thomasL_f, hLf, nModes * (nr - 1) * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(h->thomasD_f, hDf, nModes * nr * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(h->thomasU_f, hUf, nr * sizeof(float),
                   cudaMemcpyHostToDevice);

        delete[] hLf;
        delete[] hDf;
        delete[] hUf;
    }
    else
    {
        if (h->thomasL_d) cudaFree(h->thomasL_d);
        if (h->thomasD_d) cudaFree(h->thomasD_d);
        if (h->thomasU_d) cudaFree(h->thomasU_d);

        cudaMalloc(&h->thomasL_d, nModes * (nr - 1) * sizeof(double));
        cudaMalloc(&h->thomasD_d, nModes * nr * sizeof(double));
        cudaMalloc(&h->thomasU_d, nr * sizeof(double));

        cudaMemcpy(h->thomasL_d, hL, nModes * (nr - 1) * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(h->thomasD_d, hD, nModes * nr * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(h->thomasU_d, hU, nr * sizeof(double),
                   cudaMemcpyHostToDevice);
    }

    delete[] hL;
    delete[] hD;
    delete[] hU;
}


extern "C"
void cylFFTPrecondApplyFloat(
    CylFFTHandle h,
    const float* b_ptr,
    float* x_ptr,
    int n
)
{
    if (!h || !h->thomasL_f) return;

    int nr = h->nr;
    int ntheta = h->ntheta;
    int nModes = h->nModes;

    float* realBuf = static_cast<float*>(h->realBuf);
    cufftComplex* complexBuf = static_cast<cufftComplex*>(h->complexBuf);

    // Copy input to work buffer (cuFFT may be in-place or out-of-place)
    cudaMemcpy(realBuf, b_ptr, nr * ntheta * sizeof(float),
               cudaMemcpyDeviceToDevice);

    // Forward FFT: R2C, batch = nr
    cufftExecR2C(h->planR2C_f, realBuf, complexBuf);

    // Thomas solve: one thread per Fourier mode
    int blockSize = 128;
    int gridSize = (nModes + blockSize - 1) / blockSize;
    thomasSolveKernel<float, cufftComplex><<<gridSize, blockSize>>>(
        complexBuf,
        h->thomasL_f,
        h->thomasD_f,
        h->thomasU_f,
        nr, nModes
    );

    // Inverse FFT: C2R, batch = nr (output to x via realBuf)
    cufftExecC2R(h->planC2R_f, complexBuf, realBuf);

    // Normalize by 1/ntheta (cuFFT does unnormalized transforms)
    int totalCells = nr * ntheta;
    float scale = 1.0f / ntheta;
    int sBlockSize = 256;
    int sGridSize = (totalCells + sBlockSize - 1) / sBlockSize;
    scaleKernel<float><<<sGridSize, sBlockSize>>>(realBuf, scale, totalCells);

    // Copy result to output
    cudaMemcpy(x_ptr, realBuf, totalCells * sizeof(float),
               cudaMemcpyDeviceToDevice);
}


extern "C"
void cylFFTPrecondApplyDouble(
    CylFFTHandle h,
    const double* b_ptr,
    double* x_ptr,
    int n
)
{
    if (!h || !h->thomasL_d) return;

    int nr = h->nr;
    int ntheta = h->ntheta;
    int nModes = h->nModes;

    double* realBuf = static_cast<double*>(h->realBuf);
    cufftDoubleComplex* complexBuf =
        static_cast<cufftDoubleComplex*>(h->complexBuf);

    cudaMemcpy(realBuf, b_ptr, nr * ntheta * sizeof(double),
               cudaMemcpyDeviceToDevice);

    // Forward FFT: D2Z, batch = nr
    cufftExecD2Z(h->planD2Z, realBuf, complexBuf);

    // Thomas solve
    int blockSize = 128;
    int gridSize = (nModes + blockSize - 1) / blockSize;
    thomasSolveKernel<double, cufftDoubleComplex><<<gridSize, blockSize>>>(
        complexBuf,
        h->thomasL_d,
        h->thomasD_d,
        h->thomasU_d,
        nr, nModes
    );

    // Inverse FFT: Z2D, batch = nr
    cufftExecZ2D(h->planZ2D, complexBuf, realBuf);

    // Normalize
    int totalCells = nr * ntheta;
    double scale = 1.0 / ntheta;
    int sBlockSize = 256;
    int sGridSize = (totalCells + sBlockSize - 1) / sBlockSize;
    scaleKernel<double><<<sGridSize, sBlockSize>>>(realBuf, scale, totalCells);

    cudaMemcpy(x_ptr, realBuf, totalCells * sizeof(double),
               cudaMemcpyDeviceToDevice);
}


// ---- Per-sector DCT C API -------------------------------------------------

// The key insight: for an even-symmetric mirror extension y[n]=x[n],
// y[2N-1-n]=x[n], the DFT satisfies Y[k] = 2*exp(j*pi*k/(2N))*DCT[k].
// Since Thomas is linear, we can apply Thomas directly on the complex Y[k]
// without extracting/injecting DCT coefficients.  The only differences from
// the DFT path are: (1) mirror input, (2) DCT eigenvalues in Thomas,
// (3) extract first N elements and scale by 1/(2N).

extern "C"
CylFFTHandle cylFFTPrecondCreateSector(
    int nr, int nSectors, int nthetaSector,
    int useFloat
)
{
    auto* s = new CylFFTPrecondState;
    s->sectorMode = true;
    s->nr = nr;
    s->nSectors = nSectors;
    s->nthetaSector = nthetaSector;
    s->ntheta = nSectors * nthetaSector;  // total angular cells
    s->totalCells = nr * nSectors * nthetaSector;
    s->useFloat = useFloat;

    // FFT operates on mirrored data of length 2*nthetaSector
    int fftLen = 2 * nthetaSector;
    s->nModes = nthetaSector + 1;  // R2C output: fftLen/2 + 1

    int batch = nr * nSectors;

    // Initialize pointers
    s->thomasL_f = nullptr;
    s->thomasD_f = nullptr;
    s->thomasU_f = nullptr;
    s->thomasL_d = nullptr;
    s->thomasD_d = nullptr;
    s->thomasU_d = nullptr;
    s->complexBuf = nullptr;
    s->realBuf = nullptr;

    // Create cuFFT plans: batched 1D R2C/C2R of length 2*nthetaSector
    int n[] = {fftLen};
    int inembed[] = {fftLen};
    int onembed[] = {s->nModes};

    if (useFloat)
    {
        cufftResult res = cufftPlanMany(
            &s->planR2C_f, 1, n,
            inembed, 1, fftLen,       // input: stride=1, dist=fftLen
            onembed, 1, s->nModes,    // output: stride=1, dist=nModes
            CUFFT_R2C, batch
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "CylFFT Sector: R2C plan failed: %d\n", res);
            delete s;
            return nullptr;
        }

        res = cufftPlanMany(
            &s->planC2R_f, 1, n,
            onembed, 1, s->nModes,
            inembed, 1, fftLen,
            CUFFT_C2R, batch
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "CylFFT Sector: C2R plan failed: %d\n", res);
            cufftDestroy(s->planR2C_f);
            delete s;
            return nullptr;
        }

        // Allocate work buffers for mirrored (2N) data
        cudaMalloc(&s->realBuf, batch * fftLen * sizeof(float));
        cudaMalloc(&s->complexBuf, batch * s->nModes * sizeof(cufftComplex));
    }
    else
    {
        cufftResult res = cufftPlanMany(
            &s->planD2Z, 1, n,
            inembed, 1, fftLen,
            onembed, 1, s->nModes,
            CUFFT_D2Z, batch
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "CylFFT Sector: D2Z plan failed: %d\n", res);
            delete s;
            return nullptr;
        }

        res = cufftPlanMany(
            &s->planZ2D, 1, n,
            onembed, 1, s->nModes,
            inembed, 1, fftLen,
            CUFFT_Z2D, batch
        );
        if (res != CUFFT_SUCCESS)
        {
            fprintf(stderr, "CylFFT Sector: Z2D plan failed: %d\n", res);
            cufftDestroy(s->planD2Z);
            delete s;
            return nullptr;
        }

        cudaMalloc(&s->realBuf, batch * fftLen * sizeof(double));
        cudaMalloc(&s->complexBuf,
                   batch * s->nModes * sizeof(cufftDoubleComplex));
    }

    return s;
}


extern "C"
void cylFFTPrecondSetCoeffsSector(
    CylFFTHandle h,
    const double* lower,
    const double* upper,
    const double* thetaCoeff
)
{
    if (!h || !h->sectorMode) return;

    int nr = h->nr;
    int N = h->nthetaSector;
    int nModes = h->nModes;  // N + 1

    // Thomas LU factorization with DCT-II eigenvalues.
    // eigenTheta[m] = 2*(1 - cos(pi*m/N)) for m=0..N
    // This is the discrete Neumann-Laplacian eigenvalue.

    double* hL = new double[nModes * (nr - 1)];
    double* hD = new double[nModes * nr];
    double* hU = new double[nr];

    memcpy(hU, upper, nr * sizeof(double));

    for (int m = 0; m < nModes; m++)
    {
        double eigenTheta = 2.0 * (1.0 - cos(M_PI * m / N));

        // m=0: constant mode (Neumann null space), needs regularization
        double regFrac = (m == 0) ? 1e-3 : 0.0;

        // First row
        double diagBase0 = -(lower[0] + upper[0]);
        double diag0 = diagBase0 * (1.0 + regFrac)
                      - thetaCoeff[0] * eigenTheta;
        hD[m * nr + 0] = diag0;

        // Forward sweep
        for (int i = 1; i < nr; i++)
        {
            double diagBaseI = -(lower[i] + upper[i]);
            double diagI = diagBaseI * (1.0 + regFrac)
                          - thetaCoeff[i] * eigenTheta;

            double l = lower[i] / hD[m * nr + (i - 1)];
            hL[m * (nr - 1) + (i - 1)] = l;
            hD[m * nr + i] = diagI - l * upper[i - 1];
        }
    }

    // Transfer to GPU
    if (h->useFloat)
    {
        float* hLf = new float[nModes * (nr - 1)];
        float* hDf = new float[nModes * nr];
        float* hUf = new float[nr];

        for (int i = 0; i < nModes * (nr - 1); i++) hLf[i] = (float)hL[i];
        for (int i = 0; i < nModes * nr; i++) hDf[i] = (float)hD[i];
        for (int i = 0; i < nr; i++) hUf[i] = (float)hU[i];

        if (h->thomasL_f) cudaFree(h->thomasL_f);
        if (h->thomasD_f) cudaFree(h->thomasD_f);
        if (h->thomasU_f) cudaFree(h->thomasU_f);

        cudaMalloc(&h->thomasL_f, nModes * (nr - 1) * sizeof(float));
        cudaMalloc(&h->thomasD_f, nModes * nr * sizeof(float));
        cudaMalloc(&h->thomasU_f, nr * sizeof(float));

        cudaMemcpy(h->thomasL_f, hLf, nModes * (nr - 1) * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(h->thomasD_f, hDf, nModes * nr * sizeof(float),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(h->thomasU_f, hUf, nr * sizeof(float),
                   cudaMemcpyHostToDevice);

        delete[] hLf;
        delete[] hDf;
        delete[] hUf;
    }
    else
    {
        if (h->thomasL_d) cudaFree(h->thomasL_d);
        if (h->thomasD_d) cudaFree(h->thomasD_d);
        if (h->thomasU_d) cudaFree(h->thomasU_d);

        cudaMalloc(&h->thomasL_d, nModes * (nr - 1) * sizeof(double));
        cudaMalloc(&h->thomasD_d, nModes * nr * sizeof(double));
        cudaMalloc(&h->thomasU_d, nr * sizeof(double));

        cudaMemcpy(h->thomasL_d, hL, nModes * (nr - 1) * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(h->thomasD_d, hD, nModes * nr * sizeof(double),
                   cudaMemcpyHostToDevice);
        cudaMemcpy(h->thomasU_d, hU, nr * sizeof(double),
                   cudaMemcpyHostToDevice);
    }

    delete[] hL;
    delete[] hD;
    delete[] hU;
}


extern "C"
void cylFFTPrecondApplySectorFloat(
    CylFFTHandle h,
    const float* b_ptr,
    float* x_ptr,
    int n
)
{
    if (!h || !h->sectorMode || !h->thomasL_f) return;

    int nr = h->nr;
    int nSectors = h->nSectors;
    int N = h->nthetaSector;
    int nModes = h->nModes;
    int batch = nr * nSectors;
    int totalInput = batch * N;
    int fftLen = 2 * N;

    float* realBuf = static_cast<float*>(h->realBuf);
    cufftComplex* complexBuf = static_cast<cufftComplex*>(h->complexBuf);

    // 1. Mirror: input[batch*N] -> realBuf[batch*2N]
    {
        int blockSize = 256;
        int gridSize = (totalInput + blockSize - 1) / blockSize;
        mirrorForDCTKernel<float><<<gridSize, blockSize>>>(
            b_ptr, realBuf, N, totalInput
        );
    }

    // 2. Forward FFT: R2C of length 2N, batch = nr*nSectors
    cufftExecR2C(h->planR2C_f, realBuf, complexBuf);

    // 3. Thomas solve: one launch per sector (sectors share Thomas factors)
    {
        int blockSize = 128;
        int gridSize = (nModes + blockSize - 1) / blockSize;
        for (int s = 0; s < nSectors; s++)
        {
            cufftComplex* sectorData = complexBuf + s * nr * nModes;
            thomasSolveKernel<float, cufftComplex><<<gridSize, blockSize>>>(
                sectorData,
                h->thomasL_f,
                h->thomasD_f,
                h->thomasU_f,
                nr, nModes
            );
        }
    }

    // 4. Inverse FFT: C2R of length 2N, batch = nr*nSectors
    cufftExecC2R(h->planC2R_f, complexBuf, realBuf);

    // 5. Extract first N elements from each 2N batch + scale by 1/(2N)
    {
        float scale = 1.0f / fftLen;
        int blockSize = 256;
        int gridSize = (totalInput + blockSize - 1) / blockSize;
        extractScaleKernel<float><<<gridSize, blockSize>>>(
            realBuf, x_ptr, scale, N, totalInput
        );
    }
}


extern "C"
void cylFFTPrecondApplySectorDouble(
    CylFFTHandle h,
    const double* b_ptr,
    double* x_ptr,
    int n
)
{
    if (!h || !h->sectorMode || !h->thomasL_d) return;

    int nr = h->nr;
    int nSectors = h->nSectors;
    int N = h->nthetaSector;
    int nModes = h->nModes;
    int batch = nr * nSectors;
    int totalInput = batch * N;
    int fftLen = 2 * N;

    double* realBuf = static_cast<double*>(h->realBuf);
    cufftDoubleComplex* complexBuf =
        static_cast<cufftDoubleComplex*>(h->complexBuf);

    // 1. Mirror
    {
        int blockSize = 256;
        int gridSize = (totalInput + blockSize - 1) / blockSize;
        mirrorForDCTKernel<double><<<gridSize, blockSize>>>(
            b_ptr, realBuf, N, totalInput
        );
    }

    // 2. Forward FFT: D2Z
    cufftExecD2Z(h->planD2Z, realBuf, complexBuf);

    // 3. Thomas solve per sector
    {
        int blockSize = 128;
        int gridSize = (nModes + blockSize - 1) / blockSize;
        for (int s = 0; s < nSectors; s++)
        {
            cufftDoubleComplex* sectorData = complexBuf + s * nr * nModes;
            thomasSolveKernel<double, cufftDoubleComplex>
                <<<gridSize, blockSize>>>(
                sectorData,
                h->thomasL_d,
                h->thomasD_d,
                h->thomasU_d,
                nr, nModes
            );
        }
    }

    // 4. Inverse FFT: Z2D
    cufftExecZ2D(h->planZ2D, complexBuf, realBuf);

    // 5. Extract + scale
    {
        double scale = 1.0 / fftLen;
        int blockSize = 256;
        int gridSize = (totalInput + blockSize - 1) / blockSize;
        extractScaleKernel<double><<<gridSize, blockSize>>>(
            realBuf, x_ptr, scale, N, totalInput
        );
    }
}
