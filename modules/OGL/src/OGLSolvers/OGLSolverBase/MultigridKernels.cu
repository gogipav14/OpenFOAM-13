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
    CUDA kernels for geometric multigrid transfer operators and coarse-grid
    matrix construction.

    Transfer operators for cell-centered FV with 2:1 geometric coarsening:
    - Restriction: full-weighting average of 2x2x2 fine-cell blocks
    - Prolongation: trilinear interpolation with Neumann ghost clamping

    Coarse-grid operator construction:
    - Per-cell coefficient restriction (arithmetic mean of fine faces)
    - 7-point stencil CSR matrix generation from per-cell coefficients

    All kernels accept explicit CUDA stream for integration with Ginkgo.

\*---------------------------------------------------------------------------*/

#include <cuda_runtime.h>
#include <cstdio>
#include "MultigridKernels.h"


// -------------------------------------------------------------------------
// Restriction kernel: full-weighting average
// -------------------------------------------------------------------------

template<typename T>
__global__ void restrict3DKernel(
    const T* __restrict__ fine,
    T* __restrict__ coarse,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cnx * cny * cnz) return;

    int cx = idx % cnx;
    int cy = (idx / cnx) % cny;
    int cz = idx / (cnx * cny);

    T sum = T(0);
    int count = 0;

    for (int dz = 0; dz < 2; dz++)
    for (int dy = 0; dy < 2; dy++)
    for (int dx = 0; dx < 2; dx++)
    {
        int fx = 2 * cx + dx;
        int fy = 2 * cy + dy;
        int fz = 2 * cz + dz;
        if (fx < fnx && fy < fny && fz < fnz)
        {
            sum += fine[fx + fy * fnx + fz * fnx * fny];
            count++;
        }
    }

    coarse[idx] = sum / T(count);
}


// -------------------------------------------------------------------------
// Prolongation kernel: trilinear interpolation
// -------------------------------------------------------------------------

// Helper: clamp value to [lo, hi]
__device__ __forceinline__ int clampInt(int v, int lo, int hi)
{
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

template<typename T>
__global__ void prolong3DTrilinearKernel(
    const T* __restrict__ coarse,
    T* __restrict__ fine,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= fnx * fny * fnz) return;

    int fx = idx % fnx;
    int fy = (idx / fnx) % fny;
    int fz = idx / (fnx * fny);

    // Parent coarse cell
    int cx = fx / 2;
    int cy = fy / 2;
    int cz = fz / 2;

    // Sub-position within the coarse cell (0 = left/bottom, 1 = right/top)
    int sx = fx & 1;
    int sy = fy & 1;
    int sz = fz & 1;

    // 1D trilinear weights for cell-centered 2:1 coarsening:
    //   sub=0 (left half):  w_parent=3/4, w_left_neighbor=1/4
    //   sub=1 (right half): w_parent=3/4, w_right_neighbor=1/4
    //
    // In tensor product form, each fine cell reads from 2^3 = 8 coarse cells.
    // The two coarse cells per direction are:
    //   sub=0: {cx-1, cx}  with weights {1/4, 3/4}
    //   sub=1: {cx, cx+1}  with weights {3/4, 1/4}

    // Coarse cell indices for each direction's two contributors
    int ix0 = sx ? cx : (cx - 1);
    int ix1 = sx ? (cx + 1) : cx;
    int iy0 = sy ? cy : (cy - 1);
    int iy1 = sy ? (cy + 1) : cy;
    int iz0 = sz ? cz : (cz - 1);
    int iz1 = sz ? (cz + 1) : cz;

    // Clamp to grid boundaries (Neumann ghost = boundary value)
    ix0 = clampInt(ix0, 0, cnx - 1);
    ix1 = clampInt(ix1, 0, cnx - 1);
    iy0 = clampInt(iy0, 0, cny - 1);
    iy1 = clampInt(iy1, 0, cny - 1);
    iz0 = clampInt(iz0, 0, cnz - 1);
    iz1 = clampInt(iz1, 0, cnz - 1);

    // Weights: w0 = 3/4 (parent side), w1 = 1/4 (neighbor side)
    T w0 = T(0.75);
    T w1 = T(0.25);

    // wx[0] = weight for ix0, wx[1] = weight for ix1
    // sub=0: ix0 is neighbor (1/4), ix1 is parent (3/4)
    // sub=1: ix0 is parent (3/4), ix1 is neighbor (1/4)
    T wx0 = sx ? w0 : w1;
    T wx1 = sx ? w1 : w0;
    T wy0 = sy ? w0 : w1;
    T wy1 = sy ? w1 : w0;
    T wz0 = sz ? w0 : w1;
    T wz1 = sz ? w1 : w0;

    // Read 8 coarse values via __ldg (read-only texture cache path).
    // Adjacent fine threads in a 2x2x2 block share most of these reads.
    #define CVAL(i,j,k) __ldg(&coarse[(i) + (j)*cnx + (k)*cnx*cny])

    T val = T(0);
    val += wx0 * wy0 * wz0 * CVAL(ix0, iy0, iz0);
    val += wx1 * wy0 * wz0 * CVAL(ix1, iy0, iz0);
    val += wx0 * wy1 * wz0 * CVAL(ix0, iy1, iz0);
    val += wx1 * wy1 * wz0 * CVAL(ix1, iy1, iz0);
    val += wx0 * wy0 * wz1 * CVAL(ix0, iy0, iz1);
    val += wx1 * wy0 * wz1 * CVAL(ix1, iy0, iz1);
    val += wx0 * wy1 * wz1 * CVAL(ix0, iy1, iz1);
    val += wx1 * wy1 * wz1 * CVAL(ix1, iy1, iz1);

    fine[idx] = val;

    #undef CVAL
}


// -------------------------------------------------------------------------
// Coefficient restriction kernel
// -------------------------------------------------------------------------

template<typename T>
__global__ void restrictCoeffsKernel(
    const T* __restrict__ fineX,
    const T* __restrict__ fineY,
    const T* __restrict__ fineZ,
    T* __restrict__ coarseX,
    T* __restrict__ coarseY,
    T* __restrict__ coarseZ,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= cnx * cny * cnz) return;

    int cx = idx % cnx;
    int cy = (idx / cnx) % cny;
    int cz = idx / (cnx * cny);

    // Average the fine-cell coefficients within this coarse cell.
    // For directional coefficients, average across the 2x2x2 block.
    T sumX = T(0), sumY = T(0), sumZ = T(0);
    int count = 0;

    for (int dz = 0; dz < 2; dz++)
    for (int dy = 0; dy < 2; dy++)
    for (int dx = 0; dx < 2; dx++)
    {
        int fx = 2 * cx + dx;
        int fy = 2 * cy + dy;
        int fz = 2 * cz + dz;
        if (fx < fnx && fy < fny && fz < fnz)
        {
            int fIdx = fx + fy * fnx + fz * fnx * fny;
            sumX += fineX[fIdx];
            sumY += fineY[fIdx];
            sumZ += fineZ[fIdx];
            count++;
        }
    }

    // Scale: coarse coupling = 2 * fine coupling for 2:1 coarsening.
    // (Area quadruples but distance doubles -> net 2x scaling.)
    T scale = T(2) / T(count);
    coarseX[idx] = sumX * scale;
    coarseY[idx] = sumY * scale;
    coarseZ[idx] = sumZ * scale;
}


// -------------------------------------------------------------------------
// 7-point stencil CSR matrix builder
// -------------------------------------------------------------------------

// Phase 1: count non-zeros per row (for rowPtr construction)
__global__ void countNnzPerRow(
    int nx, int ny, int nz,
    int* rowNnz
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nCells = nx * ny * nz;
    if (idx >= nCells) return;

    int ix = idx % nx;
    int iy = (idx / nx) % ny;
    int iz = idx / (nx * ny);

    // Count: 1 (diagonal) + number of existing neighbors
    int count = 1;
    if (ix > 0)      count++;  // -x
    if (ix < nx - 1) count++;  // +x
    if (iy > 0)      count++;  // -y
    if (iy < ny - 1) count++;  // +y
    if (iz > 0)      count++;  // -z
    if (iz < nz - 1) count++;  // +z

    rowNnz[idx] = count;
}

// Phase 2: fill CSR arrays (colIdx + values)
template<typename T>
__global__ void fillCSR7pt(
    const T* __restrict__ coeffX,
    const T* __restrict__ coeffY,
    const T* __restrict__ coeffZ,
    int nx, int ny, int nz,
    const int* __restrict__ rowPtr,
    int* __restrict__ colIdx,
    T* __restrict__ values
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int nCells = nx * ny * nz;
    if (idx >= nCells) return;

    int ix = idx % nx;
    int iy = (idx / nx) % ny;
    int iz = idx / (nx * ny);

    int offset = rowPtr[idx];
    T diag = T(0);

    // Fill off-diagonals in sorted column order.
    // Column index = ix + iy*nx + iz*nx*ny

    // -z neighbor
    if (iz > 0)
    {
        int col = ix + iy * nx + (iz - 1) * nx * ny;
        T val = coeffZ[idx];
        colIdx[offset] = col;
        values[offset] = val;
        diag -= val;
        offset++;
    }

    // -y neighbor
    if (iy > 0)
    {
        int col = ix + (iy - 1) * nx + iz * nx * ny;
        T val = coeffY[idx];
        colIdx[offset] = col;
        values[offset] = val;
        diag -= val;
        offset++;
    }

    // -x neighbor
    if (ix > 0)
    {
        int col = (ix - 1) + iy * nx + iz * nx * ny;
        T val = coeffX[idx];
        colIdx[offset] = col;
        values[offset] = val;
        diag -= val;
        offset++;
    }

    // Diagonal (placeholder, filled after off-diags)
    int diagOffset = offset;
    colIdx[offset] = idx;
    offset++;

    // +x neighbor
    if (ix < nx - 1)
    {
        int col = (ix + 1) + iy * nx + iz * nx * ny;
        T val = coeffX[col];  // Use the +x neighbor's coeffX
        colIdx[offset] = col;
        values[offset] = val;
        diag -= val;
        offset++;
    }

    // +y neighbor
    if (iy < ny - 1)
    {
        int col = ix + (iy + 1) * nx + iz * nx * ny;
        T val = coeffY[col];  // Use the +y neighbor's coeffY
        colIdx[offset] = col;
        values[offset] = val;
        diag -= val;
        offset++;
    }

    // +z neighbor
    if (iz < nz - 1)
    {
        int col = ix + iy * nx + (iz + 1) * nx * ny;
        T val = coeffZ[col];  // Use the +z neighbor's coeffZ
        colIdx[offset] = col;
        values[offset] = val;
        diag -= val;
        offset++;
    }

    // Diagonal = negative sum of off-diagonals (Laplacian property)
    values[diagOffset] = diag;
}

// Exclusive prefix sum on CPU for small arrays (coarse grids)
static void exclusivePrefixSum(int* data, int n)
{
    int sum = 0;
    for (int i = 0; i < n; i++)
    {
        int val = data[i];
        data[i] = sum;
        sum += val;
    }
}


// -------------------------------------------------------------------------
// C API implementations
// -------------------------------------------------------------------------

extern "C"
void mgRestrict3DFloat(
    const float* fine, float* coarse,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
)
{
    int nCoarse = cnx * cny * cnz;
    int threads = 256;
    int blocks = (nCoarse + threads - 1) / threads;
    restrict3DKernel<float><<<blocks, threads, 0, stream>>>(
        fine, coarse, fnx, fny, fnz, cnx, cny, cnz
    );
}

extern "C"
void mgRestrict3DDouble(
    const double* fine, double* coarse,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
)
{
    int nCoarse = cnx * cny * cnz;
    int threads = 256;
    int blocks = (nCoarse + threads - 1) / threads;
    restrict3DKernel<double><<<blocks, threads, 0, stream>>>(
        fine, coarse, fnx, fny, fnz, cnx, cny, cnz
    );
}

extern "C"
void mgProlong3DFloat(
    const float* coarse, float* fine,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
)
{
    int nFine = fnx * fny * fnz;
    int threads = 256;
    int blocks = (nFine + threads - 1) / threads;
    prolong3DTrilinearKernel<float><<<blocks, threads, 0, stream>>>(
        coarse, fine, fnx, fny, fnz, cnx, cny, cnz
    );
}

extern "C"
void mgProlong3DDouble(
    const double* coarse, double* fine,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
)
{
    int nFine = fnx * fny * fnz;
    int threads = 256;
    int blocks = (nFine + threads - 1) / threads;
    prolong3DTrilinearKernel<double><<<blocks, threads, 0, stream>>>(
        coarse, fine, fnx, fny, fnz, cnx, cny, cnz
    );
}

extern "C"
void mgRestrictCoeffsFloat(
    const float* fineX, const float* fineY, const float* fineZ,
    float* coarseX, float* coarseY, float* coarseZ,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
)
{
    int nCoarse = cnx * cny * cnz;
    int threads = 256;
    int blocks = (nCoarse + threads - 1) / threads;
    restrictCoeffsKernel<float><<<blocks, threads, 0, stream>>>(
        fineX, fineY, fineZ, coarseX, coarseY, coarseZ,
        fnx, fny, fnz, cnx, cny, cnz
    );
}

extern "C"
void mgRestrictCoeffsDouble(
    const double* fineX, const double* fineY, const double* fineZ,
    double* coarseX, double* coarseY, double* coarseZ,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
)
{
    int nCoarse = cnx * cny * cnz;
    int threads = 256;
    int blocks = (nCoarse + threads - 1) / threads;
    restrictCoeffsKernel<double><<<blocks, threads, 0, stream>>>(
        fineX, fineY, fineZ, coarseX, coarseY, coarseZ,
        fnx, fny, fnz, cnx, cny, cnz
    );
}

extern "C"
int mgBuildCSR7ptFloat(
    const float* coeffX, const float* coeffY, const float* coeffZ,
    int nx, int ny, int nz,
    int* rowPtr, int* colIdx, float* values,
    cudaStream_t stream
)
{
    int nCells = nx * ny * nz;
    int threads = 256;
    int blocks = (nCells + threads - 1) / threads;

    // Phase 1: count nnz per row
    int* d_rowNnz;
    cudaMalloc(&d_rowNnz, nCells * sizeof(int));
    countNnzPerRow<<<blocks, threads, 0, stream>>>(nx, ny, nz, d_rowNnz);
    cudaStreamSynchronize(stream);

    // Copy to host for prefix sum (coarse grids are small, <100k cells)
    int* h_rowNnz = new int[nCells + 1];
    cudaMemcpy(h_rowNnz, d_rowNnz, nCells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_rowNnz);

    // Compute total nnz
    int nnz = 0;
    for (int i = 0; i < nCells; i++) nnz += h_rowNnz[i];

    // If caller just wants the nnz count (rowPtr/colIdx/values are NULL)
    if (!rowPtr || !colIdx || !values)
    {
        delete[] h_rowNnz;
        return nnz;
    }

    // Build rowPtr via exclusive prefix sum
    int* h_rowPtr = new int[nCells + 1];
    exclusivePrefixSum(h_rowNnz, nCells);
    for (int i = 0; i < nCells; i++) h_rowPtr[i] = h_rowNnz[i];
    h_rowPtr[nCells] = nnz;
    delete[] h_rowNnz;

    cudaMemcpy(rowPtr, h_rowPtr, (nCells + 1) * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_rowPtr;

    // Phase 2: fill colIdx and values
    fillCSR7pt<float><<<blocks, threads, 0, stream>>>(
        coeffX, coeffY, coeffZ,
        nx, ny, nz,
        rowPtr, colIdx, values
    );

    return nnz;
}

extern "C"
int mgBuildCSR7ptDouble(
    const double* coeffX, const double* coeffY, const double* coeffZ,
    int nx, int ny, int nz,
    int* rowPtr, int* colIdx, double* values,
    cudaStream_t stream
)
{
    int nCells = nx * ny * nz;
    int threads = 256;
    int blocks = (nCells + threads - 1) / threads;

    // Phase 1: count nnz per row
    int* d_rowNnz;
    cudaMalloc(&d_rowNnz, nCells * sizeof(int));
    countNnzPerRow<<<blocks, threads, 0, stream>>>(nx, ny, nz, d_rowNnz);
    cudaStreamSynchronize(stream);

    int* h_rowNnz = new int[nCells + 1];
    cudaMemcpy(h_rowNnz, d_rowNnz, nCells * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_rowNnz);

    int nnz = 0;
    for (int i = 0; i < nCells; i++) nnz += h_rowNnz[i];

    if (!rowPtr || !colIdx || !values)
    {
        delete[] h_rowNnz;
        return nnz;
    }

    int* h_rowPtr = new int[nCells + 1];
    exclusivePrefixSum(h_rowNnz, nCells);
    for (int i = 0; i < nCells; i++) h_rowPtr[i] = h_rowNnz[i];
    h_rowPtr[nCells] = nnz;
    delete[] h_rowNnz;

    cudaMemcpy(rowPtr, h_rowPtr, (nCells + 1) * sizeof(int), cudaMemcpyHostToDevice);
    delete[] h_rowPtr;

    fillCSR7pt<double><<<blocks, threads, 0, stream>>>(
        coeffX, coeffY, coeffZ,
        nx, ny, nz,
        rowPtr, colIdx, values
    );

    return nnz;
}
