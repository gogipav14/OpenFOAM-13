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
    C API for geometric multigrid transfer and coarse-grid CUDA kernels.

    Provides:
    - Full-weighting restriction (2:1 in each direction)
    - Trilinear prolongation with Neumann ghost clamping
    - Per-cell coefficient restriction for coarse-level operator construction
    - 7-point stencil CSR matrix generation from per-cell coefficients

    All kernels accept an explicit CUDA stream parameter. Pass 0 for
    the default stream, or pass Ginkgo's executor stream to avoid
    cross-stream synchronization.

\*---------------------------------------------------------------------------*/

#ifndef MULTIGRID_KERNELS_H
#define MULTIGRID_KERNELS_H

#include <cuda_runtime.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Restrict a fine-grid vector to a coarse grid using full-weighting average.
 * 2:1 coarsening in each direction. One coarse cell = average of 2x2x2 fine.
 * Boundary-aware: at domain faces, averages available fine cells only.
 *
 * fine:  device pointer, size fnx*fny*fnz
 * coarse: device pointer, size cnx*cny*cnz
 * fnx,fny,fnz: fine-grid dimensions
 * cnx,cny,cnz: coarse-grid dimensions (should be fnx/2, fny/2, fnz/2)
 * stream: CUDA stream (0 = default)
 */
void mgRestrict3DFloat(
    const float* fine, float* coarse,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
);

void mgRestrict3DDouble(
    const double* fine, double* coarse,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
);

/*
 * Prolongate a coarse-grid vector to fine grid using trilinear interpolation.
 * Cell-centered 2:1 scheme: each fine cell gets weighted contribution from
 * up to 8 neighboring coarse cells. Weights are tensor products of
 * 1D weights {3/4, 1/4} based on sub-cell position.
 *
 * Neumann ghost handling: out-of-bounds coarse indices are clamped to the
 * boundary cell (zero-gradient). Uses __ldg() for read-only coarse data.
 *
 * coarse: device pointer, size cnx*cny*cnz (input)
 * fine:   device pointer, size fnx*fny*fnz (output)
 * fnx,fny,fnz: fine-grid dimensions
 * cnx,cny,cnz: coarse-grid dimensions
 * stream: CUDA stream
 */
void mgProlong3DFloat(
    const float* coarse, float* fine,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
);

void mgProlong3DDouble(
    const double* coarse, double* fine,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
);

/*
 * Restrict per-cell directional face coefficients from fine to coarse level.
 * For each coarse cell, averages the fine-cell coefficients in each direction.
 *
 * fineCoeffX/Y/Z: device pointers, size fnx*fny*fnz (input)
 * coarseCoeffX/Y/Z: device pointers, size cnx*cny*cnz (output)
 * stream: CUDA stream
 */
void mgRestrictCoeffsFloat(
    const float* fineCoeffX, const float* fineCoeffY, const float* fineCoeffZ,
    float* coarseCoeffX, float* coarseCoeffY, float* coarseCoeffZ,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
);

void mgRestrictCoeffsDouble(
    const double* fineCoeffX, const double* fineCoeffY, const double* fineCoeffZ,
    double* coarseCoeffX, double* coarseCoeffY, double* coarseCoeffZ,
    int fnx, int fny, int fnz,
    int cnx, int cny, int cnz,
    cudaStream_t stream
);

/*
 * Build a 7-point stencil CSR matrix on GPU from per-cell directional
 * coefficients and grid dimensions.
 *
 * coeffX/Y/Z: device pointers, per-cell coupling in each direction
 * nx,ny,nz: grid dimensions
 * rowPtr: device pointer, size nCells+1 (output, CSR row pointers)
 * colIdx: device pointer, size nnz (output, CSR column indices)
 * values: device pointer, size nnz (output, CSR values)
 *
 * Returns the number of non-zeros (nnz). Call with NULL outputs first
 * to query nnz, then allocate and call again.
 *
 * For interior cells: 7 entries (self + 6 neighbors).
 * For boundary cells: fewer neighbors. Diagonal = -sum(off-diagonals).
 */
int mgBuildCSR7ptFloat(
    const float* coeffX, const float* coeffY, const float* coeffZ,
    int nx, int ny, int nz,
    int* rowPtr, int* colIdx, float* values,
    cudaStream_t stream
);

int mgBuildCSR7ptDouble(
    const double* coeffX, const double* coeffY, const double* coeffZ,
    int nx, int ny, int nz,
    int* rowPtr, int* colIdx, double* values,
    cudaStream_t stream
);

#ifdef __cplusplus
}
#endif

#endif /* MULTIGRID_KERNELS_H */
