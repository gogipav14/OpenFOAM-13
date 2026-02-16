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
    C API for FFT-based Laplacian preconditioner using cuFFT.

    Provides O(N log N) approximate inverse of the discrete Laplacian
    on uniform Cartesian grids. Used as a preconditioner for CG/PCG
    in the OGL GPU solver module.

    The preconditioner exploits FFT diagonalization of the periodic
    discrete Laplacian. For non-periodic BCs (Neumann/Dirichlet),
    this serves as an approximate preconditioner with the Krylov
    solver handling the boundary mismatch.

\*---------------------------------------------------------------------------*/

#ifndef FFT_KERNELS_H
#define FFT_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for FFT preconditioner state (cuFFT plans + eigenvalues) */
typedef struct FFTPrecondState* FFTPrecondHandle;

/*
 * Create FFT preconditioner for a structured Cartesian grid.
 *
 * nx, ny, nz: grid dimensions (cells in x, y, z)
 * dx, dy, dz: uniform cell spacing in each direction
 * useFloat:   1 for float (FP32), 0 for double (FP64)
 *
 * Returns handle to preconditioner state, or NULL on failure.
 * The cuFFT plans and eigenvalue arrays are allocated on the GPU.
 */
FFTPrecondHandle fftPrecondCreate(
    int nx, int ny, int nz,
    double dx, double dy, double dz,
    int useFloat
);

/*
 * Destroy FFT preconditioner and free all GPU resources.
 */
void fftPrecondDestroy(FFTPrecondHandle h);

/*
 * Apply FFT preconditioner in single precision.
 *
 * Computes: x = L^{-1} * b  where L is the discrete Laplacian
 * Via: x = IFFT( FFT(b) / eigenvalues ) * (1/N)
 *
 * b_ptr: device pointer to input vector (GPU memory, size n)
 * x_ptr: device pointer to output vector (GPU memory, size n)
 * n:     vector length (must equal nx*ny*nz from create)
 */
void fftPrecondApplyFloat(
    FFTPrecondHandle h,
    const float* b_ptr,
    float* x_ptr,
    int n
);

/*
 * Apply FFT preconditioner in double precision.
 * Same interface as float version.
 */
void fftPrecondApplyDouble(
    FFTPrecondHandle h,
    const double* b_ptr,
    double* x_ptr,
    int n
);

#ifdef __cplusplus
}
#endif

#endif /* FFT_KERNELS_H */
