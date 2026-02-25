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
    C API for cylindrical FFT+Thomas preconditioner using cuFFT.

    Provides approximate inverse of the discrete Laplacian on structured
    cylindrical (r, theta) grids. The theta direction is decoupled by DFT
    (periodic), leaving independent tridiagonal systems in r for each
    Fourier mode. These are solved via pre-factored Thomas algorithm.

    Algorithm per apply:
    1. Forward DFT in theta (cuFFT R2C, batch=nr)
    2. Thomas solve in r for each Fourier mode (pre-factored)
    3. Inverse DFT in theta (cuFFT C2R, batch=nr)

\*---------------------------------------------------------------------------*/

#ifndef CYL_FFT_KERNELS_H
#define CYL_FFT_KERNELS_H

#ifdef __cplusplus
extern "C" {
#endif

/* Opaque handle for cylindrical FFT preconditioner state */
typedef struct CylFFTPrecondState* CylFFTHandle;

/*
 * Create cylindrical FFT preconditioner.
 *
 * nr:       number of radial cells
 * ntheta:   number of angular cells (should be even for efficiency)
 * useFloat: 1 for float (FP32), 0 for double (FP64)
 *
 * Data layout: r-major, theta varies fastest.
 *   data[i_r * ntheta + i_theta]
 *
 * Returns handle to preconditioner state, or NULL on failure.
 */
CylFFTHandle cylFFTPrecondCreate(
    int nr, int ntheta,
    int useFloat
);

/*
 * Destroy cylindrical FFT preconditioner and free all GPU resources.
 */
void cylFFTPrecondDestroy(CylFFTHandle h);

/*
 * Set tridiagonal coefficients from actual FV matrix couplings.
 *
 * The discrete FV Laplacian in cylindrical coordinates on a structured
 * (r, theta) mesh has:
 *   - Radial coupling: varies with r (face area proportional to r)
 *   - Angular coupling: varies with r (arc length proportional to r)
 *
 * After DFT in theta, the Fourier mode m has tridiagonal system:
 *   lower[i]*x[i-1] + diag[i,m]*x[i] + upper[i]*x[i+1] = rhs[i]
 * where:
 *   diag[i,m] = -(lower[i]+upper[i]) - 2*thetaCoeff[i]*(1-cos(2*pi*m/ntheta))
 *
 * lower:      radial coupling to inner neighbor, nr values (lower[0]=0)
 * upper:      radial coupling to outer neighbor, nr values (upper[nr-1]=0)
 * thetaCoeff: angular coupling at each radius, nr values
 *
 * All arrays are host pointers. Values are copied to GPU.
 * Thomas LU factorization is computed for each Fourier mode.
 */
void cylFFTPrecondSetCoeffs(
    CylFFTHandle h,
    const double* lower,
    const double* upper,
    const double* thetaCoeff
);

/*
 * Apply cylindrical FFT preconditioner in single precision.
 *
 * Computes: x = M^{-1} * b
 * Via: DFT in theta, Thomas solve in r per mode, IDFT in theta.
 *
 * b_ptr: device pointer to input vector (GPU memory, size n)
 * x_ptr: device pointer to output vector (GPU memory, size n)
 * n:     vector length (must equal nr*ntheta from create)
 */
void cylFFTPrecondApplyFloat(
    CylFFTHandle h,
    const float* b_ptr,
    float* x_ptr,
    int n
);

/*
 * Apply cylindrical FFT preconditioner in double precision.
 */
void cylFFTPrecondApplyDouble(
    CylFFTHandle h,
    const double* b_ptr,
    double* x_ptr,
    int n
);


/* ---- Per-sector DCT (Neumann boundary) API ---- */

/*
 * Create per-sector DCT preconditioner.
 *
 * For meshes with blade walls that break angular periodicity, each sector
 * between consecutive blades is an independent sub-problem with Neumann
 * (zero-flux) BCs at the blade boundaries. DCT-II naturally encodes
 * Neumann BCs, eliminating the need for Woodbury correction.
 *
 * nr:            number of radial cells
 * nSectors:      number of angular sectors (= number of blade walls)
 * nthetaSector:  angular cells per sector (all sectors must be same size)
 * useFloat:      1 for float (FP32), 0 for double (FP64)
 *
 * Data layout: sectors concatenated, each sector r-major.
 *   data[(s * nr + i_r) * nthetaSector + i_theta_local]
 *
 * Returns handle, or NULL on failure.
 */
CylFFTHandle cylFFTPrecondCreateSector(
    int nr, int nSectors, int nthetaSector,
    int useFloat
);

/*
 * Set tridiagonal coefficients for per-sector DCT mode.
 * Same interface as cylFFTPrecondSetCoeffs. Internally uses DCT-II
 * eigenvalues: eigenTheta[m] = 2*(1 - cos(pi*m/nthetaSector)).
 */
void cylFFTPrecondSetCoeffsSector(
    CylFFTHandle h,
    const double* lower,
    const double* upper,
    const double* thetaCoeff
);

/*
 * Apply per-sector DCT preconditioner in single precision.
 * n = nSectors * nr * nthetaSector.
 */
void cylFFTPrecondApplySectorFloat(
    CylFFTHandle h,
    const float* b_ptr,
    float* x_ptr,
    int n
);

/*
 * Apply per-sector DCT preconditioner in double precision.
 */
void cylFFTPrecondApplySectorDouble(
    CylFFTHandle h,
    const double* b_ptr,
    double* x_ptr,
    int n
);

#ifdef __cplusplus
}
#endif

#endif /* CYL_FFT_KERNELS_H */
