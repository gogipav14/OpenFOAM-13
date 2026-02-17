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

Class
    Foam::OGL::lduToCSR

Description
    Converts OpenFOAM's LDU (Lower-Diagonal-Upper) sparse matrix format
    to CSR (Compressed Sparse Row) format for use with Ginkgo.

    The LDU format stores:
    - diag: diagonal coefficients (size nCells)
    - upper: upper triangular coefficients (size nFaces)
    - lower: lower triangular coefficients (size nFaces, or same as upper if symmetric)
    - upperAddr: row indices for upper coefficients
    - lowerAddr: column indices for upper coefficients (or equivalently row for lower)

    The CSR format stores:
    - values: all non-zero values in row-major order
    - colIndices: column index for each value
    - rowPointers: start index in values/colIndices for each row

    This conversion handles the LOCAL part of the matrix only.
    Interface contributions (processor/cyclic/NCC) are handled separately
    in the FoamGinkgoLinOp::apply() method.

SourceFiles
    lduToCSR.C

\*---------------------------------------------------------------------------*/

#ifndef lduToCSR_H
#define lduToCSR_H

#include "lduMatrix.H"
#include "scalarField.H"
#include "labelList.H"

#include "GinkgoCompat.H"
#include <memory>
#include <vector>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                         Class lduToCSR Declaration
\*---------------------------------------------------------------------------*/

class lduToCSR
{
public:

    // Public Types

        //- CSR matrix type for single precision
        using CsrMatrixF32 = gko::matrix::Csr<float, int>;

        //- CSR matrix type for double precision
        using CsrMatrixF64 = gko::matrix::Csr<double, int>;


private:

    // Private Data

        //- Reference to the lduMatrix
        const lduMatrix& matrix_;

        //- Number of rows (cells)
        label nRows_;

        //- Number of non-zeros
        label nNonZeros_;

        //- Row pointers (CSR format)
        std::vector<int> rowPointers_;

        //- Column indices (CSR format)
        std::vector<int> colIndices_;

        //- Values in double precision
        std::vector<double> valuesF64_;

        //- Values in single precision
        std::vector<float> valuesF32_;

        //- Structure has been built
        bool structureBuilt_;

        //- Mapping from LDU position to CSR value position
        //  Used for efficient value updates without rebuilding structure
        labelList diagToCsr_;
        labelList upperToCsr_;
        labelList lowerToCsr_;


    // Private Member Functions

        //- Build CSR structure (row pointers, column indices)
        void buildStructure();

        //- Disallow default bitwise copy construction
        lduToCSR(const lduToCSR&) = delete;

        //- Disallow default bitwise assignment
        void operator=(const lduToCSR&) = delete;


public:

    // Constructors

        //- Construct from lduMatrix
        explicit lduToCSR(const lduMatrix& matrix);


    //- Destructor
    ~lduToCSR() = default;


    // Member Functions

        // Structure

            //- Return number of rows
            label nRows() const { return nRows_; }

            //- Return number of non-zeros
            label nNonZeros() const { return nNonZeros_; }

            //- Is structure built?
            bool structureBuilt() const { return structureBuilt_; }

            //- Force rebuild of structure
            void rebuildStructure();


        // Value Updates

            //- Update values from lduMatrix (structure must be built)
            void updateValues();

            //- Update values and cast to single precision
            void updateValuesF32();


        // Matrix Creation

            //- Create Ginkgo CSR matrix in double precision
            std::shared_ptr<CsrMatrixF64> createGinkgoMatrixF64
            (
                std::shared_ptr<const gko::Executor> exec
            ) const;

            //- Create Ginkgo CSR matrix in single precision
            std::shared_ptr<CsrMatrixF32> createGinkgoMatrixF32
            (
                std::shared_ptr<const gko::Executor> exec
            ) const;

            //- Update values of an existing GPU CSR matrix in-place
            //  Only copies the values array (no structure copy).
            //  The matrix must have been previously created with
            //  createGinkgoMatrixF64/F32.
            void updateValuesInPlaceF64
            (
                std::shared_ptr<CsrMatrixF64> existingMatrix
            );

            void updateValuesInPlaceF32
            (
                std::shared_ptr<CsrMatrixF32> existingMatrix
            );


        // Access

            //- Access row pointers
            const std::vector<int>& rowPointers() const { return rowPointers_; }

            //- Access column indices
            const std::vector<int>& colIndices() const { return colIndices_; }

            //- Access values (F64)
            const std::vector<double>& valuesF64() const { return valuesF64_; }

            //- Access values (F32)
            const std::vector<float>& valuesF32() const { return valuesF32_; }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
