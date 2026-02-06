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

\*---------------------------------------------------------------------------*/

#include "lduToCSR.H"
#include "lduAddressing.H"

#include <algorithm>
#include <numeric>
#include <limits>

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

void Foam::OGL::lduToCSR::buildStructure()
{
    const lduAddressing& addr = matrix_.lduAddr();
    const labelUList& upperAddr = addr.upperAddr();
    const labelUList& lowerAddr = addr.lowerAddr();

    nRows_ = matrix_.diag().size();
    const label nFaces = upperAddr.size();

    // Check for int32 overflow (Ginkgo uses int for indices)
    // Max non-zeros = nRows (diagonal) + 2*nFaces (upper + lower)
    const label maxNonZeros = nRows_ + 2*nFaces;
    if (nRows_ > std::numeric_limits<int>::max() ||
        maxNonZeros > std::numeric_limits<int>::max())
    {
        FatalErrorInFunction
            << "Matrix too large for Ginkgo int32 indexing: "
            << "nRows = " << nRows_ << ", max nnz = " << maxNonZeros
            << abort(FatalError);
    }

    // Count non-zeros per row:
    // Each row has: 1 diagonal + number of off-diagonals
    // Off-diagonals come from both upper (where this row is lowerAddr)
    // and lower (where this row is upperAddr)

    labelList rowCounts(nRows_, 1);  // Start with diagonal

    forAll(upperAddr, facei)
    {
        // Upper coefficient: row = lowerAddr[facei], col = upperAddr[facei]
        rowCounts[lowerAddr[facei]]++;

        // Lower coefficient: row = upperAddr[facei], col = lowerAddr[facei]
        rowCounts[upperAddr[facei]]++;
    }

    // Build row pointers
    rowPointers_.resize(nRows_ + 1);
    rowPointers_[0] = 0;
    for (label i = 0; i < nRows_; i++)
    {
        rowPointers_[i + 1] = rowPointers_[i] + rowCounts[i];
    }

    nNonZeros_ = rowPointers_[nRows_];
    colIndices_.resize(nNonZeros_);
    valuesF64_.resize(nNonZeros_);
    valuesF32_.resize(nNonZeros_);

    // Initialize mapping arrays
    diagToCsr_.setSize(nRows_);
    upperToCsr_.setSize(nFaces);
    lowerToCsr_.setSize(nFaces);

    // Temporary array to track current position in each row
    labelList currentPos(nRows_, 0);

    // Structure to hold (column, csrPosition) pairs for sorting
    struct ColPos
    {
        int col;
        label csrPos;
        label lduPos;  // Position in LDU array (-1 for diagonal)
        int type;      // 0=diag, 1=upper, 2=lower
    };

    // For each row, collect all entries and sort by column
    List<List<ColPos>> rowEntries(nRows_);

    // Reserve space for entries
    for (label row = 0; row < nRows_; row++)
    {
        rowEntries[row].setSize(rowCounts[row]);
        currentPos[row] = 0;
    }

    // Add diagonal entries
    for (label row = 0; row < nRows_; row++)
    {
        ColPos& entry = rowEntries[row][currentPos[row]++];
        entry.col = row;
        entry.lduPos = row;
        entry.type = 0;  // diagonal
    }

    // Add upper entries (row = lowerAddr, col = upperAddr)
    forAll(upperAddr, facei)
    {
        label row = lowerAddr[facei];
        ColPos& entry = rowEntries[row][currentPos[row]++];
        entry.col = upperAddr[facei];
        entry.lduPos = facei;
        entry.type = 1;  // upper
    }

    // Add lower entries (row = upperAddr, col = lowerAddr)
    forAll(lowerAddr, facei)
    {
        label row = upperAddr[facei];
        ColPos& entry = rowEntries[row][currentPos[row]++];
        entry.col = lowerAddr[facei];
        entry.lduPos = facei;
        entry.type = 2;  // lower
    }

    // Sort each row by column index and fill colIndices and mappings
    label csrPos = 0;
    for (label row = 0; row < nRows_; row++)
    {
        List<ColPos>& entries = rowEntries[row];

        // Sort by column index
        std::sort
        (
            entries.begin(),
            entries.end(),
            [](const ColPos& a, const ColPos& b) { return a.col < b.col; }
        );

        // Fill column indices and build mappings
        forAll(entries, i)
        {
            const ColPos& entry = entries[i];
            colIndices_[csrPos] = entry.col;

            // Record mapping from LDU to CSR
            switch (entry.type)
            {
                case 0:  // diagonal
                    diagToCsr_[entry.lduPos] = csrPos;
                    break;
                case 1:  // upper
                    upperToCsr_[entry.lduPos] = csrPos;
                    break;
                case 2:  // lower
                    lowerToCsr_[entry.lduPos] = csrPos;
                    break;
            }

            csrPos++;
        }
    }

    structureBuilt_ = true;
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::lduToCSR::lduToCSR(const lduMatrix& matrix)
:
    matrix_(matrix),
    nRows_(0),
    nNonZeros_(0),
    structureBuilt_(false)
{
    buildStructure();
    updateValues();
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::OGL::lduToCSR::rebuildStructure()
{
    structureBuilt_ = false;
    buildStructure();
    updateValues();
}


void Foam::OGL::lduToCSR::updateValues()
{
    if (!structureBuilt_)
    {
        buildStructure();
    }

    const scalarField& diag = matrix_.diag();
    const scalarField& upper = matrix_.upper();

    // Lower is same as upper for symmetric matrices, or separate for asymmetric
    const scalarField& lower = matrix_.hasLower() ? matrix_.lower() : upper;

    // Fill diagonal values
    forAll(diag, i)
    {
        valuesF64_[diagToCsr_[i]] = diag[i];
    }

    // Fill upper values
    forAll(upper, i)
    {
        valuesF64_[upperToCsr_[i]] = upper[i];
    }

    // Fill lower values
    forAll(lower, i)
    {
        valuesF64_[lowerToCsr_[i]] = lower[i];
    }
}


void Foam::OGL::lduToCSR::updateValuesF32()
{
    // First update F64 values
    updateValues();

    // Cast to F32
    for (label i = 0; i < nNonZeros_; i++)
    {
        valuesF32_[i] = static_cast<float>(valuesF64_[i]);
    }
}


std::shared_ptr<Foam::OGL::lduToCSR::CsrMatrixF64>
Foam::OGL::lduToCSR::createGinkgoMatrixF64
(
    std::shared_ptr<const gko::Executor> exec
) const
{
    if (!structureBuilt_)
    {
        FatalErrorInFunction
            << "CSR structure not built"
            << abort(FatalError);
    }

    // Note: const_cast is required here because Ginkgo's array::view() requires
    // non-const pointers, but we immediately call .copy_to_array() which creates
    // a copy. The original data is never modified. This is a safe pattern:
    // view -> copy_to_array -> move to GPU executor.

    auto rowPtrs = gko::array<int>::view(
        exec->get_master(),
        nRows_ + 1,
        const_cast<int*>(rowPointers_.data())
    ).copy_to_array();

    auto colIdxs = gko::array<int>::view(
        exec->get_master(),
        nNonZeros_,
        const_cast<int*>(colIndices_.data())
    ).copy_to_array();

    auto vals = gko::array<double>::view(
        exec->get_master(),
        nNonZeros_,
        const_cast<double*>(valuesF64_.data())
    ).copy_to_array();

    // Move arrays to target executor and create matrix
    rowPtrs.set_executor(exec);
    colIdxs.set_executor(exec);
    vals.set_executor(exec);

    return CsrMatrixF64::create(
        exec,
        gko::dim<2>(nRows_, nRows_),
        std::move(vals),
        std::move(colIdxs),
        std::move(rowPtrs)
    );
}


std::shared_ptr<Foam::OGL::lduToCSR::CsrMatrixF32>
Foam::OGL::lduToCSR::createGinkgoMatrixF32
(
    std::shared_ptr<const gko::Executor> exec
) const
{
    if (!structureBuilt_)
    {
        FatalErrorInFunction
            << "CSR structure not built"
            << abort(FatalError);
    }

    // Note: const_cast is required here because Ginkgo's array::view() requires
    // non-const pointers, but we immediately call .copy_to_array() which creates
    // a copy. The original data is never modified. This is a safe pattern:
    // view -> copy_to_array -> move to GPU executor.

    auto rowPtrs = gko::array<int>::view(
        exec->get_master(),
        nRows_ + 1,
        const_cast<int*>(rowPointers_.data())
    ).copy_to_array();

    auto colIdxs = gko::array<int>::view(
        exec->get_master(),
        nNonZeros_,
        const_cast<int*>(colIndices_.data())
    ).copy_to_array();

    auto vals = gko::array<float>::view(
        exec->get_master(),
        nNonZeros_,
        const_cast<float*>(valuesF32_.data())
    ).copy_to_array();

    // Move arrays to target executor and create matrix
    rowPtrs.set_executor(exec);
    colIdxs.set_executor(exec);
    vals.set_executor(exec);

    return CsrMatrixF32::create(
        exec,
        gko::dim<2>(nRows_, nRows_),
        std::move(vals),
        std::move(colIdxs),
        std::move(rowPtrs)
    );
}


// ************************************************************************* //
