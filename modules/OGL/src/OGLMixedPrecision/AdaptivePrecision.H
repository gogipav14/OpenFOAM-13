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
    Foam::OGL::AdaptivePrecision

Description
    Adaptive precision strategy for mixed-precision iterative solvers.

    Automatically switches between FP32 and FP64 based on:
    - Residual reduction rate (stagnation detection)
    - Target tolerance proximity
    - Condition number estimates

    Strategies:
    - FIXED_FP32: Always use FP32 (fastest, least accurate)
    - FIXED_FP64: Always use FP64 (slowest, most accurate)
    - RESIDUAL_BASED: Switch based on residual stagnation
    - TOLERANCE_BASED: Switch when approaching target tolerance
    - HYBRID: Combine stagnation and tolerance detection

    The residual-based approach monitors:
    - r_k / r_{k-1} ratio (convergence rate)
    - Stagnation: ratio > stagnationThreshold for N iterations
    - Triggers switch to FP64 when stagnation detected

    Configuration:
    \verbatim
    OGLCoeffs
    {
        adaptivePrecision
        {
            strategy            RESIDUAL_BASED;
            stagnationThreshold 0.95;
            stagnationCount     3;
            toleranceBuffer     10.0;  // Switch at 10x target tolerance
        }
    }
    \endverbatim

SourceFiles
    AdaptivePrecision.C

\*---------------------------------------------------------------------------*/

#ifndef AdaptivePrecision_H
#define AdaptivePrecision_H

#include "dictionary.H"
#include "NamedEnum.H"

#include <deque>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                     Class AdaptivePrecision Declaration
\*---------------------------------------------------------------------------*/

class AdaptivePrecision
{
public:

    // Public Enums

        //- Precision strategy types
        enum class Strategy
        {
            FIXED_FP32,
            FIXED_FP64,
            RESIDUAL_BASED,
            TOLERANCE_BASED,
            HYBRID
        };

        //- Current precision recommendation
        enum class Precision
        {
            FP32,
            FP64
        };


private:

    // Private Data

        //- Strategy to use
        Strategy strategy_;

        //- Threshold for detecting stagnation (ratio > threshold = stagnating)
        scalar stagnationThreshold_;

        //- Number of stagnating iterations before switching
        label stagnationCount_;

        //- Buffer factor for tolerance-based switching
        scalar toleranceBuffer_;

        //- Target tolerance
        scalar targetTolerance_;

        //- History of residual reduction ratios
        std::deque<scalar> residualHistory_;

        //- Maximum history length
        label maxHistoryLength_;

        //- Current recommended precision
        Precision currentPrecision_;

        //- Statistics: iterations in FP32
        mutable label fp32Iterations_;

        //- Statistics: iterations in FP64
        mutable label fp64Iterations_;

        //- Statistics: number of precision switches
        mutable label precisionSwitches_;


public:

    // Static Data

        //- Strategy names
        static const NamedEnum<Strategy, 5> strategyNames;


    // Constructors

        //- Construct from dictionary
        AdaptivePrecision(const dictionary& dict);

        //- Construct with default settings
        AdaptivePrecision();


    //- Destructor
    ~AdaptivePrecision() = default;


    // Member Functions

        // Configuration

            //- Set target tolerance
            void setTargetTolerance(scalar tol) { targetTolerance_ = tol; }

            //- Set strategy
            void setStrategy(Strategy s) { strategy_ = s; }

            //- Reset state for new solve
            void reset();


        // Precision Selection

            //- Update with new residual and return recommended precision
            Precision update(scalar residual, label iteration);

            //- Get current recommended precision
            Precision currentPrecision() const { return currentPrecision_; }

            //- Check if using FP32
            bool useFP32() const { return currentPrecision_ == Precision::FP32; }

            //- Check if using FP64
            bool useFP64() const { return currentPrecision_ == Precision::FP64; }


        // Analysis

            //- Check if convergence is stagnating
            bool isStagnating() const;

            //- Check if close to target tolerance
            bool nearTolerance(scalar residual) const;

            //- Estimate convergence rate from history
            scalar estimateConvergenceRate() const;


        // Statistics

            //- Report statistics
            void report() const;

            //- Get FP32 iteration count
            label fp32Iterations() const { return fp32Iterations_; }

            //- Get FP64 iteration count
            label fp64Iterations() const { return fp64Iterations_; }
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
