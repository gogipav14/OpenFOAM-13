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

#include "AdaptivePrecision.H"
#include "messageStream.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{
    template<>
    const char* NamedEnum<AdaptivePrecision::Strategy, 5>::names[] =
    {
        "FIXED_FP32",
        "FIXED_FP64",
        "RESIDUAL_BASED",
        "TOLERANCE_BASED",
        "HYBRID"
    };
}
}

const Foam::NamedEnum<Foam::OGL::AdaptivePrecision::Strategy, 5>
Foam::OGL::AdaptivePrecision::strategyNames;


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::AdaptivePrecision::AdaptivePrecision(const dictionary& dict)
:
    strategy_
    (
        dict.found("strategy")
      ? strategyNames.read(dict.lookup("strategy"))
      : Strategy::RESIDUAL_BASED
    ),
    stagnationThreshold_(dict.lookupOrDefault<scalar>("stagnationThreshold", 0.95)),
    stagnationCount_(dict.lookupOrDefault<label>("stagnationCount", 3)),
    toleranceBuffer_(dict.lookupOrDefault<scalar>("toleranceBuffer", 10.0)),
    targetTolerance_(1e-6),
    maxHistoryLength_(10),
    currentPrecision_(Precision::FP32),
    fp32Iterations_(0),
    fp64Iterations_(0),
    precisionSwitches_(0)
{}


Foam::OGL::AdaptivePrecision::AdaptivePrecision()
:
    strategy_(Strategy::RESIDUAL_BASED),
    stagnationThreshold_(0.95),
    stagnationCount_(3),
    toleranceBuffer_(10.0),
    targetTolerance_(1e-6),
    maxHistoryLength_(10),
    currentPrecision_(Precision::FP32),
    fp32Iterations_(0),
    fp64Iterations_(0),
    precisionSwitches_(0)
{}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::OGL::AdaptivePrecision::reset()
{
    residualHistory_.clear();
    currentPrecision_ = Precision::FP32;  // Start in FP32
}


Foam::OGL::AdaptivePrecision::Precision
Foam::OGL::AdaptivePrecision::update(scalar residual, label iteration)
{
    // Track previous residual for ratio calculation
    scalar prevResidual = residualHistory_.empty()
        ? residual
        : residualHistory_.back();

    // Calculate reduction ratio
    scalar ratio = (prevResidual > SMALL) ? residual / prevResidual : 1.0;

    // Add to history
    residualHistory_.push_back(ratio);
    if (residualHistory_.size() > static_cast<size_t>(maxHistoryLength_))
    {
        residualHistory_.pop_front();
    }

    // Determine precision based on strategy
    Precision newPrecision = currentPrecision_;

    switch (strategy_)
    {
        case Strategy::FIXED_FP32:
            newPrecision = Precision::FP32;
            break;

        case Strategy::FIXED_FP64:
            newPrecision = Precision::FP64;
            break;

        case Strategy::RESIDUAL_BASED:
            if (currentPrecision_ == Precision::FP32 && isStagnating())
            {
                newPrecision = Precision::FP64;
            }
            break;

        case Strategy::TOLERANCE_BASED:
            if (nearTolerance(residual))
            {
                newPrecision = Precision::FP64;
            }
            break;

        case Strategy::HYBRID:
            if (currentPrecision_ == Precision::FP32)
            {
                if (isStagnating() || nearTolerance(residual))
                {
                    newPrecision = Precision::FP64;
                }
            }
            break;
    }

    // Track statistics
    if (newPrecision != currentPrecision_)
    {
        precisionSwitches_++;
    }

    if (newPrecision == Precision::FP32)
    {
        fp32Iterations_++;
    }
    else
    {
        fp64Iterations_++;
    }

    currentPrecision_ = newPrecision;
    return currentPrecision_;
}


bool Foam::OGL::AdaptivePrecision::isStagnating() const
{
    if (residualHistory_.size() < static_cast<size_t>(stagnationCount_))
    {
        return false;
    }

    // Check if last N iterations show poor convergence
    label stagnatingCount = 0;
    auto it = residualHistory_.rbegin();
    for (label i = 0; i < stagnationCount_ && it != residualHistory_.rend(); ++i, ++it)
    {
        if (*it > stagnationThreshold_)
        {
            stagnatingCount++;
        }
    }

    return stagnatingCount >= stagnationCount_;
}


bool Foam::OGL::AdaptivePrecision::nearTolerance(scalar residual) const
{
    return residual < toleranceBuffer_ * targetTolerance_;
}


Foam::scalar Foam::OGL::AdaptivePrecision::estimateConvergenceRate() const
{
    if (residualHistory_.size() < 2)
    {
        return 1.0;
    }

    // Average of recent ratios
    scalar sum = 0;
    for (const auto& ratio : residualHistory_)
    {
        sum += ratio;
    }
    return sum / residualHistory_.size();
}


void Foam::OGL::AdaptivePrecision::report() const
{
    Info<< "AdaptivePrecision statistics:" << nl
        << "  Strategy: " << strategyNames[strategy_] << nl
        << "  FP32 iterations: " << fp32Iterations_ << nl
        << "  FP64 iterations: " << fp64Iterations_ << nl
        << "  Precision switches: " << precisionSwitches_ << nl
        << "  Current precision: "
        << (currentPrecision_ == Precision::FP32 ? "FP32" : "FP64") << nl
        << "  Estimated convergence rate: " << estimateConvergenceRate() << nl
        << endl;
}


// ************************************************************************* //
