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

#include "PerformanceTimer.H"
#include "Pstream.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

const std::map<Foam::OGL::PerformanceTimer::Category, std::string>
Foam::OGL::PerformanceTimer::categoryNames_ =
{
    {Category::CONVERT_STRUCTURE, "convertStructure"},
    {Category::UPDATE_VALUES, "updateValues"},
    {Category::HALO_EXCHANGE, "haloExchange"},
    {Category::APPLY_LOCAL_SPMV, "applyLocalSpMV"},
    {Category::APPLY_INTERFACE, "applyInterface"},
    {Category::COPY_TO_DEVICE, "copyToDevice"},
    {Category::COPY_FROM_DEVICE, "copyFromDevice"},
    {Category::SOLVE_KERNEL, "solveKernel"},
    {Category::TOTAL_SOLVE, "totalSolve"},
    {Category::REFINEMENT_RESIDUAL, "refinementResidual"}
};


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

const std::string& Foam::OGL::PerformanceTimer::categoryName(Category cat)
{
    return categoryNames_.at(cat);
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::PerformanceTimer::PerformanceTimer(bool enabled)
:
    enabled_(enabled)
{
    reset();
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::OGL::PerformanceTimer::reset()
{
    accumulatedTimes_.clear();
    callCounts_.clear();

    // Initialize all categories to zero
    for (const auto& pair : categoryNames_)
    {
        accumulatedTimes_[pair.first] = 0.0;
        callCounts_[pair.first] = 0;
    }
}


void Foam::OGL::PerformanceTimer::start(Category cat)
{
    if (!enabled_) return;

    startTimes_[cat] = clockTime();
}


void Foam::OGL::PerformanceTimer::stop(Category cat)
{
    if (!enabled_) return;

    auto it = startTimes_.find(cat);
    if (it != startTimes_.end())
    {
        double elapsed = it->second.elapsedTime();
        accumulatedTimes_[cat] += elapsed;
        callCounts_[cat]++;
    }
}


double Foam::OGL::PerformanceTimer::time(Category cat) const
{
    auto it = accumulatedTimes_.find(cat);
    return (it != accumulatedTimes_.end()) ? it->second : 0.0;
}


Foam::label Foam::OGL::PerformanceTimer::count(Category cat) const
{
    auto it = callCounts_.find(cat);
    return (it != callCounts_.end()) ? it->second : 0;
}


double Foam::OGL::PerformanceTimer::averageTime(Category cat) const
{
    label n = count(cat);
    return (n > 0) ? time(cat) / n : 0.0;
}


void Foam::OGL::PerformanceTimer::report() const
{
    report("OGL");
}


void Foam::OGL::PerformanceTimer::report(const word& fieldName) const
{
    if (!enabled_) return;

    Info<< nl << "=== OGL Performance Report: " << fieldName << " ===" << nl;

    double totalTime = time(Category::TOTAL_SOLVE);
    label totalCalls = count(Category::TOTAL_SOLVE);

    Info<< "Total solves: " << totalCalls << nl;
    Info<< "Total time:   " << totalTime << " s" << nl;

    if (totalCalls > 0)
    {
        Info<< "Avg per solve: " << totalTime / totalCalls * 1000 << " ms" << nl;
    }

    Info<< nl << "Breakdown:" << nl;

    // Print each category as percentage of total
    for (const auto& pair : categoryNames_)
    {
        Category cat = pair.first;
        if (cat == Category::TOTAL_SOLVE) continue;

        double t = time(cat);
        label n = count(cat);

        if (n > 0)
        {
            double pct = (totalTime > 0) ? 100.0 * t / totalTime : 0.0;
            double avg = t / n * 1000.0;  // ms

            Info<< "  " << pair.second << ": "
                << t << " s (" << pct << "%), "
                << n << " calls, "
                << avg << " ms/call" << nl;
        }
    }

    Info<< "=============================================" << nl << endl;
}


Foam::dictionary Foam::OGL::PerformanceTimer::timingDict() const
{
    dictionary dict;

    for (const auto& pair : categoryNames_)
    {
        dictionary catDict;
        catDict.add("time", time(pair.first));
        catDict.add("count", count(pair.first));
        catDict.add("average", averageTime(pair.first));
        dict.add(pair.second, catDict);
    }

    return dict;
}


void Foam::OGL::PerformanceTimer::reduce()
{
    if (!Pstream::parRun()) return;

    // Reduce times (take max across processors for bottleneck analysis)
    for (auto& pair : accumulatedTimes_)
    {
        Foam::reduce(pair.second, maxOp<double>());
    }

    // Reduce counts (take max)
    for (auto& pair : callCounts_)
    {
        Foam::reduce(pair.second, maxOp<label>());
    }
}


// ************************************************************************* //
