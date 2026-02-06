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
    Foam::OGL::PerformanceTimer

Description
    Performance timing instrumentation for OGL GPU solver operations.

    Tracks key performance metrics:
    - T_convertStructure: CSR rowPtr/colIdx build (one-time per topology)
    - T_updateValues: Coefficient value refresh per solve
    - T_haloExchange: MPI pack/exchange time
    - T_applyLocalSpMV: GPU SpMV kernel time
    - T_applyInterface: Interface contribution (CPU)
    - T_copyToDevice: Host to GPU transfer
    - T_copyFromDevice: GPU to host transfer
    - T_solveKernel: Ginkgo solver kernel time
    - T_totalSolve: End-to-end solve time

SourceFiles
    PerformanceTimer.C

\*---------------------------------------------------------------------------*/

#ifndef PerformanceTimer_H
#define PerformanceTimer_H

#include "clockTime.H"
#include "cpuTime.H"
#include "dictionary.H"
#include "Pstream.H"

#include <map>
#include <string>
#include <vector>
#include <mutex>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                      Class PerformanceTimer Declaration
\*---------------------------------------------------------------------------*/

class PerformanceTimer
{
public:

    // Timer categories
    enum class Category
    {
        CONVERT_STRUCTURE,
        UPDATE_VALUES,
        HALO_EXCHANGE,
        APPLY_LOCAL_SPMV,
        APPLY_INTERFACE,
        COPY_TO_DEVICE,
        COPY_FROM_DEVICE,
        SOLVE_KERNEL,
        TOTAL_SOLVE,
        REFINEMENT_RESIDUAL
    };


private:

    // Private Data

        //- Mutex for thread-safe map access
        mutable std::mutex mutex_;

        //- Whether timing is enabled
        bool enabled_;

        //- Current timer start times (wall clock)
        std::map<Category, clockTime> startTimes_;

        //- Accumulated times per category
        std::map<Category, double> accumulatedTimes_;

        //- Call counts per category
        std::map<Category, label> callCounts_;

        //- Category names for output
        static const std::map<Category, std::string> categoryNames_;


    // Private Member Functions

        //- Get category name
        static const std::string& categoryName(Category cat);


public:

    // Constructors

        //- Construct with enable flag
        explicit PerformanceTimer(bool enabled = false);


    //- Destructor
    ~PerformanceTimer() = default;


    // Member Functions

        // Control

            //- Enable/disable timing
            void setEnabled(bool enabled) { enabled_ = enabled; }

            //- Check if enabled
            bool enabled() const { return enabled_; }

            //- Reset all timers
            void reset();


        // Timing

            //- Start timer for category
            void start(Category cat);

            //- Stop timer for category and accumulate
            void stop(Category cat);

            //- Get accumulated time for category
            double time(Category cat) const;

            //- Get call count for category
            label count(Category cat) const;

            //- Get average time per call for category
            double averageTime(Category cat) const;


        // Reporting

            //- Print timing summary to Info
            void report() const;

            //- Print timing summary with field name
            void report(const word& fieldName) const;

            //- Get timing data as dictionary
            dictionary timingDict() const;


        // Parallel

            //- Reduce timings across all processors (max)
            void reduce();
};


/*---------------------------------------------------------------------------*\
                      Class ScopedTimer Declaration
\*---------------------------------------------------------------------------*/

//- RAII timer that stops on destruction
class ScopedTimer
{
    PerformanceTimer& timer_;
    PerformanceTimer::Category category_;
    bool active_;

public:

    //- Construct and start timer
    ScopedTimer
    (
        PerformanceTimer& timer,
        PerformanceTimer::Category cat
    )
    :
        timer_(timer),
        category_(cat),
        active_(timer.enabled())
    {
        if (active_)
        {
            timer_.start(category_);
        }
    }

    //- Destructor stops timer
    ~ScopedTimer()
    {
        if (active_)
        {
            timer_.stop(category_);
        }
    }

    // No copy
    ScopedTimer(const ScopedTimer&) = delete;
    ScopedTimer& operator=(const ScopedTimer&) = delete;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
