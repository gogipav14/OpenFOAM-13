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
    Foam::OGL::OGLExecutor

Description
    Thread-safe singleton manager for Ginkgo GPU executors using Meyer's
    singleton pattern (C++11 thread-safe static initialization).

    Handles device selection and provides access to CPU and GPU executors
    for linear algebra operations.

    Supports multi-GPU via MPI rank-to-device mapping using environment
    variables (OMPI_COMM_WORLD_LOCAL_RANK, MPI_LOCALRANKID, etc.)

SourceFiles
    OGLExecutor.C

\*---------------------------------------------------------------------------*/

#ifndef OGLExecutor_H
#define OGLExecutor_H

#include "dictionary.H"
#include "label.H"

#include <memory>
#include <mutex>
#include <atomic>

// Forward declarations for Ginkgo types
namespace gko
{
    class Executor;
}

// Forward declaration
namespace Foam { namespace OGL { class GinkgoMemoryPool; } }

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                         Class OGLExecutor Declaration
\*---------------------------------------------------------------------------*/

class OGLExecutor
{
    // Private Data

        //- CPU reference executor
        std::shared_ptr<gko::Executor> cpuExecutor_;

        //- GPU executor (CUDA, HIP, or reference fallback)
        std::shared_ptr<gko::Executor> gpuExecutor_;

        //- Selected device index (-1 for auto)
        label deviceIndex_;

        //- Backend type (cuda, hip, omp, reference)
        word backendType_;

        //- Whether GPU is available and initialized
        bool gpuAvailable_;

        //- Debug level
        label debug_;

        //- Configuration dictionary (stored for reconfiguration)
        dictionary config_;

        //- Memory pool for GPU allocations
        std::unique_ptr<GinkgoMemoryPool> memoryPool_;

        //- Whether memory pool is enabled
        bool memoryPoolEnabled_;


    // Private Static Data

        //- Mutex for thread-safe configuration updates
        static std::mutex configMutex_;

        //- Configuration dictionary for lazy initialization
        static dictionary configDict_;

        //- Flag indicating if custom config was provided
        static std::atomic<bool> configProvided_;


    // Private Member Functions

        //- Detect local MPI rank for multi-GPU binding
        label detectLocalRank() const;

        //- Initialize executors with error handling
        void initializeExecutors();

        //- Disallow default bitwise copy construction
        OGLExecutor(const OGLExecutor&) = delete;

        //- Disallow default bitwise assignment
        void operator=(const OGLExecutor&) = delete;


public:

    // Constructors

        //- Construct from dictionary
        explicit OGLExecutor(const dictionary& dict);


    //- Destructor
    ~OGLExecutor();


    // Static Member Functions

        //- Configure the singleton before first use (thread-safe)
        //  Must be called before instance() if custom configuration needed
        static void configure(const dictionary& dict);

        //- Access the singleton instance (thread-safe, lazy initialization)
        //  Uses Meyer's singleton pattern for thread safety
        static OGLExecutor& instance();

        //- Check if singleton is initialized
        static bool initialized();


    // Member Functions

        // Access

            //- Return CPU executor
            std::shared_ptr<gko::Executor> cpuExecutor() const
            {
                return cpuExecutor_;
            }

            //- Return GPU executor (or CPU fallback if GPU unavailable)
            std::shared_ptr<gko::Executor> gpuExecutor() const
            {
                return gpuAvailable_ ? gpuExecutor_ : cpuExecutor_;
            }

            //- Return preferred executor for computation
            std::shared_ptr<gko::Executor> executor() const
            {
                return gpuExecutor();
            }

            //- Is GPU available?
            bool gpuAvailable() const
            {
                return gpuAvailable_;
            }

            //- Return backend type string
            const word& backendType() const
            {
                return backendType_;
            }

            //- Return device index
            label deviceIndex() const
            {
                return deviceIndex_;
            }

            //- Return debug level
            label debug() const
            {
                return debug_;
            }

            //- Access memory pool (may be null if disabled)
            GinkgoMemoryPool* memoryPool() const
            {
                return memoryPool_.get();
            }

            //- Check if memory pool is available
            bool hasMemoryPool() const
            {
                return memoryPool_ != nullptr;
            }


        // Utility

            //- Synchronize GPU operations (waits for all pending work)
            void synchronize() const;

            //- Print device information
            void printInfo() const;

            //- Check if Ginkgo operation succeeded, throw on failure
            static void checkGinkgoError(
                const std::string& operation,
                const std::exception& e
            );
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
