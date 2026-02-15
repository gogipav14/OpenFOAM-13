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

#include "OGLExecutor.H"
#include "GinkgoMemoryPool.H"
#include "messageStream.H"
#include "Pstream.H"

#include "GinkgoCompat.H"
#include <cstdlib>
#include <stdexcept>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

std::mutex Foam::OGL::OGLExecutor::configMutex_;
Foam::dictionary Foam::OGL::OGLExecutor::configDict_;
std::atomic<bool> Foam::OGL::OGLExecutor::configProvided_(false);

// Flag to track initialization (for initialized() check)
namespace
{
    std::atomic<bool> instanceCreated_(false);
}


// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

Foam::label Foam::OGL::OGLExecutor::detectLocalRank() const
{
    // Try various environment variables for local rank detection
    // Used for multi-GPU: each MPI rank binds to a different GPU

    const char* localRankEnv = nullptr;

    // OpenMPI
    localRankEnv = std::getenv("OMPI_COMM_WORLD_LOCAL_RANK");
    if (localRankEnv)
    {
        return std::atoi(localRankEnv);
    }

    // MPICH
    localRankEnv = std::getenv("MPI_LOCALRANKID");
    if (localRankEnv)
    {
        return std::atoi(localRankEnv);
    }

    // MVAPICH2
    localRankEnv = std::getenv("MV2_COMM_WORLD_LOCAL_RANK");
    if (localRankEnv)
    {
        return std::atoi(localRankEnv);
    }

    // Slurm
    localRankEnv = std::getenv("SLURM_LOCALID");
    if (localRankEnv)
    {
        return std::atoi(localRankEnv);
    }

    // Intel MPI
    localRankEnv = std::getenv("PMI_LOCAL_RANK");
    if (localRankEnv)
    {
        return std::atoi(localRankEnv);
    }

    // Fallback: use global rank mod expected GPUs
    if (debug_ > 0)
    {
        Info<< "OGLExecutor: No local rank environment variable found, "
            << "using global rank for device selection" << endl;
    }

    return Pstream::myProcNo();
}


void Foam::OGL::OGLExecutor::initializeExecutors()
{
    // Always create CPU reference executor first
    try
    {
        cpuExecutor_ = gko::ReferenceExecutor::create();
    }
    catch (const std::exception& e)
    {
        FatalErrorInFunction
            << "Failed to create Ginkgo reference executor: " << e.what()
            << abort(FatalError);
    }

    backendType_ = "reference";
    gpuAvailable_ = false;

    // Determine device index
    if (deviceIndex_ < 0)
    {
        // Auto-select based on local MPI rank
        deviceIndex_ = detectLocalRank();
    }

    // Try to create GPU executor using Ginkgo runtime detection.
    // Ginkgo headers always declare CudaExecutor/HipExecutor; the actual
    // GPU code lives in libginkgo_cuda.so / libginkgo_hip.so.  If no
    // device is present, create() throws and we fall through gracefully.

    // Try CUDA first
    try
    {
        gpuExecutor_ = gko::CudaExecutor::create(
            deviceIndex_,
            cpuExecutor_
        );

        backendType_ = "cuda";
        gpuAvailable_ = true;

        if (debug_ > 0)
        {
            Info<< "OGLExecutor: CUDA device " << deviceIndex_
                << " initialized" << endl;
        }
    }
    catch (const std::exception& e)
    {
        if (debug_ > 0)
        {
            Info<< "OGLExecutor: CUDA not available (" << e.what()
                << "), trying HIP..." << endl;
        }
    }

    // Try HIP if CUDA failed
    if (!gpuAvailable_)
    {
        try
        {
            gpuExecutor_ = gko::HipExecutor::create(
                deviceIndex_,
                cpuExecutor_
            );

            backendType_ = "hip";
            gpuAvailable_ = true;

            if (debug_ > 0)
            {
                Info<< "OGLExecutor: HIP device " << deviceIndex_
                    << " initialized" << endl;
            }
        }
        catch (const std::exception& e)
        {
            if (debug_ > 0)
            {
                Info<< "OGLExecutor: HIP not available (" << e.what()
                    << "), falling back to CPU" << endl;
            }
        }
    }

    // If no GPU available, use OMP executor for better CPU performance
    if (!gpuAvailable_)
    {
        try
        {
            gpuExecutor_ = gko::OmpExecutor::create();
            backendType_ = "omp";

            if (debug_ > 0)
            {
                Info<< "OGLExecutor: Using OpenMP executor (CPU)" << endl;
            }
        }
        catch (const std::exception& e)
        {
            gpuExecutor_ = cpuExecutor_;
            backendType_ = "reference";

            if (debug_ > 0)
            {
                Info<< "OGLExecutor: Using reference executor (CPU)" << endl;
            }
        }
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::OGLExecutor::OGLExecutor(const dictionary& dict)
:
    cpuExecutor_(nullptr),
    gpuExecutor_(nullptr),
    deviceIndex_(dict.lookupOrDefault<label>("deviceIndex", -1)),
    backendType_("none"),
    gpuAvailable_(false),
    debug_(dict.lookupOrDefault<label>("debug", 0)),
    config_(dict),
    memoryPool_(nullptr),
    memoryPoolEnabled_(false)
{
    initializeExecutors();

    // Initialize memory pool if enabled
    memoryPoolEnabled_ = dict.lookupOrDefault<bool>("enableMemoryPool", true);

    if (memoryPoolEnabled_ && gpuExecutor_)
    {
        dictionary poolDict;
        if (dict.found("memoryPool"))
        {
            poolDict = dict.subDict("memoryPool");
        }

        try
        {
            memoryPool_ = std::make_unique<GinkgoMemoryPool>
            (
                gpuExecutor_,
                poolDict
            );

            if (debug_ > 0)
            {
                Info<< "OGLExecutor: Memory pool initialized" << endl;
            }
        }
        catch (const std::exception& e)
        {
            WarningInFunction
                << "Failed to initialize memory pool: " << e.what()
                << ", continuing without pooling" << endl;
            memoryPool_.reset();
        }
    }

    if (debug_ > 0)
    {
        printInfo();
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::OGL::OGLExecutor::~OGLExecutor()
{
    // Ensure GPU operations complete before destruction
    try
    {
        synchronize();
    }
    catch (const std::exception& e)
    {
        // Log but don't throw from destructor
        Warning << "OGLExecutor: Error during shutdown: " << e.what() << endl;
    }

    // Report memory pool statistics before cleanup
    if (memoryPool_ && debug_ > 0)
    {
        memoryPool_->report();
    }

    // Release memory pool before executors (depends on executor)
    memoryPool_.reset();

    // Release executors in correct order
    gpuExecutor_.reset();
    cpuExecutor_.reset();

    instanceCreated_ = false;
}


// * * * * * * * * * * * * * Static Member Functions * * * * * * * * * * * * //

void Foam::OGL::OGLExecutor::configure(const dictionary& dict)
{
    std::lock_guard<std::mutex> lock(configMutex_);

    if (instanceCreated_)
    {
        WarningInFunction
            << "OGLExecutor already initialized, configure() has no effect"
            << endl;
        return;
    }

    configDict_ = dict;
    configProvided_ = true;
}


Foam::OGL::OGLExecutor& Foam::OGL::OGLExecutor::instance()
{
    // Meyer's singleton pattern - thread-safe in C++11
    // Static local variable initialization is guaranteed thread-safe

    // Get configuration (thread-safe)
    dictionary dict;
    {
        std::lock_guard<std::mutex> lock(configMutex_);
        if (configProvided_)
        {
            dict = configDict_;
        }
    }

    // Thread-safe lazy initialization
    static OGLExecutor instance_(dict);
    instanceCreated_ = true;

    return instance_;
}


bool Foam::OGL::OGLExecutor::initialized()
{
    return instanceCreated_;
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::OGL::OGLExecutor::synchronize() const
{
    if (gpuAvailable_ && gpuExecutor_)
    {
        try
        {
            gpuExecutor_->synchronize();
        }
        catch (const std::exception& e)
        {
            FatalErrorInFunction
                << "GPU synchronization failed: " << e.what()
                << abort(FatalError);
        }
    }
}


void Foam::OGL::OGLExecutor::printInfo() const
{
    Info<< "OGLExecutor Information:" << nl
        << "  Backend:       " << backendType_ << nl
        << "  GPU Available: " << (gpuAvailable_ ? "yes" : "no") << nl
        << "  Device Index:  " << deviceIndex_ << nl
        << "  MPI Rank:      " << Pstream::myProcNo() << nl
        << endl;
}


void Foam::OGL::OGLExecutor::checkGinkgoError
(
    const std::string& operation,
    const std::exception& e
)
{
    FatalErrorInFunction
        << "Ginkgo operation '" << operation << "' failed: " << e.what()
        << abort(FatalError);
}


// ************************************************************************* //
