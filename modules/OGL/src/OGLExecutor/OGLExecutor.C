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
#include "messageStream.H"
#include "Pstream.H"

#include <ginkgo/ginkgo.hpp>
#include <cstdlib>

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

Foam::OGL::OGLExecutor* Foam::OGL::OGLExecutor::instancePtr_ = nullptr;


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

    // Fallback: use global rank mod expected GPUs (not ideal but functional)
    if (debug_ > 0)
    {
        Info<< "OGLExecutor: No local rank environment variable found, "
            << "using global rank for device selection" << endl;
    }

    return Pstream::myProcNo();
}


void Foam::OGL::OGLExecutor::initializeExecutors()
{
    // Always create CPU reference executor
    cpuExecutor_ = gko::ReferenceExecutor::create();
    backendType_ = "reference";
    gpuAvailable_ = false;

    // Determine device index
    if (deviceIndex_ < 0)
    {
        // Auto-select based on local MPI rank
        deviceIndex_ = detectLocalRank();
    }

    // Try to create GPU executor
    #ifdef GKO_COMPILING_CUDA
    try
    {
        // Check if CUDA device is available
        int numDevices = 0;
        auto status = cudaGetDeviceCount(&numDevices);

        if (status == cudaSuccess && numDevices > 0)
        {
            label deviceId = deviceIndex_ % numDevices;
            gpuExecutor_ = gko::CudaExecutor::create(
                deviceId,
                cpuExecutor_,
                std::make_shared<gko::CudaAllocator>()
            );
            backendType_ = "cuda";
            gpuAvailable_ = true;
            deviceIndex_ = deviceId;

            if (debug_ > 0)
            {
                Info<< "OGLExecutor: CUDA device " << deviceId
                    << " of " << numDevices << " initialized" << endl;
            }
        }
    }
    catch (const std::exception& e)
    {
        if (debug_ > 0)
        {
            Warning
                << "OGLExecutor: CUDA initialization failed: " << e.what()
                << ", falling back to CPU" << endl;
        }
    }
    #endif

    #ifdef GKO_COMPILING_HIP
    if (!gpuAvailable_)
    {
        try
        {
            int numDevices = 0;
            auto status = hipGetDeviceCount(&numDevices);

            if (status == hipSuccess && numDevices > 0)
            {
                label deviceId = deviceIndex_ % numDevices;
                gpuExecutor_ = gko::HipExecutor::create(
                    deviceId,
                    cpuExecutor_
                );
                backendType_ = "hip";
                gpuAvailable_ = true;
                deviceIndex_ = deviceId;

                if (debug_ > 0)
                {
                    Info<< "OGLExecutor: HIP device " << deviceId
                        << " of " << numDevices << " initialized" << endl;
                }
            }
        }
        catch (const std::exception& e)
        {
            if (debug_ > 0)
            {
                Warning
                    << "OGLExecutor: HIP initialization failed: " << e.what()
                    << ", falling back to CPU" << endl;
            }
        }
    }
    #endif

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
    debug_(dict.lookupOrDefault<label>("debug", 0))
{
    initializeExecutors();

    if (debug_ > 0)
    {
        printInfo();
    }
}


// * * * * * * * * * * * * * * * * Destructor  * * * * * * * * * * * * * * * //

Foam::OGL::OGLExecutor::~OGLExecutor()
{
    // Ensure GPU operations complete before destruction
    synchronize();

    gpuExecutor_.reset();
    cpuExecutor_.reset();
}


// * * * * * * * * * * * * * Static Member Functions * * * * * * * * * * * * //

void Foam::OGL::OGLExecutor::initialize(const dictionary& dict)
{
    if (instancePtr_)
    {
        FatalErrorInFunction
            << "OGLExecutor already initialized"
            << abort(FatalError);
    }

    instancePtr_ = new OGLExecutor(dict);
}


Foam::OGL::OGLExecutor& Foam::OGL::OGLExecutor::instance()
{
    if (!instancePtr_)
    {
        // Create with default settings if not initialized
        dictionary defaultDict;
        instancePtr_ = new OGLExecutor(defaultDict);
    }

    return *instancePtr_;
}


bool Foam::OGL::OGLExecutor::initialized()
{
    return instancePtr_ != nullptr;
}


void Foam::OGL::OGLExecutor::destroy()
{
    if (instancePtr_)
    {
        delete instancePtr_;
        instancePtr_ = nullptr;
    }
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

void Foam::OGL::OGLExecutor::synchronize() const
{
    if (gpuAvailable_ && gpuExecutor_)
    {
        gpuExecutor_->synchronize();
    }
}


void Foam::OGL::OGLExecutor::printInfo() const
{
    Info<< "OGLExecutor Information:" << nl
        << "  Backend:      " << backendType_ << nl
        << "  GPU Available:" << (gpuAvailable_ ? "yes" : "no") << nl
        << "  Device Index: " << deviceIndex_ << nl
        << "  MPI Rank:     " << Pstream::myProcNo() << nl
        << endl;
}


// ************************************************************************* //
