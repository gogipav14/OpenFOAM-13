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
    Singleton manager for Ginkgo GPU executors. Handles device selection
    and provides access to CPU and GPU executors for linear algebra operations.

    Supports multi-GPU via MPI rank-to-device mapping using environment
    variables (OMPI_COMM_WORLD_LOCAL_RANK, MPI_LOCALRANKID, etc.)

SourceFiles
    OGLExecutor.C

\*---------------------------------------------------------------------------*/

#ifndef OGLExecutor_H
#define OGLExecutor_H

#include "dictionary.H"
#include "label.H"

// Forward declarations for Ginkgo types
namespace gko
{
    class Executor;
}

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

        //- Backend type (cuda, hip, reference)
        word backendType_;

        //- Whether GPU is available and initialized
        bool gpuAvailable_;

        //- Debug level
        label debug_;


    // Private Member Functions

        //- Detect local MPI rank for multi-GPU binding
        label detectLocalRank() const;

        //- Initialize executors
        void initializeExecutors();

        //- Disallow default bitwise copy construction
        OGLExecutor(const OGLExecutor&) = delete;

        //- Disallow default bitwise assignment
        void operator=(const OGLExecutor&) = delete;


    // Static Data

        //- Singleton instance
        static OGLExecutor* instancePtr_;


public:

    // Constructors

        //- Construct from dictionary
        explicit OGLExecutor(const dictionary& dict);


    //- Destructor
    ~OGLExecutor();


    // Static Member Functions

        //- Initialize the singleton with configuration
        static void initialize(const dictionary& dict);

        //- Access the singleton instance
        static OGLExecutor& instance();

        //- Check if singleton is initialized
        static bool initialized();

        //- Destroy the singleton
        static void destroy();


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


        // Utility

            //- Synchronize GPU operations
            void synchronize() const;

            //- Print device information
            void printInfo() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
