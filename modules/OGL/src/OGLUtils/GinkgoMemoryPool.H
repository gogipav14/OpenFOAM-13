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
    Foam::OGL::GinkgoMemoryPool

Description
    Memory pool for Ginkgo GPU allocations to reduce allocation overhead.

    Benefits:
    - Reuses GPU memory allocations between solver iterations
    - Reduces cudaMalloc/cudaFree overhead
    - Provides workspace vectors for iterative solvers
    - Thread-safe for parallel access

    The pool maintains:
    - Pre-allocated workspace vectors of common sizes
    - LRU-style eviction for unused allocations
    - Statistics for monitoring memory usage

    Configuration:
    \verbatim
    OGLCoeffs
    {
        memoryPool
        {
            enabled         true;
            maxPoolSize     1073741824;  // 1 GB
            preAllocate     true;
        }
    }
    \endverbatim

SourceFiles
    GinkgoMemoryPool.C

\*---------------------------------------------------------------------------*/

#ifndef GinkgoMemoryPool_H
#define GinkgoMemoryPool_H

#include "dictionary.H"

#include <ginkgo/ginkgo.hpp>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>
#include <list>

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace OGL
{

/*---------------------------------------------------------------------------*\
                     Class GinkgoMemoryPool Declaration
\*---------------------------------------------------------------------------*/

class GinkgoMemoryPool
{
public:

    // Public Types

        //- Pool entry with allocation info
        struct PoolEntry
        {
            std::shared_ptr<gko::array<char>> buffer;
            std::size_t size;
            bool inUse;
        };


private:

    // Private Data

        //- Ginkgo executor
        std::shared_ptr<const gko::Executor> exec_;

        //- Pool entries by size bucket
        std::unordered_map<std::size_t, std::list<PoolEntry>> pool_;

        //- Mutex for thread-safe access
        mutable std::mutex mutex_;

        //- Whether pool is enabled
        bool enabled_;

        //- Maximum total pool size in bytes
        std::size_t maxPoolSize_;

        //- Current total allocated size
        std::size_t currentSize_;

        //- Statistics: total allocations
        mutable std::size_t totalAllocations_;

        //- Statistics: pool hits
        mutable std::size_t poolHits_;

        //- Statistics: pool misses
        mutable std::size_t poolMisses_;


    // Private Member Functions

        //- Round size up to nearest bucket size
        static std::size_t bucketSize(std::size_t size);

        //- Evict least recently used entries to free space
        void evict(std::size_t targetFree);

        //- Disallow copy
        GinkgoMemoryPool(const GinkgoMemoryPool&) = delete;
        void operator=(const GinkgoMemoryPool&) = delete;


public:

    // Constructors

        //- Construct with executor and configuration
        GinkgoMemoryPool
        (
            std::shared_ptr<const gko::Executor> exec,
            const dictionary& dict = dictionary()
        );


    //- Destructor
    ~GinkgoMemoryPool();


    // Member Functions

        // Allocation

            //- Allocate a buffer of at least the given size
            std::shared_ptr<gko::array<char>> allocate(std::size_t size);

            //- Return a buffer to the pool
            void deallocate(std::shared_ptr<gko::array<char>> buffer);

            //- Create a dense vector using pool memory
            template<typename ValueType>
            std::shared_ptr<gko::matrix::Dense<ValueType>>
            createVector(std::size_t rows, std::size_t cols = 1);


        // Control

            //- Clear all cached allocations
            void clear();

            //- Enable/disable pooling
            void setEnabled(bool enabled) { enabled_ = enabled; }


        // Statistics

            //- Get current pool size in bytes
            std::size_t currentPoolSize() const { return currentSize_; }

            //- Get pool hit rate
            double hitRate() const;

            //- Report statistics
            void report() const;
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace OGL
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
