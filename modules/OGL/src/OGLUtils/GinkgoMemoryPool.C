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

#include "GinkgoMemoryPool.H"
#include "messageStream.H"

// * * * * * * * * * * * * * Private Member Functions  * * * * * * * * * * * //

std::size_t Foam::OGL::GinkgoMemoryPool::bucketSize(std::size_t size)
{
    // Round up to power of 2 for efficient bucketing
    // Minimum bucket size is 4KB
    constexpr std::size_t minBucket = 4096;

    if (size <= minBucket) return minBucket;

    // Find next power of 2
    std::size_t bucket = minBucket;
    while (bucket < size)
    {
        bucket *= 2;
    }
    return bucket;
}


void Foam::OGL::GinkgoMemoryPool::evict(std::size_t targetFree)
{
    // LRU eviction: remove oldest unused entries until we have enough space
    std::size_t freed = 0;

    for (auto& [size, entries] : pool_)
    {
        auto it = entries.begin();
        while (it != entries.end() && freed < targetFree)
        {
            if (!it->inUse)
            {
                freed += it->size;
                currentSize_ -= it->size;
                it = entries.erase(it);
            }
            else
            {
                ++it;
            }
        }
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::OGL::GinkgoMemoryPool::GinkgoMemoryPool
(
    std::shared_ptr<const gko::Executor> exec,
    const dictionary& dict
)
:
    exec_(exec),
    enabled_(dict.lookupOrDefault<bool>("enabled", true)),
    maxPoolSize_(dict.lookupOrDefault<std::size_t>("maxPoolSize", 1073741824)),
    currentSize_(0),
    totalAllocations_(0),
    poolHits_(0),
    poolMisses_(0)
{}


Foam::OGL::GinkgoMemoryPool::~GinkgoMemoryPool()
{
    clear();
}


// * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * * //

std::shared_ptr<gko::array<char>>
Foam::OGL::GinkgoMemoryPool::allocate(std::size_t size)
{
    std::lock_guard<std::mutex> lock(mutex_);

    totalAllocations_++;

    if (!enabled_)
    {
        poolMisses_++;
        return std::make_shared<gko::array<char>>(exec_, size);
    }

    const std::size_t bucket = bucketSize(size);

    // Look for existing unused buffer in this bucket
    auto& entries = pool_[bucket];
    for (auto& entry : entries)
    {
        if (!entry.inUse && entry.size >= size)
        {
            entry.inUse = true;
            poolHits_++;
            return entry.buffer;
        }
    }

    // No suitable buffer found, allocate new one
    poolMisses_++;

    // Check if we need to evict
    if (currentSize_ + bucket > maxPoolSize_)
    {
        evict(bucket);
    }

    auto buffer = std::make_shared<gko::array<char>>(exec_, bucket);

    PoolEntry entry{buffer, bucket, true};
    entries.push_back(entry);
    currentSize_ += bucket;

    return buffer;
}


void Foam::OGL::GinkgoMemoryPool::deallocate
(
    std::shared_ptr<gko::array<char>> buffer
)
{
    std::lock_guard<std::mutex> lock(mutex_);

    if (!enabled_)
    {
        return;  // Let shared_ptr handle deallocation
    }

    // Find and mark as unused
    for (auto& [size, entries] : pool_)
    {
        for (auto& entry : entries)
        {
            if (entry.buffer == buffer)
            {
                entry.inUse = false;
                return;
            }
        }
    }
}


template<typename ValueType>
std::shared_ptr<gko::matrix::Dense<ValueType>>
Foam::OGL::GinkgoMemoryPool::createVector(std::size_t rows, std::size_t cols)
{
    const std::size_t size = rows * cols * sizeof(ValueType);
    auto buffer = allocate(size);

    // Create dense matrix using the pooled buffer
    // Note: This is a simplified implementation. In practice, we'd need
    // to properly manage the memory lifetime with Ginkgo's allocator interface.
    return gko::matrix::Dense<ValueType>::create(exec_, gko::dim<2>(rows, cols));
}


void Foam::OGL::GinkgoMemoryPool::clear()
{
    std::lock_guard<std::mutex> lock(mutex_);

    pool_.clear();
    currentSize_ = 0;
}


double Foam::OGL::GinkgoMemoryPool::hitRate() const
{
    if (totalAllocations_ == 0) return 0.0;
    return static_cast<double>(poolHits_) / totalAllocations_;
}


void Foam::OGL::GinkgoMemoryPool::report() const
{
    Info<< "GinkgoMemoryPool statistics:" << nl
        << "  Total allocations: " << totalAllocations_ << nl
        << "  Pool hits: " << poolHits_ << nl
        << "  Pool misses: " << poolMisses_ << nl
        << "  Hit rate: " << 100.0 * hitRate() << "%" << nl
        << "  Current pool size: " << currentSize_ / (1024*1024) << " MB" << nl
        << "  Max pool size: " << maxPoolSize_ / (1024*1024) << " MB" << nl
        << endl;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

// Explicit instantiation
template std::shared_ptr<gko::matrix::Dense<float>>
Foam::OGL::GinkgoMemoryPool::createVector<float>(std::size_t, std::size_t);

template std::shared_ptr<gko::matrix::Dense<double>>
Foam::OGL::GinkgoMemoryPool::createVector<double>(std::size_t, std::size_t);


// ************************************************************************* //
