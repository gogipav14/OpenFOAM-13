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

#include "FP32CastWrapper.H"
#include "OGLExecutor.H"

// * * * * * * * * * * * * * Static Member Functions * * * * * * * * * * * * //

std::shared_ptr<Foam::OGL::FP32CastWrapper::VectorF32>
Foam::OGL::FP32CastWrapper::toGinkgoF32
(
    std::shared_ptr<const gko::Executor> exec,
    const scalarField& field
)
{
    const label n = field.size();

    // Create temporary on CPU
    auto cpuExec = OGLExecutor::instance().cpuExecutor();
    auto hostVec = VectorF32::create(cpuExec, gko::dim<2>(n, 1));
    float* data = hostVec->get_values();

    // Copy and cast data
    forAll(field, i)
    {
        data[i] = static_cast<float>(field[i]);
    }

    // Copy to target executor (GPU)
    auto result = VectorF32::create(exec, gko::dim<2>(n, 1));
    result->copy_from(hostVec.get());

    return result;
}


std::shared_ptr<Foam::OGL::FP32CastWrapper::VectorF64>
Foam::OGL::FP32CastWrapper::toGinkgoF64
(
    std::shared_ptr<const gko::Executor> exec,
    const scalarField& field
)
{
    const label n = field.size();

    // Create temporary on CPU
    auto cpuExec = OGLExecutor::instance().cpuExecutor();
    auto hostVec = VectorF64::create(cpuExec, gko::dim<2>(n, 1));
    double* data = hostVec->get_values();

    // Copy data (OpenFOAM scalar is already double)
    forAll(field, i)
    {
        data[i] = field[i];
    }

    // Copy to target executor (GPU)
    auto result = VectorF64::create(exec, gko::dim<2>(n, 1));
    result->copy_from(hostVec.get());

    return result;
}


void Foam::OGL::FP32CastWrapper::fromGinkgoF32
(
    const VectorF32* gkoVec,
    scalarField& field
)
{
    const label n = field.size();

    // Copy to CPU
    auto cpuExec = OGLExecutor::instance().cpuExecutor();
    auto hostVec = VectorF32::create(cpuExec, gko::dim<2>(n, 1));
    hostVec->copy_from(gkoVec);

    // Cast and copy to OpenFOAM field
    const float* data = hostVec->get_const_values();
    forAll(field, i)
    {
        field[i] = static_cast<scalar>(data[i]);
    }
}


void Foam::OGL::FP32CastWrapper::fromGinkgoF64
(
    const VectorF64* gkoVec,
    scalarField& field
)
{
    const label n = field.size();

    // Copy to CPU
    auto cpuExec = OGLExecutor::instance().cpuExecutor();
    auto hostVec = VectorF64::create(cpuExec, gko::dim<2>(n, 1));
    hostVec->copy_from(gkoVec);

    // Copy to OpenFOAM field
    const double* data = hostVec->get_const_values();
    forAll(field, i)
    {
        field[i] = data[i];
    }
}


void Foam::OGL::FP32CastWrapper::addCorrectionF32ToF64
(
    const VectorF32* dxF32,
    scalarField& xF64
)
{
    const label n = xF64.size();

    // Copy correction to CPU
    auto cpuExec = OGLExecutor::instance().cpuExecutor();
    auto hostDx = VectorF32::create(cpuExec, gko::dim<2>(n, 1));
    hostDx->copy_from(dxF32);

    // Add to FP64 field
    const float* dxData = hostDx->get_const_values();
    forAll(xF64, i)
    {
        xF64[i] += static_cast<scalar>(dxData[i]);
    }
}


std::shared_ptr<Foam::OGL::FP32CastWrapper::VectorF32>
Foam::OGL::FP32CastWrapper::castF64ToF32
(
    std::shared_ptr<const gko::Executor> exec,
    const VectorF64* vecF64
)
{
    const auto n = vecF64->get_size()[0];

    // Copy to CPU for casting
    auto cpuExec = OGLExecutor::instance().cpuExecutor();
    auto hostF64 = VectorF64::create(cpuExec, gko::dim<2>(n, 1));
    hostF64->copy_from(vecF64);

    // Create F32 vector and cast
    auto hostF32 = VectorF32::create(cpuExec, gko::dim<2>(n, 1));
    const double* srcData = hostF64->get_const_values();
    float* dstData = hostF32->get_values();

    for (gko::size_type i = 0; i < n; i++)
    {
        dstData[i] = static_cast<float>(srcData[i]);
    }

    // Copy to target executor
    auto result = VectorF32::create(exec, gko::dim<2>(n, 1));
    result->copy_from(hostF32.get());

    return result;
}


std::shared_ptr<Foam::OGL::FP32CastWrapper::VectorF64>
Foam::OGL::FP32CastWrapper::castF32ToF64
(
    std::shared_ptr<const gko::Executor> exec,
    const VectorF32* vecF32
)
{
    const auto n = vecF32->get_size()[0];

    // Copy to CPU for casting
    auto cpuExec = OGLExecutor::instance().cpuExecutor();
    auto hostF32 = VectorF32::create(cpuExec, gko::dim<2>(n, 1));
    hostF32->copy_from(vecF32);

    // Create F64 vector and cast
    auto hostF64 = VectorF64::create(cpuExec, gko::dim<2>(n, 1));
    const float* srcData = hostF32->get_const_values();
    double* dstData = hostF64->get_values();

    for (gko::size_type i = 0; i < n; i++)
    {
        dstData[i] = static_cast<double>(srcData[i]);
    }

    // Copy to target executor
    auto result = VectorF64::create(exec, gko::dim<2>(n, 1));
    result->copy_from(hostF64.get());

    return result;
}


// ************************************************************************* //
