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

Description
    Compatibility header for including Ginkgo headers alongside OpenFOAM.

    OpenFOAM's error.H defines a macro "NotImplemented" that collides with
    Ginkgo's gko::NotImplemented class. This header temporarily undefines
    the conflicting macros before including Ginkgo and restores them after.

\*---------------------------------------------------------------------------*/

#ifndef GinkgoCompat_H
#define GinkgoCompat_H

// Save and undefine OpenFOAM macros that collide with Ginkgo identifiers
#ifdef NotImplemented
    #define OGL_SAVE_NotImplemented
    #undef NotImplemented
#endif

#include <ginkgo/ginkgo.hpp>

// Restore OpenFOAM macros
#ifdef OGL_SAVE_NotImplemented
    #define NotImplemented notImplemented(FUNCTION_NAME)
    #undef OGL_SAVE_NotImplemented
#endif

#endif

// ************************************************************************* //
