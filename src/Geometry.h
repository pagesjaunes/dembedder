/*
 * dembedder is a software aiming at inducing taxonomies from embedding spaces.
 * Copyright (C) 2021 Solocal-SA and CNRS
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 * 
 * dembedder was developped by: François Torregrossa, Robin Allesiardo, Vincent Claveau and Guillaume Gravier.
 */

#ifndef DEMBEDDER_GEOMETRY_H
#define DEMBEDDER_GEOMETRY_H

#include "Utils.h"
#include "real.h"
#include <vector>
#define EPSILON 1e-8

namespace Geometry {

    real dot(const real*, const real*, int);
    real pDistance(const real*, const real*, real, int);
    real poincareDistance(const real*, const real*, real, int);
    real squaredLorentzianDistance(const real*, const real*, real, int);

    real hyperbolic_e(const real*, const real*, int);
    real euclidean_e(const real*, const real*, int);
    real psi(const real*, real, int);

    real hyperbolic_energy(const real*, const real*, real, int);
    real euclidean_energy(const real*, const real*, real, int);


};


#endif //DEMBEDDER_GEOMETRY_H
