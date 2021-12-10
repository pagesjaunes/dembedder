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

#ifndef DEMBEDDER_FINITEMETRICSPACE_H
#define DEMBEDDER_FINITEMETRICSPACE_H

#include "real.h"
#include "Utils.h"
#include "MetricGraph.h"
#include "Geometry.h"
#include <vector>
#include <functional>
#include <string>
#include <limits>
#include <omp.h>

#define MAXDIST 10000

struct NearestObject {
    int idx;
    real d;
    NearestObject() : idx(-1), d(std::numeric_limits<real>::infinity()) {};
    NearestObject(int i, real dist) : idx(i), d(dist)  {}
};
#pragma omp declare reduction(nearest_min : struct NearestObject : omp_out = omp_in.d < omp_out.d ? omp_in : omp_out)

template<class T> class FiniteMetricSpace {

private:
    Utils::OrderedSet<T> objects;
    std::function<real (const T&, const T&)> distance_func;

public:

    FiniteMetricSpace();
    FiniteMetricSpace(Utils::OrderedSet<T>&, std::function<real (const T&, const T&)>&);
    FiniteMetricSpace(std::vector<T>&, std::function<real (const T&, const T&)>&);
    ~FiniteMetricSpace() {};

    int getSize() const;
    real distance(const T&, const T&) const;
    real getRadiusOn(const Utils::OrderedSet<T>&) const;
    real getMaxEccentricityOn(const T&) const;
    real getMaxEccentricityOn(const T&, const Utils::OrderedSet<T>&) const;
    int nearestNeighbor(const T&, const std::vector<T>&, const std::function<bool (const T&, const T&)>&) const;
    T nearestNeighbor(const T&, const Utils::OrderedSet<T>&, const std::function<bool (const T&, const T&)>&) const;
    T nearestNeighbor(const T&, const Utils::OrderedSet<T>&) const;
    T nearestNeighbor(const T&) const;
    T centroid(const Utils::OrderedSet<T>&) const;
    Utils::OrderedSet<T> getObjects() const;
    std::function<real (const T&, const T&)> getDistanceFunction() const;
    T at(const int) const;
    real distance_idx(const int, const int) const;

};




#endif //DEMBEDDER_FINITEMETRICSPACE_H
