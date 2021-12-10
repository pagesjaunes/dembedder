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

#ifndef FMS_CPP
#define FMS_CPP

#include "FiniteMetricSpace.h"

template<class T>
FiniteMetricSpace<T>::FiniteMetricSpace(std::vector<T>& objs, std::function<real (const T&, const T&)>& dist_func) {
    this->objects = Utils::OrderedSet<T>(objs);
    this->distance_func = dist_func;
}

template<class T>
int FiniteMetricSpace<T>::getSize() const {
    return this->objects.size();
}

template<class T>
real FiniteMetricSpace<T>::distance(const T& a, const T& b) const {
    if (a == b) {
        return 0;
    }
    return std::fmin(MAXDIST, this->distance_func(a, b));
}

template<class T>
T FiniteMetricSpace<T>::nearestNeighbor(const T& x) const {
    return this->nearestNeighbor(x, this->objects);
}

template<class T>
T FiniteMetricSpace<T>::nearestNeighbor(const T& x, const Utils::OrderedSet<T>& search_item) const {
    std::function<bool (const T&, const T&)> alwaysTrue = [](const T&, const T&) {return true;};
    return this->nearestNeighbor(x, search_item, alwaysTrue);
}

template<class T>
Utils::OrderedSet<T> FiniteMetricSpace<T>::getObjects() const {
    return this->objects;
}

template<class T>
real FiniteMetricSpace<T>::getRadiusOn(const Utils::OrderedSet<T>& voc) const {

    real rad = std::numeric_limits<real>::infinity();

    #pragma omp parallel for default(none) shared(voc) reduction(min:rad)
    for (int i = 0; i < voc.size(); i++) {
        real ecc = this->getMaxEccentricityOn(voc[i], voc);
        if (rad > ecc) {
            rad = ecc;
        }
    }

    return rad;
}

template<class T>
real FiniteMetricSpace<T>::getMaxEccentricityOn(const T& x, const Utils::OrderedSet<T>& voc) const {

    real maxdist = -std::numeric_limits<real>::infinity();

    #pragma omp parallel for default(none) shared(voc, x) reduction(max:maxdist)
    for (int i = 0; i < voc.size(); i++) {
        real d = this->distance(x, voc[i]);
        if (d > maxdist) {
            maxdist = d;
        }
    }

    return maxdist;
}

template<class T>
T FiniteMetricSpace<T>::centroid(const Utils::OrderedSet<T>& search_item) const {

    NearestObject no;
    #pragma omp parallel for default(none) shared(search_item) reduction(nearest_min:no)
    for (int i = 0; i < search_item.size(); i++) {
        real cum_dist = 0;
        #pragma omp parallel for default(none) shared(search_item, i) reduction(+:cum_dist)
        for (int j = 0; j < search_item.size(); j++) {
            if (i != j) {
                cum_dist += this->distance(search_item[i], search_item[j]);
            }
        }
        NearestObject candidate(i, cum_dist);
        if (candidate.d < no.d) {
            no.d = candidate.d;
            no.idx = candidate.idx;
        }
    }

    return search_item[no.idx];
}

template<class T>
T FiniteMetricSpace<T>::nearestNeighbor(const T& x, const Utils::OrderedSet<T>& search_item,
                                        const std::function<bool (const T &, const T &)>& condition)  const{
    const std::vector<T>& vec_search_item = search_item.vecref();
    return search_item[this->nearestNeighbor(x, vec_search_item, condition)];
}

template<class T>
real FiniteMetricSpace<T>::getMaxEccentricityOn(const T& x) const {
    return this->getMaxEccentricityOn(x, this->objects);
}

template<class T>
std::function<real(const T &, const T &)> FiniteMetricSpace<T>::getDistanceFunction() const {
    return this->distance_func;
}

template<class T>
T FiniteMetricSpace<T>::at(const int idx) const {
    return this->objects[idx];
}

template<class T>
real FiniteMetricSpace<T>::distance_idx(const int i1, const int i2) const {
    return this->distance(this->at(i1), this->at(i2));
}

template<class T>
FiniteMetricSpace<T>::FiniteMetricSpace(Utils::OrderedSet<T>& objs, std::function<real(const T &, const T &)>& func) {
    this->objects = objs;
    this->distance_func = func;
}

template<class T>
int FiniteMetricSpace<T>::nearestNeighbor(const T& x, const std::vector<T>& search_item,
                                          const std::function<bool(const T &, const T &)>& condition) const {

    assert(search_item.size() > 0);
    NearestObject no(-1, std::numeric_limits<real>::infinity());

    #pragma omp parallel for default(none) shared(search_item, x, condition) reduction(nearest_min:no)
    for (int i = 0; i < search_item.size(); i++) {
        if (condition(x, search_item[i])) {
            NearestObject candidate(i, this->distance(search_item[i], x));
            if (candidate.d < no.d) {
                no.d = candidate.d;
                no.idx = candidate.idx;
            }
        }
    }
    return no.idx;
}


template class FiniteMetricSpace<std::string>;
#endif
