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

#ifndef DEMBEDDER_RANDOM_H
#define DEMBEDDER_RANDOM_H

#include <random>
#include <cassert>
#include "real.h"


class Random {

private:
    std::random_device rd;
    std::mt19937* mt;

public:
    Random();
    ~Random();

    real uniform(real inf, real sup) const;
    real exponential(real lambda) const;
    int randint(int, int) const;
    std::mt19937 generator() const;

};

#endif //DEMBEDDER_RANDOM_H
