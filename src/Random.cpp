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
#include "Random.h"

Random::Random() {
    this->mt = new std::mt19937(rd());
}

Random::~Random() {
    delete this->mt;
}

real Random::uniform(real inf, real sup) const {
    std::uniform_real_distribution<> dis(inf, sup);
    return dis(*this->mt);
}

real Random::exponential(real lambda) const {
    std::exponential_distribution<double> expdis(lambda);
    return expdis(*this->mt);
}

std::mt19937 Random::generator() const {
    return *this->mt;
}

int Random::randint(int min, int max) const {
    assert(min < max);
    std::uniform_int_distribution<int> dis(min,max);
    return dis(*this->mt);
}
