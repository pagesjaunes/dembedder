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

#ifndef DEMBEDDER_VECBINDER_H
#define DEMBEDDER_VECBINDER_H

#include "Utils.h"
#include "real.h"
#include <iostream>
#include <string>
#include <sstream>

class VecBinder {

private:
    int dim;
    int nvoc;
    std::vector<real> vectors;
    Utils::OrderedSet<std::string> items;

public:

    enum FILE_FORMAT {
        vec};
    VecBinder(std::istream&, FILE_FORMAT format);

    const real* at(int) const;
    const real* at(const std::string&) const;
    int getDim() const;
    int getVocSize() const;
    int size() const;
    std::vector<std::string> getElements() const;

};


#endif //DEMBEDDER_VECBINDER_H
