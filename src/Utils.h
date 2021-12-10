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

#ifndef DEMBEDDER_UTILS_H
#define DEMBEDDER_UTILS_H

#include "Random.h"
#include <string>
#include <vector>
#include <set>
#include <map>
#include <iterator>
#include <climits>
#include <iostream>

class Utils {



public:

    static void readTo(std::istream&, char, std::string&);

    template<class T> class OrderedSet {

    private:
        std::vector<T> objectVector;
        std::set<T> objectSet;
        std::vector<int> freqs;
        std::map<T, int> object2idx;

    public:

        OrderedSet();
        OrderedSet(std::vector<T>&);
        OrderedSet(const Utils::OrderedSet<T>&);
        ~OrderedSet() {};
        int insert(const T obj);
        T at(const int& idx) const;
        T operator[](const int& idx) const;
        Utils::OrderedSet<T>& operator=(const Utils::OrderedSet<T>&);
        int getIndex(const T&) const;
        int getFreq(const T&) const;
        int size() const;
        int pop(const T&);
        std::set<T> getSet();
        std::vector<T> toVec() const;
        const std::vector<T>& vecref() const;
        T randomElement(Random&);
        void clear();

    };

};
#endif //DEMBEDDER_UTILS_H
