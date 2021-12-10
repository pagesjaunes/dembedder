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

#ifndef DEMBEDDER_TREE_H
#define DEMBEDDER_TREE_H

#include "Utils.h"
#include <string>
#include <iostream>

template <class T> class Tree {

private:
    T centroid;
    Utils::OrderedSet<T> nodes;
    std::vector<std::pair<T, T>> edges;

public:

    Tree();
    Tree(std::istream&);
    ~Tree() {};

    void add_node(T);
    void add_edge(std::pair<T, T>);
    Utils::OrderedSet<T>& getNodes();
    Utils::OrderedSet<T> getNodesConst() const;
    std::vector<std::pair<T,T>>& edges_vec();
    std::vector<std::pair<T, T>> edges_const() const;
    void store(std::ostream&);
    void set_centroid(T);
    Tree<T>& operator=(const Tree<T>&);
    void clear();

};




#endif //DEMBEDDER_TREE_H
