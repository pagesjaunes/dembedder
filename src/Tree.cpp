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

#ifndef TREE_CPP
#define TREE_CPP
#include "Tree.h"

template<class T>
void Tree<T>::add_node(T node) {
    nodes.insert(node);
}

template<class T>
void Tree<T>::add_edge(std::pair<T, T> edge) {
    this->add_node(edge.first);
    this->add_node(edge.second);
    edges.push_back(edge);
}

template<class T>
std::vector<std::pair<T,T>>& Tree<T>::edges_vec() {
    return edges;
}

template<class T>
std::vector<std::pair<T,T>> Tree<T>::edges_const() const {
    return edges;
}

template<class T>
Tree<T>::Tree() {
    this->centroid = "";
}

template<class T>
void Tree<T>::store(std::ostream& ostream) {

    ostream << "centroid" << "\t" << centroid << "\n";
    for (int i = 0; i < this->edges.size(); i++) {
        ostream << this->edges[i].first << "\t" << this->edges[i].second << "\n";
    }
    ostream.flush();

}

template<class T>
void Tree<T>::set_centroid(T x) {
    this->centroid = x;
    this->add_node(this->centroid);
}

template<class T>
Utils::OrderedSet<T>& Tree<T>::getNodes() {
    return this->nodes;
}

template<class T>
Tree<T>& Tree<T>::operator=(const Tree<T>& other) {

    this->nodes.clear();
    this->edges.clear();

    for (int i = 0; i < other.nodes.size(); i++) {
        this->nodes.insert(other.nodes[i]);
    }

    for (int i = 0; i < other.edges.size(); i++) {
        this->edges.push_back(other.edges[i]);
    }

    this->centroid = other.centroid;
    return *this;

}

template<class T>
Tree<T>::Tree(std::istream& ifs) {

    std::string data;

    Utils::readTo(ifs, '\t', data);
    assert(data == "centroid");

    Utils::readTo(ifs, '\n', data);
    this->centroid = data;

    while (ifs.peek() != EOF) {
        std::string u, v;
        Utils::readTo(ifs, '\t', u);
        Utils::readTo(ifs, '\n', v);
        this->add_edge({u, v});
    }

}

template<class T>
void Tree<T>::clear() {

    this->nodes.clear();
    this->edges.clear();

}

template<class T>
Utils::OrderedSet<T> Tree<T>::getNodesConst() const {
    return this->nodes;
}


template class Tree<std::string>;
#endif