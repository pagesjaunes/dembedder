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

#include "MetricGraph.h"

MetricGraph<std::string> MetricGraphCsvReader(std::istream& ifs, const char& delimiter) {

    Utils::OrderedSet<std::string> objects_set;
    std::string data;

    UndirectedGraph g;
    while (ifs.peek() != EOF) {

        
        Utils::readTo(ifs, delimiter, data);
        int elA = objects_set.insert(data);

        Utils::readTo(ifs, delimiter, data);
        int elB = objects_set.insert(data);

        Utils::readTo(ifs, '\n', data);
        boost::add_edge(elA, elB, std::stof(data), g);
    }

    return MetricGraph<std::string>(g, objects_set);
}

template<class T>
MetricGraph<T>::MetricGraph(UndirectedGraph& ug, Utils::OrderedSet<T>& nodes_set) {

    this->ugraph = ug;
    this->nodes = nodes_set;

}

template<class T>
std::function<real (const T&, const T&)> MetricGraph<T>::metric_distance_function() {
    Utils::OrderedSet<T> nds = this->nodes;
    UndirectedGraph g = this->ugraph;
    return [nds, g](const T& a, const T& b) {
         return *((real *) (boost::edge(nds.getIndex(a), nds.getIndex(b), g).first.m_eproperty));
    };
}

template<class T>
std::vector<T> MetricGraph<T>::elements() const {
    return this->nodes.toVec();
}

template<class T>
real MetricGraph<T>::eccentricity(T o) {

    auto es = boost::out_edges(this->nodes.getIndex(o), this->ugraph);
    real maxweight = -std::numeric_limits<real>::infinity();
    for (auto eit = es.first; eit != es.second; ++eit) {
        int source = boost::source(*eit, this->ugraph);
        int target = boost::target(*eit, this->ugraph);
        real weight = *((real *) boost::edge(source, target, this->ugraph).first.m_eproperty);
        if (maxweight < weight) {
            maxweight = weight;
        }
    }
    return maxweight;
}

template<class T>
real MetricGraph<T>::radius() {
    real minweight = std::numeric_limits<real>::infinity();
    for (int i = 0; i < this->nodes.size(); i++){
        real w = this->eccentricity(this->nodes[i]);
        if (minweight > w) {
            minweight = w;
        }
    }
    return minweight;
}

template<class T>
real MetricGraph<T>::metric_distance(const T& a, const T& b) const {
    if (a == b) {
        return 0;
    }
    return *((real *) (boost::edge(this->nodes.getIndex(a), this->nodes.getIndex(b), this->ugraph).first.m_eproperty));
}

template class MetricGraph<std::string>;