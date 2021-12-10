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

#ifndef DEMBEDDER_METRICGRAPH_H
#define DEMBEDDER_METRICGRAPH_H

#include "real.h"
#include "Utils.h"
#include <boost/serialization/vector.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/archive/text_iarchive.hpp>
#include <boost/graph/adjacency_list.hpp>
#include <iostream>
#include <string>
#include <set>
#include <fstream>
#include <cassert>
#include <map>
#include <string>
#include <functional>
#include <limits>

typedef boost::adjacency_list<
        boost::vecS, boost::vecS, boost::undirectedS,
        boost::property<boost::vertex_index_t, int>,
        boost::property<boost::edge_weight_t, real>,
        boost::no_property> UndirectedGraph;


template <class T> class MetricGraph {

private:
    UndirectedGraph ugraph;
    Utils::OrderedSet<T> nodes;

public:

    MetricGraph(UndirectedGraph&, Utils::OrderedSet<T>&);
    ~MetricGraph() {};

    std::function<real (const T&, const T&)> metric_distance_function();
    real metric_distance(const T&, const T&) const;
    std::vector<T> elements() const;
    real eccentricity(T);
    real radius();

};

MetricGraph<std::string> MetricGraphCsvReader(std::istream&, const char&);

#include "Utils.h"
#endif //DEMBEDDER_METRICGRAPH_H
