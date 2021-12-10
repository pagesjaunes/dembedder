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

#ifndef DEMBEDDER_ALGORITHMS_H
#define DEMBEDDER_ALGORITHMS_H

#include "Random.h"
#include "Tree.h"
#include "FiniteMetricSpace.h"
#include "MetricGraph.h"
#include "Utils.h"
#include "Args.h"
#include <boost/graph/connected_components.hpp>
#include <boost/graph/graph_utility.hpp>
#include <boost/graph/floyd_warshall_shortest.hpp>
#include <boost/graph/exterior_property.hpp>
#include <boost/graph/prim_minimum_spanning_tree.hpp>
#include <boost/graph/dijkstra_shortest_paths.hpp>
#include <tuple>
#include <numeric>
#include <algorithm>
#include <memory>
#include <queue>

#define C_HAT 22
#define C 2

typedef boost::exterior_vertex_property<UndirectedGraph, real> DistanceProperty;
typedef DistanceProperty::matrix_type DistanceMatrix;
typedef DistanceProperty::matrix_map_type DistanceMatrixMap;
typedef boost::property_map<UndirectedGraph, boost::edge_weight_t>::type WeightMap;
typedef UndirectedGraph::vertex_descriptor Vertex;

namespace Algorithms {

    template<typename T> std::tuple<real, real, real, real> treeDistortion(const Tree<T>& gold, const Tree<T>& pred) {

        UndirectedGraph goldG;
        Utils::OrderedSet<T> goldNodes = gold.getNodesConst();
        std::vector<std::pair<T, T>> goldEdges = gold.edges_const();

        for (int i = 0; i < goldEdges.size(); i++) {
            T u = goldEdges[i].first;
            T v = goldEdges[i].second;
            boost::add_edge(goldNodes.getIndex(u), goldNodes.getIndex(v), 1.0, goldG);
        }
        WeightMap wmap_gold = boost::get(boost::edge_weight, goldG);
        DistanceMatrix distances_gold(boost::num_vertices(goldG));
        DistanceMatrixMap dm_gold(distances_gold, goldG);
        bool valid = boost::floyd_warshall_all_pairs_shortest_paths(goldG, dm_gold, boost::weight_map(wmap_gold));
        assert(valid);

        UndirectedGraph predG;
        Utils::OrderedSet<T> predNodes = pred.getNodesConst();
        std::vector<std::pair<T, T>> predEdges = pred.edges_const();

        for (int i = 0; i < predEdges.size(); i++) {
            T u = predEdges[i].first;
            T v = predEdges[i].second;
            assert(goldNodes.getFreq(u) > 0);
            assert(goldNodes.getFreq(v) > 0);
            boost::add_edge(predNodes.getIndex(u), predNodes.getIndex(v), 1.0, predG);
        }
        WeightMap wmap_pred = boost::get(boost::edge_weight, predG);
        DistanceMatrix distances_pred(boost::num_vertices(predG));
        DistanceMatrixMap dm_pred(distances_pred, predG);
        valid = boost::floyd_warshall_all_pairs_shortest_paths(predG, dm_pred, boost::weight_map(wmap_pred));
        assert(valid);

        real maxExpand = -std::numeric_limits<real>::infinity();
        real maxContrac = -std::numeric_limits<real>::infinity();
        real cumExpand = 0;
        real cumContrac = 0;
        long count = 0;

        for (int i = 0; i < predNodes.size(); i++) {
            for (int j = i + 1; j < predNodes.size(); j++) {
                //std::cout << "node 1: " << predNodes[i] << " node 2: " << predNodes[j] << std::endl;
                try {
                    real dpred = distances_pred[i][j];
                    real dgold = distances_gold[goldNodes.getIndex(predNodes[i])][goldNodes.getIndex(predNodes[j])];

                    real expand = dpred / dgold;
                    real contrac = dgold / dpred;

                    maxExpand = std::fmax(maxExpand, expand);
                    maxContrac = std::fmax(maxContrac, contrac);
                    cumExpand += expand;
                    cumContrac += contrac;
                    count++;

                } catch (std::exception e) {}

            }
        }

        return std::tuple<real, real, real, real>(maxExpand, cumExpand / static_cast<real>(count), maxContrac, cumContrac / static_cast<real>(count));


    }

    template<typename T> void getNeighbors(const FiniteMetricSpace<T>& fms, std::map<T, std::vector<std::pair<T, real>>*>& mapNeighbors) {

        for (int i = 0; i < fms.getSize(); i++) {
            mapNeighbors.insert({fms.at(i), new std::vector<std::pair<T, real>>(fms.getSize() - 1)});
        }

        #pragma omp parallel for default(none) shared(mapNeighbors, fms)
        for (int i = 0; i < fms.getSize(); i++) {
            #pragma omp parallel for default(none) shared(mapNeighbors, fms, i)
            for (int j = 0; j < fms.getSize(); j++) {
                int realIndex = j + 1 == fms.getSize() ? i : j;
                if (realIndex + 1 != fms.getSize()) {
                    mapNeighbors[fms.at(i)]->at(realIndex) = {fms.at(j), fms.distance(fms.at(j), fms.at(i))};
                }
            }
            std::sort(mapNeighbors[fms.at(i)]->begin(), mapNeighbors[fms.at(i)]->end(), [](std::pair<T, real> A, std::pair<T, real> B) {return A.second < B.second;});
        }

    }

    template<typename T> std::tuple<real, real, real, real> distortion(const FiniteMetricSpace<T>& fms, const Tree<T>& tree) {

        UndirectedGraph g;
        Utils::OrderedSet<T> nodes = tree.getNodesConst();
        std::vector<std::pair<T, T>> edges = tree.edges_const();

        for (int i = 0; i < edges.size(); i++) {
            T u = edges[i].first;
            T v = edges[i].second;
            boost::add_edge(nodes.getIndex(u), nodes.getIndex(v), fms.distance(u, v), g);
        }

        WeightMap wmap = boost::get(boost::edge_weight, g);
        DistanceMatrix distances(boost::num_vertices(g));
        DistanceMatrixMap dm(distances, g);
        bool valid = boost::floyd_warshall_all_pairs_shortest_paths(g, dm, boost::weight_map(wmap));
        assert(valid);

        real maxExpand = -std::numeric_limits<real>::infinity();
        real maxContrac = -std::numeric_limits<real>::infinity();
        real cumExpand = 0;
        real cumContrac = 0;
        long count = 0;

        //#pragma omp parallel for default(none) shared(distances, fms, nodes) reduction(+:cumExpand, cumContrac, count) reduction(max:maxExpand, maxContrac)
        for (int i = 0; i < nodes.size(); i++) {
            for (int j = i + 1; j < nodes.size(); j++) {

                real dT = distances[i][j];
                real dH = fms.distance(nodes[i], nodes[j]);
                real expand = dT / dH;
                real contrac = dH / dT;

                maxExpand = std::fmax(maxExpand, expand);
                maxContrac = std::fmax(maxContrac, contrac);
                cumExpand += expand;
                cumContrac += contrac;
                count++;

            }

        }

        return std::tuple<real, real, real, real>(maxExpand, cumExpand / static_cast<real>(count), maxContrac, cumContrac / static_cast<real>(count));

    };

    template <class T> class StarDecomposition {
        // from https://dl.acm.org/doi/pdf/10.5555/1109557.1109565
    public:
        static void forward_cut(const FiniteMetricSpace<T>&, const T, const Utils::OrderedSet<T>&, const real, std::tuple<T, Utils::OrderedSet<T>>&, Random&, int);
        static void backward_cut(const FiniteMetricSpace<T>&, T, const Utils::OrderedSet<T>&, std::vector<std::unique_ptr<std::tuple<T, Utils::OrderedSet<T>>>>&, real, real, Random&, int);
        static void compute_aux(const FiniteMetricSpace<T>&, T, const Utils::OrderedSet<std::string>&, real, Tree<T>&, Random&, int);
        static void compute(const FiniteMetricSpace<T>&, const Utils::OrderedSet<std::string>&, Tree<T>&, const Args&);

    };

    template <class T> class MinimumSpanningTree {
    public:
        struct Element {
            int idx;
            real x;
            Element() : idx(-1), x(std::numeric_limits<real>::infinity()){};
            Element(int a, real b) : idx(a), x(b) {};
            bool operator<(const Element& rhs) const { return x < rhs.x; };
            bool operator>(const Element& rhs) const { return x > rhs.x; };
        };

        static void compute(const FiniteMetricSpace<T>&, const Utils::OrderedSet<std::string>&, Tree<T>&, const Args&);
    };

    template <class T> class MinimumSpanningTreeMemoryEfficient {
        // Prim algorithm
    public:
        struct SearchItem {
            std::pair<T, T> obj;
            real weight;
            int idx;
            SearchItem() : weight(std::numeric_limits<real>::infinity()) {};
            SearchItem(std::pair<T, T> o, real w, int i) : obj(o), weight(w), idx(i)  {}
        };
        #pragma omp declare reduction(search_item_min : struct SearchItem : omp_out = omp_in.weight < omp_out.weight ? omp_in : omp_out)
        static void compute(const FiniteMetricSpace<T>&, const Utils::OrderedSet<std::string>&, Tree<T>&, const Args&);
    };

    template <class T> class NaiveTree {
    public:
        static void compute(const FiniteMetricSpace<T>&, const Utils::OrderedSet<std::string>&, Tree<T>&, const Args&);
    };
};

#endif //DEMBEDDER_ALGORITHMS_H
