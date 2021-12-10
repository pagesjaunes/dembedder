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

#include "Algorithms.h"

template<class T>
void
Algorithms::StarDecomposition<T>::forward_cut(const FiniteMetricSpace<T>& fms, const T root, const Utils::OrderedSet<T>& voc, const real delta, std::tuple<T, Utils::OrderedSet<T>>& G0Info, Random& rdhandler, int verbose) {

    assert(voc.getFreq(root) > 0);

    // Draw random cut radius
    real gamma = rdhandler.uniform(delta / 4, delta / 2);

    // Create graph
    Utils::OrderedSet<T>& componentVoc = std::get<1>(G0Info);
    componentVoc.insert(root);

    if (verbose > 0) {
        std::cout << "\t" << "Build component 0" << std::endl;
    }
    #pragma omp parallel for default(none) shared(voc, componentVoc, fms, root, gamma)
    for (int i = 0; i < voc.size(); i++) {
        T objA = voc[i];
        #pragma omp parallel for default(none) shared(voc, componentVoc, fms, i, objA, root, gamma)
        for (int j = i + 1; j < voc.size(); j++) {
            T objB = voc[j];
            real dist_root_A = fms.distance(root, objA);
            real dist_root_B = fms.distance(root, objB);
            if ((dist_root_A <= gamma && gamma <= dist_root_B) || (dist_root_B <= gamma && gamma <= dist_root_A)) {
            } else {
                if (dist_root_A <= gamma && dist_root_B <= gamma) {
                    #pragma omp critical (append)
                    {
                        componentVoc.insert(voc[i]);
                        componentVoc.insert(voc[j]);
                    }
                }
            }
        }
    }

}

template<class T>
void
Algorithms::StarDecomposition<T>::backward_cut(const FiniteMetricSpace<T>& fms, T root, const Utils::OrderedSet<T>& voc,
                                               std::vector<std::unique_ptr<std::tuple<T, Utils::OrderedSet<T>>>>& Gcomponents, real delta, real tol, Random& rdhandler, int verbose) {

    // Extract root graph info
    assert(Gcomponents.size() == 1);

    T G0root = std::get<0>(*Gcomponents[0]);
    Utils::OrderedSet<T>& G0voc = std::get<1>(*Gcomponents[0]);

    std::function<real(T, T)> lMetric;
    if (tol < 0) {
        lMetric = [fms, G0root, tol](T x, T y) {
            return fms.distance(G0root, y) + fms.distance(x, y) < fms.distance(G0root, x) ? 1 : 0;
        };
    } else {
        lMetric = [fms, G0root, tol](T x, T y) {
            return std::abs(fms.distance(G0root, y) + fms.distance(x, y) - fms.distance(G0root, x)) < tol ? 1 : 0;
        };
    }

    // Extract unattached word voc
    if (verbose > 0) {
        std::cout << "\t" << "Find unattached elements" << std::endl;
    }
    Utils::OrderedSet<std::string> unattached;
    #pragma omp parallel for default(none) shared(G0voc, voc, unattached)
    for (int i = 0; i < voc.size(); i++) {
        if (!G0voc.getFreq(voc[i])) {
            #pragma omp critical (insert)
            {
                unattached.insert(voc[i]);
            }
        }
    }

    // Create opposite graph
    UndirectedGraph GMinus;
    if (verbose > 0) {
        std::cout << "\t" << "Create opposite graph" << std::endl;
    }
    #pragma omp parallel for default(none) shared(GMinus, lMetric, unattached)
    for (int i = 0; i < unattached.size(); i++) {
        #pragma omp critical (add_edge)
        {
            boost::add_vertex(i, GMinus);
        }
        #pragma omp parallel for default(none) shared(GMinus, lMetric, unattached, i)
        for (int j = 0; j < unattached.size(); j++) {
            real d = lMetric(unattached[i], unattached[j]);
            if (d > 0) {
                #pragma omp critical (add_edge)
                {
                    boost::add_edge(i, j, d, GMinus);
                }
            }
        }
    }

    // Find shortest paths between any pair
    WeightMap wmap = boost::get(boost::edge_weight, GMinus);
    DistanceMatrix distances(boost::num_vertices(GMinus));
    DistanceMatrixMap dm(distances, GMinus);
    if (verbose > 0) {
        std::cout << "\t" << "Find shortests paths" << std::endl;
    }
    bool valid = boost::floyd_warshall_all_pairs_shortest_paths(GMinus, dm, boost::weight_map(wmap));
    assert(valid);

    Utils::OrderedSet<T> set_indexer = unattached;
    if (verbose > 0) {
        std::cout << "\t" << "Find components" << std::endl;
    }
    while (unattached.size() > 0) {

        std::cout << "\tprogress: " << unattached.size() << " remaining." << "\r" << std::flush;
        //std::vector<int> arange(unattached.size());
        //std::iota(arange.begin(), arange.end(), 0);
        //std::shuffle(arange.begin(), arange.end(), rdhandler.generator());

        T xi = fms.centroid(unattached); //unattached[arange[0]];
        Utils::OrderedSet<T> componentVoc;
        componentVoc.insert(xi);
        real gamma = delta * rdhandler.exponential(std::log(voc.size()));


        #pragma omp parallel for default(none) shared(distances, unattached, gamma, lMetric, set_indexer, componentVoc, xi)
        for (int i = 0; i < unattached.size(); i++) {
            #pragma omp parallel for default(none) shared(distances, unattached, gamma, lMetric, set_indexer, componentVoc, xi, i)
            for (int j = 0; j < unattached.size(); j++) {
                T u = unattached[i];
                T v = unattached[j];

                real d_xi_u = distances[set_indexer.getIndex(xi)][set_indexer.getIndex(u)];
                real d_xi_v = distances[set_indexer.getIndex(xi)][set_indexer.getIndex(v)];

                if (d_xi_u <= gamma && gamma < d_xi_v) {
                } else {
                    if (lMetric(u, v) > 0 && d_xi_u <= gamma && d_xi_v <= gamma) {
                        #pragma omp critical (compVocAdd)
                        {
                            componentVoc.insert(u);
                            componentVoc.insert(v);
                        }
                    }
                }
            }
        }

        Utils::OrderedSet<T> next_unattached;
        #pragma omp parallel for default(none) shared(unattached, next_unattached, componentVoc)
        for (int i = 0; i < unattached.size(); i++) {
            if (!componentVoc.getFreq(unattached[i])) {
                #pragma omp critical (add_next_unattached)
                {
                    next_unattached.insert(unattached[i]);
                }
            }
        }

        std::unique_ptr<std::tuple<T, Utils::OrderedSet<T>>> GiInfo(new std::tuple<T, Utils::OrderedSet<T>>(xi, componentVoc));
        Gcomponents.push_back(std::move(GiInfo));
        unattached = next_unattached;

    }
    std::cout << std::endl;

}

template<class T>
void Algorithms::StarDecomposition<T>::compute(const FiniteMetricSpace<T>& fms, const Utils::OrderedSet<std::string>& voc, Tree<T>& tree, const Args& args) {
    std::string root = args.root == "" ? fms.centroid(voc) : args.root;
    tree.set_centroid(root);
    Random rdhandler;
    compute_aux(fms, root, voc, args.tol, tree, rdhandler, args.verbose);
}

template<class T>
void
Algorithms::StarDecomposition<T>::compute_aux(const FiniteMetricSpace<T>& fms, T root, const Utils::OrderedSet<std::string>& voc, real tol, Tree<T>& tree, Random& rdhandler, int verbose) {

    assert(voc.getFreq(root) > 0);

    if (voc.size() > 1) {

        if (verbose > 0) {
            std::cout << "find radius" << std::endl;
        }
        real delta = fms.getRadiusOn(voc);

        // forward cut
        if (verbose > 0) {
            std::cout << "forward cut" << std::endl;
        }
        Utils::OrderedSet<T> G0voc;
        std::unique_ptr<std::tuple<T, Utils::OrderedSet<T>>> G0Info(new std::tuple<T, Utils::OrderedSet<T>>(root, G0voc));
        forward_cut(fms, root, voc, delta, *G0Info, rdhandler, verbose);

        // backward cut
        if (verbose > 0) {
            std::cout << "backward cut" << std::endl;
        }
        std::vector<std::unique_ptr<std::tuple<T, Utils::OrderedSet<T>>>> Gcomponents;
        Gcomponents.push_back(std::move(G0Info));
        backward_cut(fms, root, voc, Gcomponents, delta, tol, rdhandler, verbose);

        // Recursive application
        if (verbose > 0) {
            std::cout << "Apply recursion on " << Gcomponents.size() << " components" << std::endl;
        }
        std::vector<std::unique_ptr<std::tuple<Utils::OrderedSet<T>, T, T, Tree<T>>>> trees;
        for (int i = 0; i < Gcomponents.size(); i++) {

            // Get all required info
            T Giroot = std::get<0>(*Gcomponents[i]);
            Utils::OrderedSet<T>& Givoc = std::get<1>(*Gcomponents[i]);
            T nearest = fms.nearestNeighbor(Giroot, std::get<1>(*Gcomponents[0]));
            Tree<T> GiTree;
            assert((i == 0 && nearest == Giroot) || (i != 0 && nearest != Giroot));

            // Recursive computation
            std::unique_ptr<std::tuple<Utils::OrderedSet<T>, T, T, Tree<T>>> GiTreeInfo(new std::tuple<Utils::OrderedSet<T>, T, T, Tree<T>>(Givoc, Giroot, nearest, GiTree));
            compute_aux(fms, Giroot, std::get<0>(*GiTreeInfo), tol, std::get<3>(*GiTreeInfo), rdhandler, 0);

            trees.push_back(std::move(GiTreeInfo));
        }

        // Combine trees
        if (verbose > 0) {
            std::cout << "Recompose trees" << std::endl;
        }
        for (int i = 0; i < trees.size(); i++) {
            std::vector<std::pair<T,T>>& edges = std::get<3>(*trees[i]).edges_vec();
            for (int ke = 0; ke < edges.size(); ke++) {
                tree.add_edge(edges[ke]);
            }
            if (i > 0) {
                tree.add_edge({std::get<2>(*trees[i]), std::get<1>(*trees[i])});
            }
        }

        trees.clear();
        Gcomponents.clear();

    } else {

        tree.add_node(root);

    }
}

template<class T>
void Algorithms::MinimumSpanningTree<T>::compute(const FiniteMetricSpace<T>& fms, const Utils::OrderedSet<std::string>& voc, Tree<T>& tree, const Args& args) {

    T root = args.root == "" ? fms.centroid(voc) : args.root;
    tree.set_centroid(root);

    UndirectedGraph g;
    if (args.verbose > 0) {
        std::cout << "Create graph" << std::endl;
    }

    int k = static_cast<int>(std::ceil(args.tol));

    for (int i = 0; i < voc.size(); i++) {
        std::priority_queue<Element, std::vector<Element>, std::greater<Element>> distances;
        for (int j = 0; j < voc.size(); j++) {

            if (i != j) {
                if (distances.size() < k) {
                    distances.push(Element(j, -fms.distance(voc[i], voc[j])));
                } else {
                    distances.pop();
                    distances.push(Element(j, -fms.distance(voc[i], voc[j])));
                }
            }
        }

        for (int j = 0; j < k; j++) {
            Element e = distances.top();
            boost::add_edge(i, e.idx, -e.x, g);
            distances.pop();
        }

        std::cout << "progress: " << (i+1) << " / " << voc.size() << "\r" << std::flush;
    }

    std::cout << std::endl;

    std::vector < boost::graph_traits < UndirectedGraph >::vertex_descriptor > p(boost::num_vertices(g));
    if (args.verbose > 0) {
        std::cout << "Compute MST" << std::endl;
    }
    boost::prim_minimum_spanning_tree(g, &p[0]);
    for (std::size_t i = 0; i < p.size(); i++) {
        tree.add_edge({voc[i], voc[p[i]]});
    }

}

template<class T>
void Algorithms::NaiveTree<T>::compute(const FiniteMetricSpace<T>& fms, const Utils::OrderedSet<std::string>& voc, Tree<T>& tree, const Args& args) {

    // preprocess
    std::function<real(const std::string &, const ::std::string &)> disfunc = [fms, args](const std::string& a, const std::string& b) {
        // B is son, A is parent --> norm(A) < norm(B)
        real normA = Geometry::dot(args.vb->at(a), args.vb->at(a), args.vb->getDim());
        real normB = Geometry::dot(args.vb->at(b), args.vb->at(b), args.vb->getDim());
        if (normA >= normB) {
            return std::numeric_limits<real>::infinity();
        } else {
            return fms.distance(a, b);
        }
    };
    std::vector<T> elements = voc.toVec();
    FiniteMetricSpace<T> orderedFms(elements, disfunc);

    real minNorm = std::numeric_limits<real>::infinity();
    T argMinNorm;
    for (int i = 0; i < voc.size(); i++) {
        real norm = Geometry::dot(args.vb->at(voc[i]), args.vb->at(voc[i]), args.vb->getDim());
        if (norm < minNorm) {
            argMinNorm = voc[i];
            minNorm = norm;
        }
    }

    // program
    T seed =  args.root == "" ? fms.centroid(voc) : args.root;
    tree.set_centroid(seed);
    std::function<bool (const T&, const T&)> notEqual = [] (const T& a, const T& b) {return a != b;};
    for (int i = 0; i < voc.size(); i++) {
        if (voc[i] != argMinNorm) {
            T parent = orderedFms.nearestNeighbor(voc[i], voc, notEqual);
            tree.add_edge({voc[i], parent});
        }
        std::cout << "progress: " << (i + 1) << " / " << voc.size() << "\r" << std::flush;
    }

}

template <class T>
void Algorithms::MinimumSpanningTreeMemoryEfficient<T>::compute(const FiniteMetricSpace<T>& fms,
                                                             const Utils::OrderedSet<std::string>& voc, Tree<T>& tree,
                                                             const Args & args) {

    // Using Prim's algorithm

    T seed = args.root == "" ? fms.centroid(voc) : args.root;
    tree.set_centroid(seed);

    Utils::OrderedSet<T>& nodes = tree.getNodes();
    std::vector<T> vertices;
    std::set<T> vertices_set;
    for (int i = 0; i < voc.size(); i++) {
        if (voc[i] != seed) {
            vertices.push_back(voc[i]);
            vertices_set.insert(voc[i]);
        }
    }

    std::map<T, T> nearestMap;
    std::function<bool (const T&, const T&)> different = [] (const T& a, const T& b) {return a != b;};

    std::cout << std::endl << "\tProcessing (Prim):" << std::endl;
    while (vertices.size() > 0) {

        SearchItem si;

        #pragma omp parallel for default(none) shared(voc, nodes, fms, vertices_set, vertices, nearestMap, different) reduction(search_item_min:si)
        for (int i = 0; i < nodes.size(); i++) {
            #pragma omp critical (update_nearestMap)
            {
                if (nearestMap.count(nodes[i]) == 0 || nodes.getFreq(nearestMap[nodes[i]]) > 0) {
                    int idx = fms.nearestNeighbor(nodes[i], vertices, different);
                    nearestMap[nodes[i]] = vertices[idx];
                }
            }

            T nn = nearestMap[nodes[i]];
            real weight = fms.distance(nn, nodes[i]);
            if (nodes.getFreq(nn) == 0 && weight < si.weight) {
                for (int k = 0; k < vertices.size(); k++) {
                    if (vertices[k] == nn) {
                        si.idx = k;
                        break;
                    }
                }
                si.weight = weight;
                si.obj = {nn, nodes[i]};
            }
        }

        assert(si.idx < vertices.size());
        assert(nodes.getFreq(si.obj.first) == 0);

        tree.add_edge(si.obj);
        vertices_set.erase(vertices[si.idx]);
        vertices.erase(vertices.begin() + si.idx);

        std::cout << "\tprogress: " << nodes.size() << " / " << voc.size() << "\r" << std::flush;
    }

}

#include "Tree.cpp"
#include "FiniteMetricSpace.cpp"
template class Algorithms::StarDecomposition<std::string>;
template class Algorithms::MinimumSpanningTree<std::string>;
template class Algorithms::MinimumSpanningTreeMemoryEfficient<std::string>;
template class Algorithms::NaiveTree<std::string>;
