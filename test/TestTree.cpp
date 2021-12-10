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

#include "boost/test/unit_test.hpp"
#include "../src/Tree.h"
#include <sstream>

BOOST_AUTO_TEST_CASE(tree_add) {

    Tree<std::string> t;

    t.add_edge({"A", "B"});
    t.add_edge({"B", "C"});
    t.add_edge({"A", "D"});

    std::vector<std::pair<std::string, std::string>> edges = t.edges_vec();
    BOOST_CHECK_EQUAL(edges[0].first, "A");
    BOOST_CHECK_EQUAL(edges[0].second, "B");
    BOOST_CHECK_EQUAL(edges[1].first, "B");
    BOOST_CHECK_EQUAL(edges[1].second, "C");
    BOOST_CHECK_EQUAL(edges[2].first, "A");
    BOOST_CHECK_EQUAL(edges[2].second, "D");

}

BOOST_AUTO_TEST_CASE(tree_store) {

    Tree<std::string> t;
    std::stringstream sstream;

    t.set_centroid("A");
    t.add_edge({"A", "B"});
    t.add_edge({"B", "C"});
    t.add_edge({"A", "D"});

    t.store(sstream);
    BOOST_CHECK_EQUAL(sstream.str(), "centroid\tA\nA\tB\nB\tC\nA\tD\n");

}

BOOST_AUTO_TEST_CASE(load_tree) {

    std::stringstream sstream;

    sstream
    << "centroid\tA\n"
    << "A\tB\n"
    << "B\tC\n"
    << "A\tD\n";

    Tree<std::string> tree(sstream);

    Utils::OrderedSet<std::string>& nodes = tree.getNodes();
    BOOST_CHECK_EQUAL(nodes.size(), 4);
    BOOST_CHECK_EQUAL(nodes.getFreq("A"), 2);
    BOOST_CHECK_EQUAL(nodes.getFreq("B"), 2);
    BOOST_CHECK_EQUAL(nodes.getFreq("C"), 1);
    BOOST_CHECK_EQUAL(nodes.getFreq("D"), 1);

    std::vector<std::pair<std::string, std::string>>& edges = tree.edges_vec();
    BOOST_CHECK_EQUAL(edges[0].first, "A");
    BOOST_CHECK_EQUAL(edges[0].second, "B");
    BOOST_CHECK_EQUAL(edges[1].first, "B");
    BOOST_CHECK_EQUAL(edges[1].second, "C");
    BOOST_CHECK_EQUAL(edges[2].first, "A");
    BOOST_CHECK_EQUAL(edges[2].second, "D");

    tree.store(sstream);
    BOOST_CHECK_EQUAL(sstream.str(), "centroid\tA\nA\tB\nB\tC\nA\tD\n");


}