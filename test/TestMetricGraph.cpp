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

#define BOOST_TEST_MAIN
#if !defined( WIN32 )
#define BOOST_TEST_DYN_LINK
#endif

#include "boost/test/unit_test.hpp"
#include "../src/MetricGraph.h"
#include <sstream>
#define THRES 1e-4

BOOST_AUTO_TEST_CASE(loadcsv) {

    std::stringstream sstream;

    sstream
    << "node1,node2,3.0" << "\n"
    << "node2,node3,4" << "\n"
    << "node3,node4,1.2" << "\n"
    << "node4,node2,0.5" << "\n";

    MetricGraph<std::string> mgraph = MetricGraphCsvReader(sstream, ',');
    std::vector<std::string> nodes = mgraph.elements();
    BOOST_CHECK_EQUAL(nodes[0], "node1");
    BOOST_CHECK_EQUAL(nodes[1], "node2");
    BOOST_CHECK_EQUAL(nodes[2], "node3");
    BOOST_CHECK_EQUAL(nodes[3], "node4");
    BOOST_CHECK_EQUAL(nodes.size(), 4);

    std::function<real (std::string, std::string)> func = mgraph.metric_distance_function();
    BOOST_CHECK_CLOSE(func("node1", "node2"), 3.0, THRES);
    BOOST_CHECK_CLOSE(func("node2", "node3"), 4.0, THRES);
    BOOST_CHECK_CLOSE(func("node3", "node4"), 1.2, THRES);
    BOOST_CHECK_CLOSE(func("node4", "node2"), 0.5, THRES);

    BOOST_CHECK_CLOSE(mgraph.eccentricity("node2"), 4, THRES);
    BOOST_CHECK_CLOSE(mgraph.radius(), 1.2, THRES);

}