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
#include "../src/Utils.h"


BOOST_AUTO_TEST_CASE(ordereset_test) {

    Utils::OrderedSet<std::string> oset;
    BOOST_CHECK_EQUAL(oset.getFreq("test"), 0);
    BOOST_CHECK_EQUAL(oset.size(), 0);

    std::string elA = "A";
    std::string elAbis = "A";
    oset.insert(elA);
    oset.insert(elAbis);
    BOOST_CHECK_EQUAL(oset.getFreq("A"), 2);
    BOOST_CHECK_EQUAL(oset[0], "A");
    BOOST_CHECK_EQUAL(oset.getIndex("A"), 0);
    BOOST_CHECK_EQUAL(oset.size(), 1);

    std::string elB = "B";
    std::string elC = "C";
    oset.insert(elB);
    oset.insert(elC);
    BOOST_CHECK_EQUAL(oset.getFreq("B"), 1);
    BOOST_CHECK_EQUAL(oset.getFreq("C"), 1);
    BOOST_CHECK_EQUAL(oset.getIndex("B"), 1);
    BOOST_CHECK_EQUAL(oset.getIndex("C"), 2);
    BOOST_CHECK_EQUAL(oset[1], "B");
    BOOST_CHECK_EQUAL(oset[2], "C");
    BOOST_CHECK_EQUAL(oset.size(), 3);

    std::vector<std::string> v = oset.toVec();
    BOOST_CHECK_EQUAL(v[0], "A");
    BOOST_CHECK_EQUAL(v[1], "B");
    BOOST_CHECK_EQUAL(v[2], "C");

    Utils::OrderedSet<std::string> other = oset;
    BOOST_CHECK_EQUAL(other.getFreq("A"), 2);
    BOOST_CHECK_EQUAL(other[0], "A");
    BOOST_CHECK_EQUAL(other.getIndex("A"), 0);
    BOOST_CHECK_EQUAL(other.getFreq("B"), 1);
    BOOST_CHECK_EQUAL(other.getFreq("C"), 1);
    BOOST_CHECK_EQUAL(other.getIndex("B"), 1);
    BOOST_CHECK_EQUAL(other.getIndex("C"), 2);
    BOOST_CHECK_EQUAL(other[1], "B");
    BOOST_CHECK_EQUAL(other[2], "C");
    BOOST_CHECK_EQUAL(other.size(), 3);

    std::string elD = "D";
    other.insert(elD);
    BOOST_CHECK_EQUAL(oset.size(), 3);
    BOOST_CHECK_EQUAL(other.size(), 4);
    BOOST_CHECK_EQUAL(other[3], "D");
    BOOST_CHECK_EQUAL(other[2], "C");
    BOOST_CHECK_EQUAL(other[1], "B");
    BOOST_CHECK_EQUAL(other[0], "A");
    BOOST_CHECK_EQUAL(oset[0], "A");
    BOOST_CHECK_EQUAL(oset[1], "B");
    BOOST_CHECK_EQUAL(oset[2], "C");

}