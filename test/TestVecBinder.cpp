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
#include "../src/VecBinder.h"
#include <sstream>

#define VEC_THRES 1e-4


BOOST_AUTO_TEST_CASE(loadvec) {

    std::stringstream sstream;

    sstream
            << "4 3" << "\n"
            << "A 1.5 0 2" << "\n"
            << "B 2.665 0.22 0.1" << "\n"
            << "C 0.98 5.34 2.18" << "\n"
            << "D 0.4 0.2 0.1" << "\n";

    VecBinder w2v(sstream, VecBinder::FILE_FORMAT::vec);
    BOOST_CHECK_EQUAL(w2v.getDim(), 3);
    BOOST_CHECK_EQUAL(w2v.getVocSize(), 4);
    BOOST_CHECK_EQUAL(w2v.size(), 3 * 4);

    const real* vec = w2v.at(0);
    BOOST_CHECK_CLOSE(*(vec + 0),1.5, VEC_THRES);
    BOOST_CHECK_CLOSE(*(vec + 1),0, VEC_THRES);
    BOOST_CHECK_CLOSE(*(vec + 2),2, VEC_THRES);

    vec = w2v.at(1);
    BOOST_CHECK_CLOSE(*(vec + 0), 2.665, VEC_THRES);
    BOOST_CHECK_CLOSE(*(vec + 1), 0.22, VEC_THRES);
    BOOST_CHECK_CLOSE(*(vec + 2), 0.1, VEC_THRES);

    vec = w2v.at(2);
    BOOST_CHECK_CLOSE(*(vec + 0),0.98, VEC_THRES);
    BOOST_CHECK_CLOSE(*(vec + 1),5.34, VEC_THRES);
    BOOST_CHECK_CLOSE(*(vec + 2),2.18, VEC_THRES);

    vec = w2v.at(3);
    BOOST_CHECK_CLOSE(*(vec + 0),0.4, VEC_THRES);
    BOOST_CHECK_CLOSE(*(vec + 1),0.2, VEC_THRES);
    BOOST_CHECK_CLOSE(*(vec + 2),0.1, VEC_THRES);

}