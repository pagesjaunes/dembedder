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
#include "../src/Geometry.h"
#include <sstream>

#define GEO_THRES 1e-4


BOOST_AUTO_TEST_CASE(distances) {

    std::stringstream sstream;

    sstream
            << "4 3" << "\n"
            << "A 0.25 0 0" << "\n"
            << "B 1 1 1" << "\n"
            << "C 0.4 0.1 0.2" << "\n"
            << "D 0.2 0.2 0.1" << "\n";

    VecBinder w2v(sstream, VecBinder::FILE_FORMAT::vec);

    // p distance
    BOOST_CHECK_CLOSE(Geometry::pDistance(w2v.at(0), w2v.at(1), 1.0, 3), 2.75, GEO_THRES);

    // Poincaré
    BOOST_CHECK_CLOSE(Geometry::poincareDistance(w2v.at(2), w2v.at(3), 1.0, 3), 0.5700423030150583, GEO_THRES);
    BOOST_CHECK_CLOSE(Geometry::poincareDistance(w2v.at(2), w2v.at(3), 0.5, 3), 0.5268463116630734, GEO_THRES);
    BOOST_CHECK_CLOSE(Geometry::poincareDistance(w2v.at(2), w2v.at(3), 1.5, 3), 0.6213318117937418, GEO_THRES);

    // Lorentzian
    BOOST_CHECK_CLOSE(Geometry::squaredLorentzianDistance(w2v.at(2), w2v.at(3), 1.0, 3), 0.3338433718180558, 10 * GEO_THRES);
    BOOST_CHECK_CLOSE(Geometry::squaredLorentzianDistance(w2v.at(2), w2v.at(3), 0.5, 3), 0.2917572378798532, 10 * GEO_THRES);
    BOOST_CHECK_CLOSE(Geometry::squaredLorentzianDistance(w2v.at(2), w2v.at(3), 1.5, 3), 0.35765206891502066, 10 * GEO_THRES);
    BOOST_CHECK_CLOSE(Geometry::squaredLorentzianDistance(w2v.at(2), w2v.at(3), 0.01, 3), 0.19772215217941905, 10 * GEO_THRES);

    std::stringstream sstream_bis;
    sstream_bis
        << "2 10" << "\n"
        << "A 0.206659 -0.796929 -0.062148 -0.301514 -0.285774 -0.211337 0.238666 -0.191998 -0.024298 0.071451" << "\n"
        << "B -0.563672 0.224041 0.256587 0.068536 0.260257 0.345382 0.031667 -0.305425 0.073372 -0.523891" << "\n";

    VecBinder w2v_bis(sstream_bis, VecBinder::FILE_FORMAT::vec);
    BOOST_CHECK_CLOSE(Geometry::poincareDistance(w2v_bis.at(0), w2v_bis.at(1), 1, 10), 16.754643933660425, 10 * GEO_THRES);


}