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

#include "Args.h"

Args::Args() {

}

Args::Args(char* argv[]) {

    argh::parser cmdl(argv);
    distortion = false;
    centroid = false;
    tree_distortion = false;
    neighbors = false;

    if (cmdl["distortion"]) {
        distortion = true;
        if (!(cmdl("input") >> input)) {
            throw std::string("Should specify -input parameter");
        }
        if (!(cmdl("tree_input", "") >> tree_input)) {
            throw std::string("Should specify -tree_input parameter");
        }
        if (!(cmdl("output") >> output)) {
            throw std::string("Should specify -output parameter");
        }

    } else if (cmdl["centroid"]) {
        centroid = true;

        if (!(cmdl("input") >> input)) {
            throw std::string("Should specify -input parameter");
        }

        if (!(cmdl("output") >> output)) {
            throw std::string("Should specify -output parameter");
        }

    } else if (cmdl["neighbors"]) {
            neighbors = true;
            if (!(cmdl("input") >> input)) {
                throw std::string("Should specify -input parameter");
            }
            if (!(cmdl("output") >> output)) {
                throw std::string("Should specify -output parameter");
            }


    }  else if (cmdl["tree_distortion"]) {
        tree_distortion = true;
        distortion = true;
        if (!(cmdl("tree_input_1") >> tree_input_1)) {
            throw std::string("Should specify -input parameter");
        }
        if (!(cmdl("tree_input_2", "") >> tree_input_2)) {
            throw std::string("Should specify -tree_input parameter");
        }
        if (!(cmdl("output") >> output)) {
            throw std::string("Should specify -output parameter");
        }

    }  else {

        if (!(cmdl("input") >> input)) {
            throw std::string("Should specify -input parameter");
        }

        if (!(cmdl("output") >> output)) {
            throw std::string("Should specify -output parameter");
        }

        if (!(cmdl("algorithm") >> algorithm)) {
            throw std::string("Should specify -algorithm parameter");
        }

    }


    cmdl("delimiter", ',') >> delimiter;
    cmdl("root", "") >> root;
    cmdl("tol", 1e-4) >> tol;
    cmdl("distance", "lorentzian") >> distance;
    cmdl("dparam", 0.01) >> dparam;
    cmdl("verbose", 1) >> verbose;
    cmdl("tree_input", "") >> tree_input;

    vb = NULL;

}

void Args::display() const {

    if (!distortion) {
        std::cout
                << "\t" << "Input file (-input): " << input << std::endl
                << "\t" << "Output file (-output): " << output << std::endl
                << "\t" << "Algorithm (-algorithm={NAIVE,MST,MST-EFF,SMID,ABRAHAM,STAR}): " << algorithm << std::endl;

        if (input.substr(input.find_last_of(".") + 1) == "csv") {
            std::cout
                    << "\t" << "Delimiter (csv mode, -delimiter): " << delimiter << std::endl;
        }

        std::cout
                << "\t" << "Root (-root): " << root << std::endl
                << "\t" << "Distance (-distance={euclidean,poincare,lorentzian}): " << distance << std::endl
                << "\t" << "Distance param (beta, celerity, p-distance, -dparam): " << dparam << std::endl
                << "\t" << "Verbosity (-verbose): " << verbose << std::endl;

        if (algorithm == "STAR" || algorithm == "NPOINT" || algorithm == "ASWAP") {
            std::cout
                    << "\t" << "Tolerance (for star, aswap, npoint, -tol): " << tol << std::endl;
        }

        if (algorithm == "ASWAP") {
            std::cout
                    << "\t" << "Tree input file: " << tree_input << std::endl;
        }

        std::cout << std::endl;
    } else {
        std::cout
                << "\t" << "Input file (-input): " << input << std::endl
                << "\t" << "Tree input file (for accurate swapping): " << tree_input << std::endl
                << "\t" << "Output file (-output): " << output << std::endl
                << "\t" << "Distance (-distance={euclidean,poincare,lorentzian}): " << distance << std::endl
                << "\t" << "Distance param (beta, celerity, p-distance, -dparam): " << dparam << std::endl;

        if (input.substr(input.find_last_of(".") + 1) == "csv") {
            std::cout
                    << "\t" << "Delimiter (csv mode, -delimiter): " << delimiter << std::endl;
        }

    }
}
