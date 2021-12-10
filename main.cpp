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

#include <iostream>
#include <fstream>

#include "src/Args.h"
#include "src/MetricGraph.h"
#include "src/VecBinder.h"
#include "src/Geometry.h"
#include "src/FiniteMetricSpace.h"
#include "src/Algorithms.h"
#include "src/Tree.h"
#include "src/Random.h"

int main(int argc, char* argv[]) {

    Args args(argv);
    args.display();
    std::ifstream ifs;

    if (args.tree_distortion) {
        ifs.open(args.tree_input_1);
        Tree<std::string> predTree(ifs);
        ifs.close();

        ifs.open(args.tree_input_2);
        Tree<std::string> goldTree(ifs);
        ifs.close();

        std::tuple<real, real, real, real> answers = Algorithms::treeDistortion<std::string>(goldTree, predTree);

        std::ofstream ofs;
        ofs.open(args.output);
        ofs << "expansion-max " << std::get<0>(answers) << "\n"
            << "expansion-avg " << std::get<1>(answers) << "\n"
            << "contraction-max " << std::get<2>(answers) << "\n"
            << "contraction-avg " << std::get<3>(answers) << "\n";
        ofs.close();

        return EXIT_SUCCESS;

    } else if (args.centroid) {

        ifs.open(args.input);
        std::vector<std::string> elements;
        std::function<real(const std::string &, const ::std::string &)> disfunc;

        VecBinder *vb = new VecBinder(ifs, VecBinder::FILE_FORMAT::vec);

        elements = vb->getElements();

        if (args.distance == "euclidean") {
            disfunc = [vb, p = args.dparam](const std::string &a, const std::string &b) {
                return Geometry::pDistance(vb->at(a), vb->at(b), p, vb->getDim());
            };
        } else if (args.distance == "lorentzian") {
            disfunc = [vb, beta = args.dparam](const std::string &a, const std::string &b) {
                return Geometry::squaredLorentzianDistance(vb->at(a), vb->at(b), beta, vb->getDim());
            };
        } else if (args.distance == "poincare") {
            disfunc = [vb, cel = args.dparam](const std::string &a, const std::string &b) {
                return Geometry::poincareDistance(vb->at(a), vb->at(b), cel, vb->getDim());
            };
        } else if (args.distance == "Henergy") {
            disfunc = [vb, K = args.dparam](const std::string &a, const std::string &b) {
                return 10000 - Geometry::hyperbolic_energy(vb->at(a), vb->at(b), K, vb->getDim());
            };
        } else if (args.distance == "Eenergy") {
            disfunc = [vb, K = args.dparam](const std::string &a, const std::string &b) {
                return 10000 - Geometry::euclidean_energy(vb->at(a), vb->at(b), K, vb->getDim());
            };
        } else {
            throw std::string("distance should be: lorentzian / poincare / euclidean");
        }

        ifs.close();

        FiniteMetricSpace<std::string> fms(elements, disfunc);
        Utils::OrderedSet<std::string> voc = fms.getObjects();
        std::string centroid = fms.centroid(voc);
        std::ofstream ofs;
        ofs.open(args.output);
        ofs << "centroid\t" << centroid;
        ofs.close();

    } else {

        ifs.open(args.input);
        std::vector<std::string> elements;
        std::function<real(const std::string &, const ::std::string &)> disfunc;

        if (args.input.substr(args.input.find_last_of(".") + 1) == "csv") {
            assert(args.algorithm != "NAIVE");

            MetricGraph<std::string> mgraph = MetricGraphCsvReader(ifs, args.delimiter);
            ifs.close();

            elements = mgraph.elements();
            disfunc = [mgraph](const std::string &a, const std::string &b) {
                return mgraph.metric_distance(a, b);
            };

        } else if (args.input.substr(args.input.find_last_of(".") + 1) == "vec") {

            VecBinder *vb = new VecBinder(ifs, VecBinder::FILE_FORMAT::vec);

            elements = vb->getElements();

            if (args.distance == "euclidean") {
                disfunc = [vb, p = args.dparam](const std::string &a, const std::string &b) {
                    return Geometry::pDistance(vb->at(a), vb->at(b), p, vb->getDim());
                };
            } else if (args.distance == "lorentzian") {
                disfunc = [vb, beta = args.dparam](const std::string &a, const std::string &b) {
                    return Geometry::squaredLorentzianDistance(vb->at(a), vb->at(b), beta, vb->getDim());
                };
            } else if (args.distance == "poincare") {
                disfunc = [vb, cel = args.dparam](const std::string &a, const std::string &b) {
                    return Geometry::poincareDistance(vb->at(a), vb->at(b), cel, vb->getDim());
                };
            } else if (args.distance == "Henergy") {
                disfunc = [vb, K = args.dparam](const std::string &a, const std::string &b) {
                    return 10000 - Geometry::hyperbolic_energy(vb->at(a), vb->at(b), K, vb->getDim());
                };
            } else if (args.distance == "Eenergy") {
                disfunc = [vb, K = args.dparam](const std::string &a, const std::string &b) {
                    return 10000 - Geometry::euclidean_energy(vb->at(a), vb->at(b), K, vb->getDim());
                };
            } else {
                throw std::string("distance should be: lorentzian / poincare / euclidean");
            }

            if (args.algorithm == "NAIVE") {
                args.vb = vb;
            }

        } else {
            throw std::string("File format should be .csv or .vec");
        }
        ifs.close();

        FiniteMetricSpace<std::string> fms(elements, disfunc);
        Utils::OrderedSet<std::string> voc = fms.getObjects();


        if (args.distortion) {

            ifs.open(args.tree_input);
            Tree<std::string> tree(ifs);
            ifs.close();

            std::tuple<real, real, real, real> answers = Algorithms::distortion<std::string>(fms, tree);

            std::ofstream ofs;
            ofs.open(args.output);
            ofs << "expansion-max " << std::get<0>(answers) << "\n"
                << "expansion-avg " << std::get<1>(answers) << "\n"
                << "contraction-max " << std::get<2>(answers) << "\n"
                << "contraction-avg " << std::get<3>(answers) << "\n";
            ofs.close();

        } else if (args.tree_distortion) {


        } else if (args.neighbors) {

            std::map<std::string, std::vector<std::pair<std::string, real>> *> mapNeighbors;
            Algorithms::getNeighbors<std::string>(fms, mapNeighbors);

            std::ofstream ofs;
            ofs.open(args.output);
            ofs << "{\n";
            for (int i = 0; i < fms.getSize(); i++) {
                std::string word = fms.at(i);
                ofs << "\t" << "\"" << word << "\": [";
                for (int j = 0; j < mapNeighbors[word]->size(); j++) {
                    std::pair<std::string, real> pair = mapNeighbors[word]->at(j);
                    ofs << "[" << "\"" << pair.first << "\"," << pair.second << "]";
                    if (j + 1 < mapNeighbors[word]->size()) {
                        ofs << ",";
                    }
                }
                ofs << "]";
                if (i + 1 < fms.getSize()) {
                    ofs << ",";
                }
                ofs << "\n";
            }
            ofs << "}";

        } else {

            Tree<std::string> tree;


            if (args.algorithm == "NPOINT") {
                Algorithms::StarDecomposition<std::string> algStar;
                algStar.compute(fms, voc, tree, args);
            } else if (args.algorithm == "MST-EFF") {
                Algorithms::MinimumSpanningTreeMemoryEfficient<std::string> algMST;
                algMST.compute(fms, voc, tree, args);
            } else if (args.algorithm == "MST") {
                Algorithms::MinimumSpanningTree<std::string> algMST;
                algMST.compute(fms, voc, tree, args);
            } else if (args.algorithm == "NAIVE") {
                Algorithms::NaiveTree<std::string> algNAIVE;
                algNAIVE.compute(fms, voc, tree, args);
            } else {
                throw std::string("Algorithm not implemented");
            }

            std::ofstream ofs;
            ofs.open(args.output);
            tree.store(ofs);
            ofs.close();

            return EXIT_SUCCESS;

        }
    }


}