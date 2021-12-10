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

#include "VecBinder.h"

VecBinder::VecBinder(std::istream& ifs, VecBinder::FILE_FORMAT format) {

    if (VecBinder::FILE_FORMAT::vec == format) {

        std::cout << "Loading vectors" << std::endl;

        std::string data;

        // parse header
        Utils::readTo(ifs, ' ', data);
        this->nvoc = std::stoi(data);

        Utils::readTo(ifs, '\n', data);
        this->dim = std::stoi(data);
        this->vectors = std::vector<real>(nvoc * dim, 0);

        int i = 0;
        
        while (i < nvoc) {
            std::stringstream rowStream;
            std::vector<std::string> row;
            Utils::readTo(ifs, '\n', data);
            rowStream << data;
            rowStream.seekg(0);
            while(rowStream.peek() != EOF) {
                Utils::readTo(rowStream, ' ', data);
                row.push_back(data);
            }

            for (int j = 0; j < this->dim; j++) {
                this->vectors[j + i * this->dim] = std::stof(row[row.size() - this->dim + j]);
            }

            std::stringstream expression;
            for (int j = 0; j < row.size() - this->dim; j++) {
                expression << row[j];
                if (j < row.size() - this->dim - 1) {
                    expression << ' ';
                }
            }
            this->items.insert(expression.str());
            i++;
        }
        std::cout << "End" << std::endl;
    } else {
        throw std::string("Unimplemented format");
    }

}

const real* VecBinder::at(int idx) const{
    assert(idx < this->nvoc);
    return &(this->vectors[idx * this->dim]);
}

int VecBinder::getDim() const {
    return dim;
}

int VecBinder::getVocSize() const {
    return nvoc;
}

int VecBinder::size() const {
    return vectors.size();
}

std::vector<std::string> VecBinder::getElements() const {
    return this->items.toVec();
}

const real *VecBinder::at(const std::string& a) const{
    return this->at(this->items.getIndex(a));
}
