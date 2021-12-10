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

#include "Utils.h"

void Utils::readTo(std::istream& input, char delimiter, std::string &buffer) {
    buffer.clear();
    while(input.peek() != EOF && input.peek() != delimiter) {
        buffer.push_back(input.get());
    }
    input.get();
}


template<class T>
int Utils::OrderedSet<T>::insert(const T obj) {

    if (!this->objectSet.count(obj)) {
        this->object2idx.insert({obj, this->size()});
        this->objectSet.insert(obj);
        this->objectVector.push_back(obj);
        this->freqs.push_back(1);
    } else {
        this->freqs[this->object2idx[obj]] += 1;
    }

    return this->getIndex(obj);
}

template<class T>
T Utils::OrderedSet<T>::operator[](const int& idx) const {
    return this->objectVector[idx];
}

template<class T>
int Utils::OrderedSet<T>::getIndex(const T& obj) const {
    return this->object2idx.at(obj);
}

template<class T>
int Utils::OrderedSet<T>::getFreq(const T& obj) const {
    return this->objectSet.count(obj) ? this->freqs[this->getIndex(obj)] : 0;
}

template<class T>
int Utils::OrderedSet<T>::size() const {
    return this->objectVector.size();
}

template<class T>
std::vector<T> Utils::OrderedSet<T>::toVec() const {
    return this->objectVector;
}

template<class T>
Utils::OrderedSet<T>::OrderedSet() {

}

template<class T>
Utils::OrderedSet<T> &Utils::OrderedSet<T>::operator=(const Utils::OrderedSet<T>& other) {
    this->objectVector.clear();
    this->objectSet.clear();
    this->freqs.clear();
    this->object2idx.clear();
    for (int i = 0; i < other.size(); i++) {
        this->insert(other[i]);
    }
    this->freqs = other.freqs;
    return *this;
}

template<class T>
const std::vector<T>& Utils::OrderedSet<T>::vecref() const {
    return this->objectVector;
}

template<class T>
Utils::OrderedSet<T>::OrderedSet(std::vector<T>& objs) {

    for (int i = 0; i < objs.size(); i++)
        this->insert(objs[i]);

}

template<class T>
T Utils::OrderedSet<T>::randomElement(Random& rdhandler) {
    return this->operator[](rdhandler.randint(0, this->size() - 1));
}

template<class T>
int Utils::OrderedSet<T>::pop(const T& obj) {

    int idx = this->getIndex(obj);

    this->object2idx.erase(obj);
    this->freqs.erase(this->freqs.begin() + idx);
    this->objectSet.erase(obj);
    this->objectVector.erase(this->objectVector.begin() + idx);

    return 1;
}

template<class T>
std::set<T> Utils::OrderedSet<T>::getSet() {
    return this->objectSet;
}

template<class T>
void Utils::OrderedSet<T>::clear() {

    this->object2idx.clear();
    this->freqs.clear();
    this->objectSet.clear();
    this->objectVector.clear();

}

template<class T>
Utils::OrderedSet<T>::OrderedSet(const Utils::OrderedSet<T>& other) {
    for (int i = 0; i < other.size(); i++) {
        this->insert(other[i]);
    }
    this->freqs = other.freqs;
}

template<class T>
T Utils::OrderedSet<T>::at(const int &idx) const {
    return this->operator[](idx);
}

template class Utils::OrderedSet<std::string>;