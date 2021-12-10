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

#include "Geometry.h"

real Geometry::pDistance(const real* vecA, const real* vecB, real p, int dim) {
    real lpnorm = 0;
    for (int i = 0; i < dim; i++) {
        lpnorm += std::pow(std::abs(vecA[i] - vecB[i]), p);
    }
    return std::pow(lpnorm, 1.0 / p);
}

real Geometry::poincareDistance(const real* vecX, const real* vecY, real cel, int dim) {

    std::vector<real> minusX(dim);
    for (int i = 0; i < dim; i++)
        minusX[i] = - *(vecX + i);

    real mxsquare = dot(&minusX[0], &minusX[0], dim);
    real ysquare = dot(vecY, vecY, dim);
    real dotxy = dot(&minusX[0], vecY, dim);

    std::vector<real> mobiusAdd(dim);
    for (int i = 0; i < dim; i++) {
        real leftup = (1 + 2 * cel * dotxy + cel * ysquare) * minusX[i];
        real rightup = (1 - cel * mxsquare) * vecY[i];
        real down = 1 + 2 * cel * dotxy + cel * cel * mxsquare * ysquare;
        mobiusAdd[i] = (leftup + rightup) / down;
    }

    real vA=std::sqrt(dot(&mobiusAdd[0], &mobiusAdd[0], dim));
    real vB = 1 - EPSILON;
    real norm = std::fmin(1 - EPSILON, std::sqrt(dot(&mobiusAdd[0], &mobiusAdd[0], dim)));

    return 2 / std::sqrt(cel) * std::atanh(std::sqrt(cel) * norm);
}

real Geometry::dot(const real* vecA, const real* vecB, int dim) {

    real dotValue = 0;
    for (int i = 0; i < dim; i++)
        dotValue += vecA[i] * vecB[i];
    return dotValue;
}

real Geometry::squaredLorentzianDistance(const real* vecA, const real* vecB, real beta, int dim) {

    std::vector<real> tmpA(dim + 1);
    std::vector<real> tmpB(dim + 1);

    real Asquare = dot(vecA, vecA, dim);
    real Bsquare = dot(vecB, vecB, dim);

    for (int i = 1; i < dim + 1; i++) {
        tmpA[i] = 2 * vecA[i - 1] / (1 - Asquare);
        tmpB[i] = 2 * vecB[i - 1] / (1 - Bsquare);
    }

    tmpA[0] = std::sqrt(dot(&tmpA[1], &tmpA[1], dim) + beta);
    tmpB[0] = std::sqrt(dot(&tmpB[1], &tmpB[1], dim) + beta);

    return std::abs(- 2 * beta - 2 * (-tmpA[0] * tmpB[0] + dot(&tmpA[1], &tmpB[1], dim)));
}

real Geometry::hyperbolic_e(const real* vecX, const real* vecY, int dim) {

    std::vector<real> XminusY(dim);
    for (int i = 0; i < dim; i++)
        XminusY[i] = vecX[i] - vecY[i];

    real Xsquare = dot(vecX, vecX, dim);
    real Ysquare = dot(vecY, vecY, dim);
    real XYdot = dot(vecX, vecY, dim);
    real XminusYnorm = std::sqrt(dot(&XminusY[0], &XminusY[0], dim));

    real num = XYdot * (1 + Xsquare) - Xsquare * (1 + Ysquare);
    real denom = std::sqrt(Xsquare) * XminusYnorm * std::sqrt(1 + Xsquare * Ysquare - 2 * XYdot);

    return std::acos(num / denom);

}

real Geometry::euclidean_e(const real* vecX, const real* vecY, int dim) {
    std::vector<real> XminusY(dim);
    for (int i = 0; i < dim; i++)
        XminusY[i] = vecX[i] - vecY[i];

    real Xsquare = dot(vecX, vecX, dim);
    real Ysquare = dot(vecY, vecY, dim);
    real XminusYsquare = dot(&XminusY[0], &XminusY[0], dim);

    real num = Ysquare - Xsquare - XminusYsquare;
    real denom = 2 * std::sqrt(Xsquare * XminusYsquare);

    return std::acos(num / denom);

}

real Geometry::psi(const real* vecX, real K, int dim) {
    real Xsquare = dot(vecX, vecX, dim);
    return std::asin(K * (1 - Xsquare) / std::sqrt(Xsquare));
}

real Geometry::hyperbolic_energy(const real* vecX, const real* vecY, real K, int dim) {
    return std::fmax(0, hyperbolic_e(vecX, vecY, dim) - psi(vecX, K, dim));
}

real Geometry::euclidean_energy(const real* vecX, const real* vecY, real K, int dim) {
    return std::fmax(0, euclidean_e(vecX, vecY, dim) - psi(vecX, K, dim));
}
