#pragma once
#include <armadillo>
#include "VectorsUtils.h"

inline std::vector<arma::mat> convertToBasis(std::vector<arma::mat> set) {
	std::vector<arma::mat> res;

	for (unsigned i = 0; i < set.size(); i++) {
		arma::mat v = set[i];
		for (unsigned j = 0; j < res.size(); j++) {
			v -= vectorProjection(set[i], res[j]);
		}
		if(arma::dot(v, v) != 0)
			res.push_back(v);
	}

	return res;
}

inline arma::mat findOrthogonalVector(std::vector<arma::mat> basis) {
	arma::mat A(basis.size(), basis[0].size(), arma::fill::zeros);
	for (unsigned i = 0; i < basis.size(); i++) {
		for (unsigned j = 0; j < basis[0].size(); j++) {
			A(i, j) = basis[i][j];
		}
	}
	
	arma::mat nullSpaceBasis;
	arma::null(nullSpaceBasis, A);
	return nullSpaceBasis.col(0);
}

inline std::vector<arma::mat> complementoryBasis(std::vector<arma::mat> basis) {
	std::vector<arma::mat> res;
	while (basis.size() < basis[0].size()) {
		arma::mat v = findOrthogonalVector(basis);
		basis.push_back(v);
		res.push_back(v);
	}
	return res;
}
