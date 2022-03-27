#pragma once
#include <armadillo>
#include "BasisUtils.h"

inline std::vector<arma::mat> getBasisToImage(arma::mat transformation) {
	std::vector<arma::mat> imageSet;
	for (unsigned i = 0; i < transformation.n_cols; i++) {
		arma::mat v(transformation.n_cols, 1, arma::fill::zeros);
		v(i) = 1;
		imageSet.push_back(transformation * v);
	}
	return convertToBasis(imageSet);
}