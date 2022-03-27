#pragma once
#include <armadillo>

inline bool AreVectorsParallel(arma::mat v1, arma::mat v2) {
	if (v1.n_rows != v2.n_rows || v1.n_cols != v2.n_cols) return false;


	//Find the ratio
	double ratio = 0;
	for (unsigned i = 0; i < v1.n_elem; i++) {
		double d1 = v1(i);
		double d2 = v2(i);
		if (d1 == 0 && d2 != 0) return false;
		if (d1 != 0 && d2 == 0) return false;
		if (d1 == 0 && d2 == 0) continue;
		ratio = d1 / d2;
		break;
	}
	
	//Make sure everything is by this ratio
	for (unsigned i = 0; i < v1.n_elem; i++) {
		double d1 = v1(i);
		double d2 = v2(i);
		if (d1 == 0 && d2 != 0) return false;
		if (d1 != 0 && d2 == 0) return false;
		if (d1 == 0 && d2 == 0) continue;
		if (d1 / d2 != ratio) return false;
	}
	return true;
}

inline arma::mat vectorProjection(arma::mat v, arma::mat on) {
	return (arma::dot(on, v) / arma::dot(on, on)) * on;
}



