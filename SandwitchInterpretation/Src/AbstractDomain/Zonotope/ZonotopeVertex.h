#pragma once
#include <armadillo>

namespace AI {

	struct ZonotopeVertex {
		arma::mat vertex;
		arma::mat generatorCoefs;
	
		ZonotopeVertex(arma::mat vertex, arma::mat coefs) : vertex(vertex), generatorCoefs(coefs) {};
	
	};

}