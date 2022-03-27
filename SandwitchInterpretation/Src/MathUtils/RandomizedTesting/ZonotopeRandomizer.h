#pragma once
#include "../../AbstractDomain/Zonotope/Zonotope.h"
#include <stdlib.h>

AI::Zonotope GetRandomZonotope(unsigned dim, unsigned generatorCount) {
	std::vector<arma::mat> generators;

	for (unsigned g = 0; g < generatorCount; g++) {
		arma::mat generator = arma::mat(dim, 1, arma::fill::zeros);
		for (unsigned d = 0; d < dim; d++) {
			generator(d) = std::rand() % 10 - 5;
		}
		generators.push_back(generator);
	}

	arma::mat bias = arma::mat(dim, 1, arma::fill::zeros);
	for (unsigned d = 0; d < dim; d++) {
		//bias(d) = std::rand() % 10 - 5;
	}

	AI::Zonotope z = AI::Zonotope(generators, bias);
	return z;
}