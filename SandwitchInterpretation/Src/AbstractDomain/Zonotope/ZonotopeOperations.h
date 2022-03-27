#pragma once
#include "Zonotope.h"
#include <armadillo>

namespace AI {

	class Zonotope;

	Zonotope createBoundingZonotope(std::vector<arma::mat> points, std::vector<arma::mat> generators);
	Zonotope joinZonotopes(Zonotope z1, Zonotope z2);

	std::pair<Zonotope, Zonotope> splitZonotope(Zonotope z);

}