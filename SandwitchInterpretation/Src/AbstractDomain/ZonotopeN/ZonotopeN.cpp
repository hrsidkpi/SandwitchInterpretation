#include "ZonotopeN.h"


void AI::ZonotopeN::applyFullyConnectedLayer(arma::mat linear, arma::mat translate)
{
	for (auto& z : zonotopes) {
		z.applyFullyConnectedLayer(linear, translate);
	}
}

void AI::ZonotopeN::applyReLuOnDim(unsigned dim)
{
	for (auto& z : zonotopes) {
		z.applyReLuOnDim(dim);
	}
}

void AI::ZonotopeN::applyReLu()
{
	
	splitAllZonotopes();
	
	std::cout << "After split: " << std::endl;
	printFull();
	
	for (unsigned i = 0; i < getDimension(); ++i) {
		std::cout << "relu on dim " << i << std::endl;
		std::cout << "Current size: " << getSizeString() << std::endl;
		applyReLuOnDim(i);
	}
	joinToN();
}


AI::AbstractDomainDimBound AI::ZonotopeN::getBoundsForDim(unsigned dim)
{
	double lb = 0;
	double ub = 0;
	for (auto z : zonotopes) {
		AI::AbstractDomainDimBound b = z.getBoundsForDim(dim);
		if (b.lb < lb) lb = b.lb;
		if (b.ub > ub) ub = b.ub;
	}
	return AbstractDomainDimBound(lb, ub);
}

double** AI::ZonotopeN::getBounds()
{
	return nullptr;
}


unsigned AI::ZonotopeN::getDimension()
{
	return zonotopes[0].getDimension();
}

std::string AI::ZonotopeN::getSizeString()
{
	return std::to_string(zonotopes.size()) + " zonotopes. ";
}

void AI::ZonotopeN::print()
{
	std::cout << "Zonotope" << N << " with " << zonotopes.size() << " zonotopes." << std::endl;
}

void AI::ZonotopeN::printFull()
{
	std::cout << "Zonotope" << N << " with " << zonotopes.size() << " zonotopes." << std::endl;
	for (unsigned i = 0; i < zonotopes.size(); i++) {
		std::cout << "Zonotope " << i << ":" << std::endl;
		zonotopes[i].print();
	}
}

void AI::ZonotopeN::splitAllZonotopes()
{
	std::vector<AI::Zonotope> newZonotopes;
	for (auto z : zonotopes) {
		std::pair<AI::Zonotope, AI::Zonotope> split = AI::splitZonotope(z);
		newZonotopes.push_back(split.first);
		newZonotopes.push_back(split.second);
	}
	zonotopes = newZonotopes;
}

void AI::ZonotopeN::joinBestTwo()
{
	unsigned iBest = -1;
	unsigned jBest = -1;
	double bestDist = -1;
	for (unsigned i = 0; i < zonotopes.size() - 1; i++) {
		for (unsigned j = i + 1; j < zonotopes.size(); j++) {
			double dist = arma::norm(zonotopes[i].getBias() - zonotopes[j].getBias());
			if (bestDist == -1 || dist < bestDist) {
				iBest = i;
				jBest = j;
				bestDist = dist;
			}
		}
	}

	AI::Zonotope join = AI::joinZonotopes(zonotopes[iBest], zonotopes[jBest]);

	if (iBest > jBest) {
		zonotopes.erase(zonotopes.begin() + iBest);
		zonotopes.erase(zonotopes.begin() + jBest);
	}
	else {
		zonotopes.erase(zonotopes.begin() + jBest);
		zonotopes.erase(zonotopes.begin() + iBest);
	}
	zonotopes.push_back(join);
}

void AI::ZonotopeN::joinToN()
{
	while (zonotopes.size() > N) {
		std::cout << "Zonotope" << N << " has " << zonotopes.size() << " zonotopes. Joining the closest two." << std::endl;
		joinBestTwo();
	}
}
