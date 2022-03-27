#include "PolyhedronUnderN.h"


void AI::PolyhedronUnderN::applyFullyConnectedLayer(arma::mat linear, arma::mat translate)
{
	for (auto& z : polyhedrons) {
		z.applyFullyConnectedLayer(linear, translate);
	}
}

void AI::PolyhedronUnderN::applyReLuOnDim(unsigned dim)
{
	for (auto& z : polyhedrons) {
		z.applyReLuOnDim(dim);
	}
}

void AI::PolyhedronUnderN::applyReLu()
{
	
	splitAllPolyhedrons();
	
	std::cout << "After split: " << std::endl;
	printFull();
	
	for (unsigned i = 0; i < getDimension(); ++i) {
		std::cout << "relu on dim " << i << std::endl;
		std::cout << "Current size: " << getSizeString() << std::endl;
		applyReLuOnDim(i);
	}
	joinToN();
}


AI::AbstractDomainDimBound AI::PolyhedronUnderN::getBoundsForDim(unsigned dim)
{
	double lb = 0;
	double ub = 0;
	for (auto z : polyhedrons) {
		AI::AbstractDomainDimBound b = z.getBoundsForDim(dim);
		if (b.lb < lb) lb = b.lb;
		if (b.ub > ub) ub = b.ub;
	}
	return AbstractDomainDimBound(lb, ub);
}

double** AI::PolyhedronUnderN::getBounds()
{
	return nullptr;
}


unsigned AI::PolyhedronUnderN::getDimension()
{
	return polyhedrons[0].getDimension();
}

std::string AI::PolyhedronUnderN::getSizeString()
{
	return std::to_string(polyhedrons.size()) + " PolyhedronUnders. ";
}

void AI::PolyhedronUnderN::print()
{
	std::cout << "PolyhedronUnder" << N << " with " << polyhedrons.size() << " PolyhedronUnders." << std::endl;
}

void AI::PolyhedronUnderN::printFull()
{
	std::cout << "PolyhedronUnder" << N << " with " << polyhedrons.size() << " PolyhedronUnders." << std::endl;
	for (unsigned i = 0; i < polyhedrons.size(); i++) {
		std::cout << "PolyhedronUnder " << i << ":" << std::endl;
		polyhedrons[i].print();
	}
}

void AI::PolyhedronUnderN::splitAllPolyhedrons()
{
	std::vector<AI::PolyhedronUnder> newPolyhedrons;
	for (auto p : polyhedrons) {

		AI::PolyhedronUnder pos = AI::PolyhedronUnder(p);
		pos.meetGt0(0);
		newPolyhedrons.push_back(pos);

		AI::PolyhedronUnder neg = AI::PolyhedronUnder(p);
		neg.meetLt0(0);
		neg.snapTo0(0);
		newPolyhedrons.push_back(neg);
	}
	polyhedrons = newPolyhedrons;
}

void AI::PolyhedronUnderN::joinBestTwo()
{
	unsigned iBest = -1;
	unsigned jBest = -1;
	double bestDist = -1;
	for (unsigned i = 0; i < polyhedrons.size() - 1; i++) {
		for (unsigned j = i + 1; j < polyhedrons.size(); j++) {
			double dist = arma::norm(polyhedrons[i].approximateCenter() - polyhedrons[j].approximateCenter());
			if (bestDist == -1 || dist < bestDist) {
				iBest = i;
				jBest = j;
				bestDist = dist;
			}
		}
	}

	//TODO maybe choose the smaller one of the two polyhedrons? Right now I'm taking an arbitrary one
	polyhedrons.erase(polyhedrons.begin() + jBest);

}

void AI::PolyhedronUnderN::joinToN()
{
	while (polyhedrons.size() > N) {
		std::cout << "PolyhedronUnder" << N << " has " << polyhedrons.size() << " polyhedrons. Joining the closest two." << std::endl;
		joinBestTwo();
	}
}
