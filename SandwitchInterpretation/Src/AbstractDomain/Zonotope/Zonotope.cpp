#include "Zonotope.h"
#include "../../MathUtils/LinearAlgebraUtils/VectorsUtils.h"


void AI::Zonotope::applyFullyConnectedLayer(arma::mat linear, arma::mat translate)
{
	bias = linear * bias + translate;
	for (unsigned i = 0; i < generators.size(); i++) {
		generators[i] = linear * generators[i];
	}
}

void AI::Zonotope::applyReLuOnDim(unsigned dim)
{
	AI::AbstractDomainDimBound bounds = getBoundsForDim(dim);
	double lower = bounds.lb;
	double upper = bounds.ub;

	if (lower > 0) return;
	if (upper < 0) {
		bias[dim] = 0;
		for (unsigned g = 0; g < generators.size(); ++g) {
			generators[g][dim] = 0;
		}
	}

	double lambda = upper / (upper - lower);
	double u = -(upper * lower) / (2 * (upper - lower));

	bias[dim] = lambda * bias[dim] + u;
	for (unsigned g = 0; g < generators.size(); ++g) {
		generators[g][dim] = lambda * generators[g][dim];
	}

	arma::mat new_g(bias.size(), 1, arma::fill::zeros);
	new_g[dim] = u;
	generators.push_back(new_g);
}

double** AI::Zonotope::getBounds()
{
	double** bounds = new double* [bias.size()];
	for (unsigned d = 0; d < bias.size(); d++) {
		bounds[d] = new double[2];
		double upper = bias(d);
		double lower = bias(d);
		for (arma::mat g : generators) {
			upper += abs(g(d));
			lower -= abs(g(d));
		}
		bounds[d][0] = lower;
		bounds[d][1] = upper;
	}
	return bounds;
}

AI::AbstractDomainDimBound AI::Zonotope::getBoundsForDim(unsigned dim)
{

	double upper = bias(dim);
	double lower = bias(dim);
	for (arma::mat g : generators) {
		upper += abs(g(dim));
		lower -= abs(g(dim));
	}
	return AbstractDomainDimBound(lower, upper);
}


void AI::Zonotope::changeEpsilonBounds(unsigned generatorIndex, double lower, double upper)
{
	if (lower > upper) {
		throw std::runtime_error("Tryng to change epsilon for generator to invalid bounds");
	}
	if (lower == upper) {
		bias += lower * generators[generatorIndex];
		generators.erase(generators.begin() + generatorIndex);
	}
	else {
		bias += (lower + upper) / 2 * generators[generatorIndex];
		generators[generatorIndex] *= (upper - lower) / 2;
	}
}

bool AI::Zonotope::isVertex(arma::mat vert)
{
	arma::mat dir = vert - bias;
	LPSolver solver = LPSolver(generators.size() + 1);

	for (unsigned gIndex = 0; gIndex < generators.size(); ++gIndex)
		solver.addVariableName("g" + std::to_string(gIndex));
	solver.addVariableName("dist");

	for (unsigned dim = 0; dim < bias.size(); dim++) {
		LPRowBuilder row = solver.getRowBuilder();
		row.setColumnCoef("dist", dir(dim));
		for (unsigned gIndex = 0; gIndex < generators.size(); ++gIndex) {
			double gCoef = -generators[gIndex](dim);
			std::string varName = "g" + std::to_string(gIndex);
			row.setColumnCoef(varName, gCoef);
		}
		row.setRightHandSide(0);
		row.setConstraintType(AI::EQUALS);
		row.build();
	}

	for (unsigned gIndex = 0; gIndex < generators.size(); gIndex++) {
		solver.boundVariable("g" + std::to_string(gIndex), -1, 1);
	}

	AI::LPObjectiveBuilder obj = solver.getObjectiveBuilder();
	obj.setColumnCoef("dist", 1);
	obj.setObjectiveType(AI::MAXIMIZE);
	obj.build();

	solver.includeVarInResult("dist");
	AI::LPResult res = solver.solveLP();

	double MAX_ALLOWED_DIFF = 0.0001;
	if (abs(res.vars[0].val - 1) < MAX_ALLOWED_DIFF)
		return true;
	return false;

}

std::vector<arma::mat> AI::Zonotope::getAllPossibleEpsilonValues(unsigned gIndex)
{
	if (gIndex == generators.size()) {
		std::vector<arma::mat> res;
		arma::mat m(generators.size(), 1, arma::fill::zeros);
		res.push_back(m);
		return res;
	}
	else {
		std::vector<arma::mat> res;
		for (arma::mat m : getAllPossibleEpsilonValues(gIndex + 1)) {
			arma::mat m1(m);
			m1(gIndex) = 1;
			res.push_back(m1);

			arma::mat m2(m);
			m2(gIndex) = -1;
			res.push_back(m2);
		}
		return res;
	}
}

std::vector<AI::ZonotopeVertex> AI::Zonotope::getVertices()
{
	std::vector<arma::mat> epsilonsValues = getAllPossibleEpsilonValues(0);
	std::vector<AI::ZonotopeVertex> res;
	for (arma::mat epsilons : epsilonsValues) {
		arma::mat vert = bias;
		for (unsigned gIndex = 0; gIndex < generators.size(); ++gIndex)
			vert += epsilons(gIndex) * generators[gIndex];
		if (isVertex(vert))
			res.push_back(ZonotopeVertex(vert, epsilons));
	}
	return res;
}

unsigned AI::Zonotope::getDimension()
{
	return bias.size();
}

std::string AI::Zonotope::getSizeString()
{
	std::string s = std::to_string(generators.size());
	s += " generators";
	return s;
}


void AI::Zonotope::print()
{
	std::cout << "Zonotope of dimension " << bias.size() << std::endl;
	std::cout << "bias: " << bias << std::endl;
	std::cout << "generators: " << std::endl;
	for (arma::mat g : generators)
		std::cout << g << std::endl;
}

std::vector<arma::mat> AI::Zonotope::getGenerators()
{
	return generators;
}

arma::mat AI::Zonotope::getBias()
{
	return bias;
}
