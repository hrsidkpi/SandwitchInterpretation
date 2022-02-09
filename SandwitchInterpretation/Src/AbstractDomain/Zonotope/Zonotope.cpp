#include "Zonotope.h"
#include <stdlib.h>
#include "../../MathUtils/LPSolver/LPSolver.h"
#include "ZonotopeOperations.h"

void AI::Zonotope::applyFullyConnectedLayer(arma::mat linear, arma::mat translate)
{
	bias = linear * bias + translate;
	for (int i = 0; i < generators.size(); i++) {
		generators[i] = linear * generators[i];
	}
}

void AI::Zonotope::applyReLuOnDim(unsigned dim)
{
	AI::Zonotope z_neg = *this;

	meetGt0(dim);
	z_neg.meetLt0(dim);
	z_neg.snapTo0(dim);

	joinWith(z_neg);

}

void AI::Zonotope::joinWith(AI::AbstractDomain& other)
{
	AI::Zonotope *z= (AI::Zonotope *) &other;
	
	std::vector<AI::ZonotopeVertex> thisVerts = getVertices();
	std::vector<AI::ZonotopeVertex> otherVerts = z->getVertices();


	std::vector<arma::mat> points;
	for (AI::ZonotopeVertex v : thisVerts)
		points.push_back(v.vertex);
	for (AI::ZonotopeVertex v : otherVerts)
		points.push_back(v.vertex);

	arma::mat centers_vec = z->bias - bias;

	std::vector<arma::mat> vectors = generators;

	if(arma::norm(centers_vec) > 0.001)
		vectors.push_back(centers_vec);

	AI::Zonotope res = AI::createBoundingZonotope(points, vectors);
	generators = res.generators;
	bias = res.bias;
}

double** AI::Zonotope::getBounds()
{
	double** bounds = new double* [bias.size()];
	for (int d = 0; d < bias.size(); d++) {
		bounds[d] = new double[2];
		double upper = 0;
		double lower = 0;
		for (arma::mat g : generators) {
			upper += abs(g(d));
			lower -= abs(g(d));
		}
	}
	return bounds;

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

bool AI::Zonotope::getEdgiestPoint(unsigned dim, unsigned generatorIndex, AI::BoundType boundType, AI::SearchSign sign, AI::ZonotopeVertex* dest)
{
	std::vector<AI::ZonotopeVertex> allPoints = getVertices();

	AI::ZonotopeVertex* edgiest = nullptr;
	double edgiestVal = 0;

	for (AI::ZonotopeVertex p : allPoints) {
		if (sign == AI::POSITIVE && p.vertex(dim) <= 0) continue;
		if (sign == AI::NEGATIVE && p.vertex(dim) >= 0) continue;
		if (boundType == AI::LOWER && p.generatorCoefs(generatorIndex) >= 0) continue;
		if (boundType == AI::UPPER && p.generatorCoefs(generatorIndex) <= 0) continue;

		if (edgiest == nullptr) {
			edgiest = new AI::ZonotopeVertex(p);
			edgiestVal = p.vertex[dim];
		}

		else if (sign == AI::POSITIVE && p.vertex[dim] < edgiestVal) {
			edgiest = new AI::ZonotopeVertex(p);
			edgiestVal = p.vertex[dim];
		}
		else if (sign == AI::NEGATIVE && p.vertex[dim] > edgiestVal) {
			edgiest = new AI::ZonotopeVertex(p);
			edgiestVal = p.vertex[dim];
		}
	}

	if (edgiest == nullptr) return false;

	*dest = *edgiest;
	return true;
}

void AI::Zonotope::meetGt0(unsigned dim)
{
	std::vector<AI::ZonotopeVertex> verts = getVertices();
	for (int j = 0; j < generators.size(); j++) {
		bool canChangeUb = true;
		bool canChangeLb = true;
		for (AI::ZonotopeVertex v : verts) {
			if (v.vertex[dim] >= 0 && v.generatorCoefs(j) > 0) canChangeUb = false;
			if (v.vertex[dim] >= 0 && v.generatorCoefs(j) < 0) canChangeLb = false;
		}
		
		if (!canChangeLb && !canChangeUb) 
			continue;
		
		if (canChangeLb) {
			AI::ZonotopeVertex* closest = new AI::ZonotopeVertex(arma::mat(1, bias.size(), arma::fill::zeros), arma::mat(1, generators.size(), arma::fill::zeros));
			if (!getEdgiestPoint(dim, j, AI::LOWER, AI::NEGATIVE, closest)) 
				continue;

			double newLb = -1 - closest->vertex(dim) * generators[j](dim);
			if (newLb <= 1) {
				changeEpsilonBounds(j, newLb, 1);
				return;
			}
		}

		if (canChangeUb) {
			AI::ZonotopeVertex* closest = new AI::ZonotopeVertex(arma::mat(1, bias.size(), arma::fill::zeros), arma::mat(1, generators.size(), arma::fill::zeros));
			if (!getEdgiestPoint(dim, j, AI::UPPER, AI::NEGATIVE, closest)) 
				continue;
			double newUb = 1 - closest->vertex(dim) / generators[j](dim);
			if (newUb <= 1) {
				changeEpsilonBounds(j, -1, newUb);
				return;
			}

		}
	}
}

void AI::Zonotope::meetLt0(unsigned dim)
{
	for (int j = 0; j < generators.size(); j++) {
		std::vector<AI::ZonotopeVertex> verts = getVertices();
		bool canChangeUb = true;
		bool canChangeLb = true;
		for (AI::ZonotopeVertex v : verts) {
			if (v.vertex[dim] <= 0 && v.generatorCoefs(j) > 0) canChangeUb = false;
			if (v.vertex[dim] <= 0 && v.generatorCoefs(j) < 0) canChangeLb = false;
			if (!canChangeLb && !canChangeUb) continue;
		}
		if (canChangeLb) {
			AI::ZonotopeVertex* closest = new AI::ZonotopeVertex(arma::mat(1, bias.size(), arma::fill::zeros), arma::mat(1, generators.size(), arma::fill::zeros));
			if (!getEdgiestPoint(dim, j, AI::LOWER, AI::POSITIVE, closest)) continue;

			double newLb = -1 - closest->vertex(dim) / generators[j](dim);
			if (newLb <= 1) {
				changeEpsilonBounds(j, newLb, 1);
				return;
			}
		}

		if (canChangeUb) {
			AI::ZonotopeVertex* closest = new AI::ZonotopeVertex(arma::mat(1, bias.size(), arma::fill::zeros), arma::mat(1, generators.size(), arma::fill::zeros));
			if (!getEdgiestPoint(dim, j, AI::UPPER, AI::POSITIVE, closest)) continue;
			double newUb = 1 - closest->vertex(dim) / generators[j](dim);
			if (newUb <= 1) {
				changeEpsilonBounds(j, -1, newUb);
				return;
			}
		}

	}
}

void AI::Zonotope::snapTo0(unsigned dim)
{
	for (arma::mat& g : generators) {
		g(dim) = 0;
	}
	bias(dim) = 0;
}

bool AI::Zonotope::isVertex(arma::mat vert)
{
	arma::mat dir = vert - bias;
	LPSolver solver = LPSolver(generators.size() + 1);

	for (int gIndex = 0; gIndex < generators.size(); ++gIndex)
		solver.addVariableName("g" + std::to_string(gIndex));
	solver.addVariableName("dist");

	for (int dim = 0; dim < bias.size(); dim++) {
		LPRowBuilder row = solver.getRowBuilder();
		row.setColumnCoef("dist", dir(dim));
		for (int gIndex = 0; gIndex < generators.size(); ++gIndex) {
			double gCoef = -generators[gIndex](dim);
			std::string varName = "g" + std::to_string(gIndex);
			row.setColumnCoef(varName, gCoef);
		}
		row.setRightHandSide(0);
		row.setConstraintType(AI::EQUALS);
		row.build();
	}

	for (int gIndex = 0; gIndex < generators.size(); gIndex++) {
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
		arma::mat m(1, generators.size(), arma::fill::zeros);
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
		for (int gIndex = 0; gIndex < generators.size(); ++gIndex)
			vert += epsilons(gIndex) * generators[gIndex];
		if (isVertex(vert))
			res.push_back(ZonotopeVertex(vert, epsilons));
	}
	return res;
}

void AI::Zonotope::print()
{
	std::cout << "Zonotope of dimension " << bias.size() << std::endl;
	std::cout << "bias: " << bias << std::endl;
	std::cout << "generators: " << std::endl;
	for (arma::mat g : generators)
		std::cout << "\t" << g << std::endl;
}
