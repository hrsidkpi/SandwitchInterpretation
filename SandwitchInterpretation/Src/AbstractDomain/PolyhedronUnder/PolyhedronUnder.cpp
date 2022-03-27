#include "PolyhedronUnder.h"
#include "../../MathUtils/LinearAlgebraUtils/BasisUtils.h"
#include "../../MathUtils/LinearAlgebraUtils/TransformationUtils.h"

void AI::PolyhedronUnder::applyFullyConnectedLayer(arma::mat linear, arma::mat translation)
{

	linear = linear.t();
	std::vector<AI::Halfspace*> new_halfspaces{};
	for (auto& h : halfspaces) {
		bool has_sol = h->applyAffineTransformation(linear, translation);
		if (has_sol) 
			new_halfspaces.push_back(h);
		else 
			delete h;
	}
	
	std::vector<arma::mat> imageBasis = complementoryBasis(getBasisToImage(linear));
	for (auto b : imageBasis) {
		AI::Halfspace *gt = new AI::Halfspace(dimension, b, -arma::dot(b, translation));

		arma::mat bMinus = -b;
		AI::Halfspace *lt = new AI::Halfspace(dimension, bMinus, arma::dot(b, translation));
		new_halfspaces.push_back(gt);
		new_halfspaces.push_back(lt);
	}

	dimension = translation.n_cols;
	halfspaces = new_halfspaces;

}

void AI::PolyhedronUnder::applyReLuOnDim(unsigned dim)
{
	arma::mat weights(1, dimension, arma::fill::zeros);
	for (unsigned dd = 0; dd < dimension; ++dd) {
		if (dd == dim) weights[dd] = -1;
		else weights[dd] = 0;
	}
	AI::Halfspace* h = new AI::Halfspace(dimension, weights, 0);
	halfspaces.push_back(h);
}

double** AI::PolyhedronUnder::getBounds()
{
	double** res = new double* [dimension];
	for (unsigned d = 0; d < dimension; d++) {
		res[d] = new double[2];
		auto dRes = getBoundsForDim(d);
		res[d][0] = dRes.lb;
		res[d][1] = dRes.ub;
	}
	return res;
}

AI::AbstractDomainDimBound AI::PolyhedronUnder::getBoundsForDim(unsigned dim)
{
	LPSolver solver(dimension);
	for (unsigned d = 0; d < dimension; d++) {
		solver.addVariableName("x_" + std::to_string(d));
	}
	
	for (auto h : halfspaces) {
		auto row = solver.getRowBuilder();
		for (unsigned d = 0; d < dimension; ++d) {
			row.setColumnCoef("x_" + std::to_string(d), h->weights[d]);
		}
		row.setConstraintType(AI::LPConstraintType::LESS_EQUALS);
		row.setRightHandSide(-h->bias);
		row.build();
	}
	
	auto obj = solver.getObjectiveBuilder();
	obj.setColumnCoef("x_" + std::to_string(dim), 1);
	obj.setObjectiveType(AI::LPObjectiveType::MAXIMIZE);
	obj.build();
	solver.includeVarInResult("x_" + std::to_string(dim));

	auto res = solver.solveLP();
	double ub = res.objectiveValue;

	obj.setObjectiveTypeAndRebuild(AI::LPObjectiveType::MINIMIZE);
	res = solver.solveLP();
	double lb = res.objectiveValue;

	return AI::AbstractDomainDimBound(lb, ub);
}

unsigned AI::PolyhedronUnder::getDimension()
{
	return dimension;
}

std::string AI::PolyhedronUnder::getSizeString()
{
	return std::to_string(halfspaces.size()) + " halfspaces.";
}

void AI::PolyhedronUnder::print()
{
	std::cout << "PolyhedronUnder with " << halfspaces.size() << " halfspaces:" << std::endl;
	for (auto h : halfspaces) {
		h->print();
		std::cout << std::endl;
	}
}

std::vector<AI::Halfspace*> AI::PolyhedronUnder::getHalfspaces()
{
	return std::vector<AI::Halfspace*>();
}

void AI::PolyhedronUnder::meetGt0(unsigned dim)
{
	//Apply relu, because relu on polyhedron under estimator is simply meet with >= 0
	applyReLuOnDim(dim);
}

void AI::PolyhedronUnder::meetLt0(unsigned dim)
{
	arma::mat weights(1, dimension, arma::fill::zeros);
	for (unsigned dd = 0; dd < dimension; ++dd) {
		if (dd == dim) weights[dd] = 1;
		else weights[dd] = 0;
	}
	AI::Halfspace* h = new AI::Halfspace(dimension, weights, 0);
	halfspaces.push_back(h);
}

void AI::PolyhedronUnder::snapTo0(unsigned dim)
{
	arma::mat weights(dimension, dimension, arma::fill::zeros);
	for (unsigned dd = 0; dd < dimension; ++dd) {
		if (dd == dim) weights(dd,dd) = 0;
		else weights(dd,dd) = 1;
	}
	arma::mat translation(1, dimension, arma::fill::zeros);
	applyFullyConnectedLayer(weights, translation);
}

arma::mat AI::PolyhedronUnder::approximateCenter()
{
	arma::mat res(1, dimension, arma::fill::zeros);
	for (unsigned d = 0; d < dimension; d++) {
		auto b = getBoundsForDim(d);
		res(d) = (b.lb + b.ub) / 2;
	}
	return res;
}
