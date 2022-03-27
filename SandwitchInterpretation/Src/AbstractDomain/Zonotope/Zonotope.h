#pragma once
#include <armadillo>
#include "../AbstractDomain.h"
#include "ZonotopeVertex.h"
#include <stdlib.h>
#include "../../MathUtils/LPSolver/LPSolver.h"
#include "ZonotopeOperations.h"


namespace AI {

	enum SearchSign { POSITIVE, NEGATIVE };

	class Zonotope : public AbstractDomain {
	public:

		~Zonotope() {};

		Zonotope(std::vector<arma::mat>& generators, arma::mat& bias) : generators(generators), bias(bias) {}
		Zonotope(const AI::Zonotope& other) : generators(other.generators), bias(other.bias) {}
		
		virtual void applyFullyConnectedLayer(arma::mat linear, arma::mat translation) override;
		virtual void applyReLuOnDim(unsigned dim) override;
		
		virtual double** getBounds() override;
		virtual AbstractDomainDimBound getBoundsForDim(unsigned dim) override;
		
		virtual unsigned getDimension() override;

		std::vector<ZonotopeVertex> getVertices();

		virtual std::string getSizeString() override;

		void changeEpsilonBounds(unsigned generatorIndex, double lower, double upper);

		virtual void print() override;

		std::vector<arma::mat> getGenerators();
		arma::mat getBias();

	private:
		std::vector<arma::mat> generators;
		arma::mat bias;

		bool isVertex(arma::mat vert);

		std::vector<arma::mat> getAllPossibleEpsilonValues(unsigned dim);

	};



	

}