#pragma once
#include <armadillo>
#include "../AbstractDomain.h"
#include <stdlib.h>
#include "Halfspace.h"
#include <vector>
#include "../../MathUtils/LPSolver/LPSolver.h"


namespace AI {

	class PolyhedronUnder : public AbstractDomain {
	public:

		~PolyhedronUnder() {};

		PolyhedronUnder(std::vector<AI::Halfspace*> halfspaces) : halfspaces(halfspaces), dimension(halfspaces[0]->dim) {}
		PolyhedronUnder(const AI::PolyhedronUnder& other) : halfspaces(other.halfspaces), dimension(other.dimension) {}
		
		virtual void applyFullyConnectedLayer(arma::mat linear, arma::mat translation) override;
		virtual void applyReLuOnDim(unsigned dim) override;
		
		virtual double** getBounds() override;
		virtual AbstractDomainDimBound getBoundsForDim(unsigned dim) override;
		
		virtual unsigned getDimension() override;

		virtual std::string getSizeString() override;

		virtual void print() override;

		std::vector<AI::Halfspace*> getHalfspaces();

		void meetGt0(unsigned dim);
		void meetLt0(unsigned dim);
		void snapTo0(unsigned dim);

		arma::mat approximateCenter();

	private:

		std::vector<AI::Halfspace*> halfspaces;
		unsigned dimension;

	};



	

}