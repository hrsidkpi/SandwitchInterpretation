#pragma once

#include <armadillo>
#include "../AbstractDomain.h"
#include <stdlib.h>
#include "../PolyhedronUnder/PolyhedronUnder.h"


namespace AI {

	class PolyhedronUnderN : public AbstractDomain {
	public:

		~PolyhedronUnderN() {};

		PolyhedronUnderN(PolyhedronUnder initialZonotope, unsigned N) { polyhedrons = std::vector<PolyhedronUnder>(); polyhedrons.push_back(initialZonotope); this->N = N; }
		
		virtual void applyFullyConnectedLayer(arma::mat linear, arma::mat translation) override;
		virtual void applyReLuOnDim(unsigned dim) override;
		virtual void applyReLu() override;
		
		virtual double** getBounds() override;
		virtual AI::AbstractDomainDimBound getBoundsForDim(unsigned dim) override;
		
		virtual unsigned getDimension() override;

		virtual std::string getSizeString() override;


		virtual void print() override;
		void printFull();

	private:


		void splitAllPolyhedrons();
		void joinToN();
		void joinBestTwo();

		std::vector<PolyhedronUnder> polyhedrons;
		unsigned N;
	};



	

}