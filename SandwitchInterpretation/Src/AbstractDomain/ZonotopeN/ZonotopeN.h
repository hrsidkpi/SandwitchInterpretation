#pragma once

#include <armadillo>
#include "../AbstractDomain.h"
#include <stdlib.h>
#include "../Zonotope/Zonotope.h"
#include "../Zonotope/ZonotopeOperations.h"


namespace AI {

	class ZonotopeN : public AbstractDomain {
	public:

		~ZonotopeN() {};

		ZonotopeN(Zonotope initialZonotope, unsigned N) { zonotopes = std::vector<Zonotope>(); zonotopes.push_back(initialZonotope); this->N = N; }
		
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


		void splitAllZonotopes();
		void joinToN();
		void joinBestTwo();

		std::vector<Zonotope> zonotopes;
		unsigned N;
	};



	

}