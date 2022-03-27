#pragma once
#include <armadillo>


namespace AI {

	struct AbstractDomainDimBound {

		AbstractDomainDimBound(double lb, double ub) : ub(ub), lb(lb) {}

		double ub;
		double lb;
	};

	class AbstractDomain
	{
	public:

		virtual ~AbstractDomain() {};

		virtual void applyFullyConnectedLayer(arma::mat linear, arma::mat translation) = 0;
		virtual void applyReLuOnDim(unsigned dim) = 0;
		
		virtual double** getBounds() = 0;

		virtual AbstractDomainDimBound getBoundsForDim(unsigned dim) = 0;

		virtual void print() = 0;

		virtual unsigned getDimension() = 0;

		//Return a string representation of the size of the current domain (for example, number of generators in a zonotope)
		virtual std::string getSizeString() = 0; 

		virtual void applyReLu() {
			for (unsigned i = 0; i < getDimension(); ++i) {
				std::cout << "relu on dim " << i << std::endl;
				std::cout << "Current size: " << getSizeString() << std::endl;
				applyReLuOnDim(i);
			}
		}

		void printBounds() {
			std::cout << "Bounds of abstract domain:" << std::endl;
			double **bounds = getBounds();
			for(unsigned d = 0; d < getDimension(); ++d) {
				std::cout << bounds[d][0] << " <= x_" << d << " <= " << bounds[d][1] << std::endl; 
				delete[] bounds[d];
			}
			delete[] bounds;
			
		}

	};
}
