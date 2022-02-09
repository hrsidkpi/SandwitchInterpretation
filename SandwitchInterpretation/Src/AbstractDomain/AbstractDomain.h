#pragma once
#include <armadillo>


namespace AI {
	class AbstractDomain
	{
	public:
		virtual void applyFullyConnectedLayer(arma::mat linear, arma::mat translation) = 0;
		virtual void applyReLuOnDim(unsigned dim) = 0;
		
		virtual void joinWith(AbstractDomain& other) = 0;

		virtual double** getBounds() = 0;

		void applyReLu();

	private:
		unsigned dim;

	};
}
