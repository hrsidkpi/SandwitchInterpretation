
#include "lp_lib.h"
#include "AbstractDomain/Zonotope/Zonotope.h"
#include "AbstractDomain/ZonotopeN/ZonotopeN.h"
#include <armadillo>
#include "MathUtils/RandomizedTesting/ZonotopeRandomizer.h"
#include "AbstractDomain/AbstractDomainBuilder.h"
#include "AbstractDomain/ComplexAbstractDomainBuilder.h"
#include "AbstractDomain/PolyhedronUnder/PolyhedronUnder.h"
#include "AbstractDomain/PolyhedronUnderN/PolyhedronUnderN.h"


int main()
{
	std::vector<AI::Halfspace*> halfspaces;

	AI::Halfspace* hs = new AI::Halfspace(2, arma::mat {{0,1}}, 1);
	halfspaces.push_back(hs);

	AI::PolyhedronUnder p(halfspaces);

	AI::PolyhedronUnderN pn(p, 1);
	pn.print();
	std::cout << "\n\n" << std::endl;

	pn.applyReLu();
	std::cout << "\n\n" << std::endl;

	pn.printFull();
}