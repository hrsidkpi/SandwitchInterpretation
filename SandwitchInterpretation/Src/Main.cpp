
#include "lp_lib.h"
#include "AbstractDomain/Zonotope/Zonotope.h"
#include <armadillo>

int main()
{
    std::vector<arma::mat> generators;

    arma::mat g1 = { 1, 0, -1 };
    generators.push_back(g1);

    arma::mat g2 = { 1, 1, 2 };
    generators.push_back(g2);

    arma::mat g3 = { 1, 3, 0 };
    generators.push_back(g3);

    

    arma::mat bias = { {0, 1, 2} };

    AI::Zonotope z = AI::Zonotope(generators, bias);

    std::cout << "==== start: ====" << std::endl;
    z.print();

    z.applyReLuOnDim(0);

    std::cout << "==== after relu: ====" << std::endl;
    z.print();

}