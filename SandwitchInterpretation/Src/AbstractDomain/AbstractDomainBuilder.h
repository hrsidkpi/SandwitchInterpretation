#pragma once
#include "AbstractDomain.h"
#include <armadillo>
#include "Zonotope/Zonotope.h"

namespace AI {


    enum class AbstractDomainType { ZONOTOPE_DOMAIN, BOX_DOMAIN, POLYHEDRON_DOMAIN, NONE_DOMAIN };

    class AbstractDomainBuilder {
    public:

        AbstractDomainBuilder(AbstractDomainType type) : _type(type) {}

        AbstractDomain *build(double **bounds, unsigned dim) {
            if(_type == AbstractDomainType::ZONOTOPE_DOMAIN) {
                arma::mat bias(dim, 1, arma::fill::zeros);
                std::vector<arma::mat> generators;
                for(unsigned d = 0; d < dim; d++) {
                    double lb = bounds[d][0];
                    double ub = bounds[d][1];
                    bias(d) = (ub + lb) / 2;
                    arma::mat g(dim, 1, arma::fill::zeros);
                    g(d) = (ub - lb) / 2;
                    generators.push_back(g);
                }
                return new Zonotope(generators, bias);
            }

            return NULL;
        }       

    private:
        
        AbstractDomainType _type;

    };
}