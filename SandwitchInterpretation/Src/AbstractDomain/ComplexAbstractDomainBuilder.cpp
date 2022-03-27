#include "ComplexAbstractDomainBuilder.h"

AI::AbstractDomain* AI::ComplexAbstractDomainBuilder::build(double** bounds, unsigned dim)
{
	if (_type == AI::ComplexAbstractDomainType::ZONOTOPE_N_DOMAIN) {
		AI::Zonotope *initial = (AI::Zonotope*) AbstractDomainBuilder(AI::AbstractDomainType::ZONOTOPE_DOMAIN).build(bounds, dim);
		return new AI::ZonotopeN(*initial, n);
	}
}
