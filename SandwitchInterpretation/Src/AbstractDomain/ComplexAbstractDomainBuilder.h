#pragma once
#include "AbstractDomain.h"
#include "Zonotope/Zonotope.h"
#include "AbstractDomainBuilder.h"
#include "ZonotopeN/ZonotopeN.h"


namespace AI {

	enum class ComplexAbstractDomainType { ZONOTOPE_N_DOMAIN };

	class ComplexAbstractDomainBuilder
	{
	public:
		ComplexAbstractDomainBuilder(ComplexAbstractDomainType type, unsigned n) : _type(type), n(n) {}

		AbstractDomain* build(double** bounds, unsigned dim);

	private:
		unsigned n;
		ComplexAbstractDomainType _type;
	};

}