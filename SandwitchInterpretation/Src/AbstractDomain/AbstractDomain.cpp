#include "AbstractDomain.h"

void AI::AbstractDomain::applyReLu() {
	for (unsigned i = 0; i < dim; ++i) {
		applyReLuOnDim(i);
	}
}
