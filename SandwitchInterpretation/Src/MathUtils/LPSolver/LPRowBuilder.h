#pragma once
#include "LPSolver.h"
#include "lp_lib.h"
#include <stdlib.h>
#include <string>

namespace AI {

	class LPSolver;

	enum LPConstraintType {LESS_EQUALS, GREATER_EQUALS, EQUALS};

	class LPRowBuilder
	{

	public:
		void setColumnCoef(std::string name, double val);
		void setRightHandSide(double rh);
		void setConstraintType(LPConstraintType type);

		void build();

	private:
		LPRowBuilder(LPSolver *lp);

		LPSolver* solver;

		int *colno;
		REAL* row;

		unsigned numberOfVars;

		LPConstraintType constraintType;
		double rh;

		friend class LPSolver;
	};

}