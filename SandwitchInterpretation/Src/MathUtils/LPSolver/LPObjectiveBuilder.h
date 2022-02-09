#pragma once
#include "LPSolver.h"
#include "lp_lib.h"
#include <stdlib.h>
#include <string>

namespace AI {

	class LPSolver;

	enum LPObjectiveType { MAXIMIZE, MINIMIZE };

	class LPObjectiveBuilder
	{

	public:
		void setColumnCoef(std::string name, double val);
		void setObjectiveType(LPObjectiveType type);

		void build();

	private:
		LPObjectiveBuilder(LPSolver* lp);

		LPSolver* solver;

		int* colno;
		REAL* row;

		unsigned numberOfVars;

		LPObjectiveType objectiveType;

		friend class LPSolver;
	};

}