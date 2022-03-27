#pragma once

#include "lp_lib.h"
#include "LPRowBuilder.h"
#include "LPObjectiveBuilder.h"
#include "LPResult.h"

namespace AI {

	enum BoundType { UPPER, LOWER };

	class LPRowBuilder;
	class LPObjectiveBuilder;

	class LPSolver
	{

	public:

		~LPSolver() {
			free_lp(&lp);
		}

		LPSolver(unsigned variableCount);
		
		void addVariableName(std::string name);
		
		LPRowBuilder getRowBuilder();
		LPObjectiveBuilder getObjectiveBuilder();
		void boundVariable(std::string varName, double lb, double ub);
		void boundVariable(std::string varName, double bound, AI::BoundType boundType);

		AI::LPResult solveLP();

		void printProblem();

		void includeVarInResult(std::string name);

		void print();

	private:

		lprec* lp;
		unsigned variableCount;

		unsigned addedNamesCount;

		std::vector<unsigned> variablesInObjective;

		unsigned rowCount;

		friend class LPRowBuilder;
		friend class LPObjectiveBuilder;
	};

}