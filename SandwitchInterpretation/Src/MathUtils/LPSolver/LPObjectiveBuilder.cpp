#include "LPObjectiveBuilder.h"


		AI::LPObjectiveBuilder::LPObjectiveBuilder(AI::LPSolver* lp) : solver(lp) 
		{
			colno = (int*)malloc(lp->variableCount * sizeof(*colno));
			row = (REAL*)malloc(lp->variableCount * sizeof(*row));
		
			numberOfVars = 0;
		}
		
		void AI::LPObjectiveBuilder::setColumnCoef(std::string name, double val)
		{
			char* c_name = new char[name.length() + 1];
			strcpy(c_name, name.c_str());
			int index = get_nameindex(solver->lp, c_name, false);
		
			row[numberOfVars] = val;
			colno[numberOfVars] = index;
			numberOfVars++;

			delete[] c_name;

		}
		
		void AI::LPObjectiveBuilder::setObjectiveType(LPObjectiveType type)
		{
			objectiveType = type;
		}

		void AI::LPObjectiveBuilder::setObjectiveTypeAndRebuild(LPObjectiveType type) {
			objectiveType = type;
			if (objectiveType == AI::MAXIMIZE) set_maxim(solver->lp);
			if (objectiveType == AI::MINIMIZE) set_minim(solver->lp);
		}
		
		void AI::LPObjectiveBuilder::build()
		{
		
			set_add_rowmode(solver->lp, FALSE);
		
			set_obj_fnex(solver->lp, numberOfVars, row, colno);
		
			if (objectiveType == AI::MAXIMIZE) set_maxim(solver->lp);
			if (objectiveType == AI::MINIMIZE) set_minim(solver->lp);
			
			free(colno);
			free(row);
		}
		