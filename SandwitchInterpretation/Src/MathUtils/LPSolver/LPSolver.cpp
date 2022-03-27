#include "LPSolver.h"
#include "../../AbstractDomain/Zonotope/Zonotope.h"

AI::LPSolver::LPSolver(unsigned variableCount) : variableCount(variableCount)
{
	lp = make_lp(0, variableCount);
	set_add_rowmode(lp, TRUE);

	addedNamesCount = 0;
	rowCount = 0;
}

void AI::LPSolver::addVariableName(std::string name)
{
	char* c_name = new char[name.length() + 1];
	strcpy(c_name, name.c_str());
	set_col_name(lp, addedNamesCount + 1, c_name); //using namesCount+1 because lp_solve uses column indexes starting at 1.
	addedNamesCount++;

	boundVariable(name, -get_infinite(lp), AI::LOWER);

	delete[] c_name;
}

AI::LPRowBuilder AI::LPSolver::getRowBuilder()
{
	return LPRowBuilder(this);
}

AI::LPObjectiveBuilder AI::LPSolver::getObjectiveBuilder()
{
	return AI::LPObjectiveBuilder(this);
}

void AI::LPSolver::boundVariable(std::string varName, double lb, double ub)
{
	char* c_name = new char[varName.length() + 1];
	strcpy(c_name, varName.c_str());
	int index = get_nameindex(lp, c_name, false);

	set_lowbo(lp, index, lb);
	set_upbo(lp, index, ub);

	delete[] c_name;
}

void AI::LPSolver::boundVariable(std::string varName, double bound, AI::BoundType boundType)
{
	char* c_name = new char[varName.length() + 1];
	strcpy(c_name, varName.c_str());
	int index = get_nameindex(lp, c_name, false);

	if (boundType == AI::LOWER)
		set_lowbo(lp, index, bound);
	if (boundType == AI::UPPER)
		set_upbo(lp, index, bound);

	delete[] c_name;

}

void AI::LPSolver::printProblem() 
{
	print_lp(lp);
}

AI::LPResult AI::LPSolver::solveLP()
{
	if (rowCount == 0) {
		AI::LPRowBuilder row = getRowBuilder();
		for(unsigned i = 0; i < variableCount; i++)
			row.setColumnCoef(get_col_name(lp, i+1), 0);
		row.setConstraintType(AI::LPConstraintType::EQUALS);
		row.setRightHandSide(0);
		row.build();
		rowCount++;
	}


	set_verbose(lp, IMPORTANT);

	solve(lp); //TODO check result

	double objectiveVal = get_objective(lp);

	REAL* varVals;
	varVals = (REAL*)malloc(variableCount * sizeof(*varVals));
	get_variables(lp, varVals);

	std::vector<AI::LPResultVariable> res;
	for (unsigned colno : variablesInObjective) {
		double varVal = varVals[colno - 1]; //colno-1 because colno is lp_solve index which starts at 1
		char* varName = get_col_name(lp, colno);
		AI::LPResultVariable var(varName, varVal);
		res.push_back(var);
	}

	free(varVals);
	return AI::LPResult(res, objectiveVal);

}

void AI::LPSolver::includeVarInResult(std::string name)
{
	char* c_name = new char[name.length() + 1];
	strcpy(c_name, name.c_str());
	variablesInObjective.push_back(get_nameindex(lp, c_name, false));
	delete[] c_name;

}

void AI::LPSolver::print()
{
	write_LP(lp, stdout);
}
