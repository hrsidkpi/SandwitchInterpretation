#include "LPRowBuilder.h"
#include <stdlib.h>


AI::LPRowBuilder::LPRowBuilder(AI::LPSolver* lp) : solver(lp) {
	colno = (int*)malloc(lp->variableCount * sizeof(*colno));
	row = (REAL*)malloc(lp->variableCount * sizeof(*row));
	
	rh = 0;

	numberOfVars = 0;
}

void AI::LPRowBuilder::setColumnCoef(std::string name, double val)
{
	char* c_name = new char[name.length() + 1];
	strcpy(c_name, name.c_str());
	int index = get_nameindex(solver->lp, c_name, false);
	 
	row[numberOfVars] = val;
	colno[numberOfVars] = index;
	numberOfVars++;

}

void AI::LPRowBuilder::setRightHandSide(double rh)
{
	this->rh = rh;
}

void AI::LPRowBuilder::setConstraintType(LPConstraintType type)
{
	constraintType = type;
}

void AI::LPRowBuilder::build()
{
	//TODO check if result is true or false and throw exception if false
	set_add_rowmode(solver->lp, TRUE);

	if (constraintType == AI::EQUALS)
		add_constraintex(solver->lp, numberOfVars, row, colno, EQ, rh);
	if (constraintType == AI::GREATER_EQUALS)
		add_constraintex(solver->lp, numberOfVars, row, colno, GE, rh);
	if (constraintType == AI::LESS_EQUALS)
		add_constraintex(solver->lp, numberOfVars, row, colno, LE, rh);
}
