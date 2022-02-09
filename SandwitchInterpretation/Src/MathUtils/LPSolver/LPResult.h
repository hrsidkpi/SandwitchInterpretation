#pragma once
#include <armadillo>

namespace AI {

	struct LPResultVariable
	{
		LPResultVariable(std::string name, double val) : name(name), val(val) {};

		std::string name;
		double val;
	};

	struct LPResult
	{

		LPResult(std::vector<LPResultVariable> vars, double objectiveValue) : vars(vars), objectiveValue(objectiveValue) {};

		std::vector<LPResultVariable> vars;
		double objectiveValue;

	};

	

}