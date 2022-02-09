#include "ZonotopeOperations.h"
#include "../../MathUtils/LPSolver/LPSolver.h"
#include "../../MathUtils/LPSolver/LPRowBuilder.h"

AI::Zonotope AI::createBoundingZonotope(std::vector<arma::mat> points, std::vector<arma::mat> generators)
{
	unsigned dim = points[0].size();
	unsigned numPoints = points.size();
	unsigned numGens = generators.size();

	LPSolver solver = LPSolver(numGens + numPoints * numGens + dim);
	for (unsigned g = 0; g < numGens; g++) {
		solver.addVariableName("g" + std::to_string(g));
		for (unsigned p = 0; p < numPoints; p++) {
			solver.addVariableName("p_" + std::to_string(p) + "_" + std::to_string(g));
		}
	}
	for (unsigned d = 0; d < dim; ++d) {
		solver.addVariableName("bias_" + std::to_string(d));
	}


	//Add constrains for points inside the zonotope:
	for (unsigned p = 0; p < numPoints; ++p) {
		for (unsigned d = 0; d < dim; ++d) {
			AI::LPRowBuilder row = solver.getRowBuilder();
			for (unsigned g = 0; g < numGens; g++) {
				std::string varName = "p_" + std::to_string(p) + "_" + std::to_string(g);
				row.setColumnCoef(varName, generators[g](d));
			}
			row.setColumnCoef("bias_" + std::to_string(d), 1);
			row.setConstraintType(AI::EQUALS);
			row.setRightHandSide(points[p](d));
			row.build();
		}
	}

	// Set bounds for the epsilons, depending on the scaling factor of each generator
	// (scaling the epsilon bounds instead of the generator itself)
	for (unsigned g = 0; g < numGens; g++) {
		for (unsigned p = 0; p < numPoints; p++) {
			//lower bouind
			AI::LPRowBuilder row = solver.getRowBuilder();
			row.setColumnCoef("g" + std::to_string(g), -1);
			std::string varName = "p_" + std::to_string(p) + "_" + std::to_string(g);
			row.setColumnCoef(varName, -1);
			row.setRightHandSide(0);
			row.setConstraintType(AI::LESS_EQUALS);
			row.build();

			//upper bouind
			row = solver.getRowBuilder();
			row.setColumnCoef("g" + std::to_string(g), -1);
			varName = "p_" + std::to_string(p) + "_" + std::to_string(g);
			row.setColumnCoef(varName, 1);
			row.setRightHandSide(0);
			row.setConstraintType(AI::LESS_EQUALS);
			row.build();
		}
	}

	//make scaling positive
	for (unsigned g = 0; g < numGens; g++) {
		solver.boundVariable("g" + std::to_string(g), 0, AI::LOWER);
	}


	AI::LPObjectiveBuilder objective = solver.getObjectiveBuilder();
	for (unsigned g = 0; g < numGens; g++) {
		objective.setColumnCoef("g" + std::to_string(g), 1);
	}
	objective.setObjectiveType(AI::MINIMIZE);
	objective.build();


	for (unsigned g = 0; g < numGens; g++) {
		solver.includeVarInResult("g" + std::to_string(g));
	}
	for (unsigned d = 0; d < dim; ++d) {
		solver.includeVarInResult("bias_" + std::to_string(d));
	}

	AI::LPResult res = solver.solveLP();

	std::vector<arma::mat> resGenerators;
	for (unsigned g = 0; g < numGens; g++) {
		if (arma::norm(generators[g]) < 0.0000001 || res.vars[g].val < 0.0000001) continue;
		resGenerators.push_back(res.vars[g].val * generators[g]);
	}
	arma::mat resBias(1, dim, arma::fill::zeros);
	for (unsigned d = 0; d < dim; d++) {
		resBias[d] = res.vars[numGens + d].val;
	}

	return AI::Zonotope(resGenerators, resBias);

}