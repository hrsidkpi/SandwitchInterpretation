#include "ZonotopeOperations.h"
#include "../../MathUtils/LPSolver/LPSolver.h"
#include "../../MathUtils/LPSolver/LPRowBuilder.h"

AI::Zonotope AI::createBoundingZonotope(std::vector<arma::mat> points, std::vector<arma::mat> generators)
{
	unsigned dim = points[0].size();
	unsigned numPoints = points.size();
	unsigned numGens = generators.size();

	std::cout << "Bounding " << numPoints << " points with a zonotope (" << numGens << " generators, " << dim << " dims)." << std::endl;

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
	arma::mat resBias(dim, 1, arma::fill::zeros);
	for (unsigned d = 0; d < dim; d++) {
		resBias[d] = res.vars[numGens + d].val;
	}

	return AI::Zonotope(resGenerators, resBias);

}

AI::Zonotope AI::joinZonotopes(AI::Zonotope z1, AI::Zonotope z2) {
	arma::mat bias = (z1.getBias() + z2.getBias()) / 2;

	std::vector<arma::mat> g1 = z1.getGenerators();
	std::vector<arma::mat> g2 = z2.getGenerators();
	if (g1.size() < g2.size()) {
		for (unsigned _ = 0; _ < g2.size() - g1.size(); _++)
			g1.push_back(arma::mat(1, z1.getDimension(), arma::fill::zeros));
	}
	if (g2.size() < g1.size()) {
		for (unsigned _ = 0; _ < g1.size() - g2.size(); _++)
			g2.push_back(arma::mat(1, z1.getDimension(), arma::fill::zeros));
	}

	std::vector<arma::mat> newGens;
	for (unsigned i = 0; i < g1.size(); i++) {
		newGens.push_back((g1[i] + g2[i]) / 2);
		newGens.push_back((g1[i] - g2[i]) / 2);
	}
	newGens.push_back((z1.getBias() - z2.getBias()) / 2);

	return AI::Zonotope(newGens, bias);
}

std::pair<AI::Zonotope, AI::Zonotope> AI::splitZonotope(AI::Zonotope z) {
	unsigned bestIndex = -1;
	unsigned bestMag = 0;
	for (unsigned i = 0; i < z.getGenerators().size(); i++) {
		double norm = arma::norm(z.getGenerators()[i]);
		if (norm > bestMag) {
			bestMag = norm;
			bestIndex = i;
		}
	}

	AI::Zonotope zPos(z);
	AI::Zonotope zNeg(z);

	zPos.changeEpsilonBounds(bestIndex, 0, 1);
	zNeg.changeEpsilonBounds(bestIndex, -1, 0);

	std::pair<AI::Zonotope, AI::Zonotope> res(zPos, zNeg);
	return res;
}