#include "Halfspace.h"

bool AI::Halfspace::containsVertex(double* vertex)
{
    double res = 0;
    for (unsigned i = 0; i < dim; i++) {
        res += vertex[i] * (this->weights[i]);
    }
    return res + this->bias < 0;
}

bool AI::Halfspace::seperatorContainsVertex(double* vertex)
{
    double res = 0;
    for (unsigned i = 0; i < dim; i++) {
        res += vertex[i] * this->weights[i];
    }
    return res + this->bias == 0;
}

bool AI::Halfspace::applyAffineTransformation(arma::mat transformation_linear_mat, arma::mat transformation_translate_mat)
{
    unsigned new_dim = transformation_linear_mat.n_rows;

    arma::mat new_weights(1, new_dim, arma::fill::zeros);

    bool has_sol = arma::solve(new_weights, transformation_linear_mat.t(), weights.t());
    if (has_sol && arma::dot(new_weights, new_weights) != 0) {
        double* new_weights_ptr = new_weights.memptr();

        for (unsigned i = 0; i < new_dim; ++i) {
            this->weights[i] = new_weights_ptr[i];
        }
        this->bias -= arma::dot(new_weights, transformation_translate_mat);
        this->dim = new_dim;
        return true;
    }
    else {
        return false;
    }
}

void AI::Halfspace::print()
{
    std::cout << "halfpsace of dim " << dim << ":" << std::endl;

    bool added = false;

    for (unsigned i = 0; i < dim; i++) {
        if (!added) {
            if (weights[i] > 0) {
                std::cout << weights[i] << "x" << i;
                added = true;
            }
            if (weights[i] < 0) {
                std::cout << "-" << -weights[i] << "x" << i;
                added = true;
            }
        }
        else {
            if (weights[i] > 0)
                std::cout << " + " << weights[i] << "x" << i;
            if (weights[i] < 0)
                std::cout << " - " << -weights[i] << "x" << i;
        }
    }

    if (added && bias < 0)
        std::cout << " - " << -bias;
    else if (added) {
        std::cout << " + " << bias;
    }
    else
        std::cout << bias;
    std::cout << " < 0" << std::endl;
}
