#pragma once

#include <armadillo>
#include <algorithm>
#include <iterator>


namespace AI {

    class Halfspace {

    public:


        ~Halfspace() {
        }

        Halfspace(unsigned dim, arma::mat weights, double bias) : dim(dim), weights(weights), bias(bias) {}

        bool containsVertex(double* vertex);

        bool seperatorContainsVertex(double* vertex);

        bool applyAffineTransformation(arma::mat transformation_linear_mat, arma::mat transformation_translate_mat);

        void print();

        unsigned dim;

        arma::mat weights;
        double bias;
    private:



    };


}