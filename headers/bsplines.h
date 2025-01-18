#pragma once

#include <vector>
#include <string>
#include <complex>
#include <functional>
#include "simulation.h"
#include <petscmat.h>

namespace bsplines
{
    std::complex<double> B(int i, int degree, std::complex<double> x, const simulation& sim);
    std::complex<double> dB(int i, int degree, std::complex<double> x, const simulation& sim);

    std::complex<double> integrate_matrix_element(
        int i, int j,
        std::function<std::complex<double>(int, int, std::complex<double>, const simulation&)> integrand,
        const simulation& sim);

    std::complex<double> overlap_integrand(int i, int j, std::complex<double> x, const simulation& sim);
    std::complex<double> kinetic_integrand(int i, int j, std::complex<double> x, const simulation& sim);
    std::complex<double> invr_integrand(int i, int j, std::complex<double> x, const simulation& sim);
    std::complex<double> invr2_integrand(int i, int j, std::complex<double> x, const simulation& sim);
    std::complex<double> der_integrand(int i, int j, std::complex<double> x, const simulation& sim);

    void save_debug_bsplines(int rank, const simulation& sim);

    PetscErrorCode construct_matrix(const simulation& sim, Mat& M, std::function<std::complex<double>(int, int, std::complex<double>, const simulation&)> integrand);
    PetscErrorCode construct_overlap(const simulation& sim, Mat& S);
};
