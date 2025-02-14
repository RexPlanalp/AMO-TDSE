#pragma once

#include <vector>
#include <string>
#include <complex>
#include <functional>
#include "simulation.h"
#include <petscmat.h>

namespace bsplines
{
    void save_debug_bsplines(int rank, const simulation& sim);
    std::complex<double> B(int i, int degree, std::complex<double> x, const std::vector<std::complex<double>>& knot_vector);
    std::complex<double> dB(int i, int degree, std::complex<double> x, const std::vector<std::complex<double>>& knot_vector);
    std::complex<double> integrate_matrix_element(int i, int j,std::function<std::complex<double>(int, int, std::complex<double>, int,const std::vector<std::complex<double>>&)> integrand,const simulation& sim,bool use_ecs);
    std::complex<double> overlap_integrand(int i, int j, std::complex<double> x, int degree,const std::vector<std::complex<double>>& knot_vector);
    std::complex<double> kinetic_integrand(int i, int j, std::complex<double> x, int degree,const std::vector<std::complex<double>>& knot_vector);
    std::complex<double> invr_integrand(int i, int j, std::complex<double> x, int degree,const std::vector<std::complex<double>>& knot_vector);
    std::complex<double> invr2_integrand(int i, int j, std::complex<double> x, int degree,const std::vector<std::complex<double>>& knot_vector);
    std::complex<double> der_integrand(int i, int j, std::complex<double> x,int degree,const std::vector<std::complex<double>>& knot_vector);
    PetscErrorCode construct_matrix(const simulation& sim, Mat& M, std::function<std::complex<double>(int, int, std::complex<double>, int,std::vector<std::complex<double>>)> integrand,bool use_mpi,bool use_ecs);
   

    PetscErrorCode save_matrix(Mat A, const char *filename);
    PetscErrorCode SaveMatrixToCSV(Mat M, const std::string& filename);
};
