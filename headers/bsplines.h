#pragma once

#include <vector>
#include <string>
#include <complex>
#include <functional>
#include "simulation.h"
struct bsplines
{   





    std::complex<double> B(int i, int degree, std::complex<double> x,const simulation& sim);
    std::complex<double> dB(int i, int degree, std::complex<double> x,const simulation& sim);
    std::complex<double> integrate_matrix_element(int i, int j, std::function<std::complex<double>(int,int,std::complex<double>)> integrand,const simulation& sim);

    std::complex<double> overlap_integrand(int i, int j, std::complex<double> x,const simulation& sim);
    std::complex<double> bsplines::kinetic_integrand(int i, int j, std::complex<double> x,const simulation& sim);
    std::complex<double> bsplines::inv_r_integrand(int i, int j, std::complex<double> x,const simulation& sim);
    std::complex<double> bsplines::inv_r2_integrand(int i, int j, std::complex<double> x,const simulation& sim);
    std::complex<double> bsplines::der_integrand(int i, int j, std::complex<double> x,const simulation& sim);
    
    void save_debug_bsplines(int rank, const simulation& sim);



};