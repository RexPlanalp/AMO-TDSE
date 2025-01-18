#pragma once

#include <petscmat.h>
#include "simulation.h"
#include "bsplines.h"

class tise
{   
public: 
    static PetscErrorCode construct_matrix(const simulation& sim, Mat& M, std::function<std::complex<double>(int, int, std::complex<double>, const simulation&)> integrand);
    static PetscErrorCode construct_overlap(const simulation& sim, Mat& S);
private:
};