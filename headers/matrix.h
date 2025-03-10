#pragma once

#include <petscmat.h>
#include <functional>
#include <complex>
#include <vector>
#include "simulation.h"

using radialIntegrand = std::function<std::complex<double>(int, int, std::complex<double>, int, const std::vector<std::complex<double>>&)>;

enum class RadialMatrixType
{
    SEQUENTIAL,
    PARALLEL
};

class PetscMatrix 
{
    public:
        PetscMatrix() = default;
        ~PetscMatrix();
        
        Mat& getMatrix();

        void saveMatrix();

    private:
        Mat matrix;
};

class RadialMatrix : public PetscMatrix
{
    public:
        RadialMatrix(const simulation& sim, int nnz, RadialMatrixType matrixType);

        

        int local_start,local_end;

        radialIntegrand integrand_func;
        void setIntegrand(radialIntegrand integrand);

        void RadialMatrix::populateMatrix(const simulation& sim,bool use_ecs);
};



