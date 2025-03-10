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

enum class ECSMode
{
    ON,
    OFF
};

class PetscMatrix 
{
    public:
        PetscMatrix() = default;
        ~PetscMatrix();
        
        Mat& getMatrix();

        void saveMatrix(const char* filename);
        void duplicateMatrix(PetscMatrix& M);

        template <typename T>
        void axpy(T alpha, PetscMatrix& X, MatStructure PATTERN);
        



       

    private:
        Mat matrix;
};

class RadialMatrix : public PetscMatrix
{
    public:
        RadialMatrix(const simulation& sim,RadialMatrixType matrixType);

        

        int local_start,local_end;

        radialIntegrand integrand_func;
        void setIntegrand(radialIntegrand integrand);

        void populateMatrix(const simulation& sim,ECSMode ecs);
};



