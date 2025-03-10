#pragma once

#include <petscmat.h>

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
        RadialMatrix(int size, int nnz, RadialMatrixType matrixType);

        int local_start,local_end;

};


