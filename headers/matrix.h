#pragma once

#include <petscmat.h>

class PetscMatrix 
{
    public:
        PetscMatrix() = default;
        ~PetscMatrix();
        
        Mat getMatrix();

    private:
        Mat matrix;
};

class RadialMatrix : public PetscMatrix
{
    public:
        RadialMatrix()

}

