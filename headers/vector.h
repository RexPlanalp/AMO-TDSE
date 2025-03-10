#pragma once

#include <petscvec.h>
#include "matrix.h"

enum class VectorType
{
    SEQUENTIAL,
    PARALLEL
};

class PetscVector 
{
    public:
        PetscVector();

        PetscVector(int size, VectorType type);

        PetscVector(const Vec& existingVec);

        ~PetscVector();

        Vec& getVector();

        // Returns the global size of the vector
        void getSize(int& size);

        // Set value in the vector
        void setValue(int i, std::complex<double> value);

        // Assemble the vector after setting values
        void assemble();

        // Compute the norm with respect to a given matrix
        void computeNorm(std::complex<double>& norm, PetscMatrix& S);

        // Scale the vector by a factor
        template <typename T>
        void scale(T factor);

    private:
        Vec vector;
};


