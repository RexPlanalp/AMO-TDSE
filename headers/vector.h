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
        // Default constructor: Creates a parallel vector
        PetscVector();

        // Constructor: Creates a vector of given size and type
        PetscVector(int size, VectorType type);

        // Copy constructor: Creates a vector by duplicating an existing PETSc Vec
        PetscVector(const Vec& existingVec);

        // Destructor: Ensures proper cleanup of PETSc vector
        ~PetscVector();

        // Returns the PETSc Vec object
        Vec& getVector();

        // Returns the global size of the vector
        PetscInt getSize();

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


