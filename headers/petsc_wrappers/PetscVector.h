#pragma once

#include <petscvec.h>
#include <complex>
#include "petsc_wrappers/PetscMatrix.h"

//////////////////////////
// Petsc Vector Wrapper //
//////////////////////////

enum class VectorType
{
    SEQUENTIAL,
    PARALLEL
};

class PetscVector
{
    public:

        // Default Constructor
        PetscVector() = default;

        // Copy Constructor
        PetscVector(const PetscVector& other);

        // Copy assignment operator
        PetscVector& operator=(const PetscVector& other);

        // Destructor
        ~PetscVector();

        // Assemble the vector
        void assemble();

        // Internal vector
        Vec vector = nullptr;
};

//////////////////////////
// Wavefunction Subclass//
//////////////////////////


class Wavefunction : public PetscVector
{
    public:

        std::complex<double> computeNorm(const PetscMatrix& S);
};