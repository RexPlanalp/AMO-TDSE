#pragma once

#include <petscvec.h>
#include <complex>
#include "petsc_wrappers/PetscMatrix.h"
#include "mpi.h"

//////////////////////////
// Petsc Vector Wrapper //
//////////////////////////

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

        // Set petsc name
        void setName(const char* name);

        // Internal vector
        Vec vector = nullptr;
};

//////////////////////////
// Wavefunction Subclass//
//////////////////////////


class Wavefunction : public PetscVector
{
    public:
        Wavefunction() = default;
        Wavefunction(int size, RunMode type);
        std::complex<double> computeNorm(const PetscMatrix& S);
        void normalize(const PetscMatrix& S);

        int local_start,local_end;
        MPI_Comm comm = MPI_COMM_NULL;
};