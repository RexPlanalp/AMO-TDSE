#pragma once

enum class RunMode;

class PetscMatrix;

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
        PetscVector() : vector(nullptr), comm(MPI_COMM_NULL), local_start(0), local_end(0) { }

        // Copy Constructor
        PetscVector(const PetscVector& other);

        // Explicit Constructor
        PetscVector(int size, RunMode type);

        // Copy assignment operator
        PetscVector& operator=(const PetscVector& other);

        // Destructor
        ~PetscVector();

        // Assemble the vector
        void assemble();

        // Set petsc name
        void setName(const char* name);

        // Internal Data
        Vec vector;
        MPI_Comm comm;
        int local_start,local_end;
};

//////////////////////////
// Wavefunction Subclass//
//////////////////////////


class Wavefunction : public PetscVector
{
    public:
        using PetscVector::PetscVector;

        std::complex<double> computeNorm(const PetscMatrix& S) const;
        void normalize(const PetscMatrix& S);
};