#pragma once

enum class RunMode;

#include <slepceps.h>
#include "simulation.h"
#include "petsc_wrappers/PetscMatrix.h"
#include "petsc_wrappers/PetscVector.h"
#include "mpi.h"

//////////////////////////
// Petsc KSP Wrapper    //
//////////////////////////


class PetscKSP
{
    public:

        // Default Constructor
        PetscKSP(RunMode run);

        // Destructor
        ~PetscKSP();

        // Set convergence params
        void setConvergenceParams(const simulation& sim); 

        // Set operators
        void setOperators(const PetscMatrix& L);

        // Solve the eigenvalue problem
        void solve();

        // Internal EPS 
        KSP ksp = nullptr;

        // Internal communicator
        MPI_Comm comm = MPI_COMM_NULL;
};
