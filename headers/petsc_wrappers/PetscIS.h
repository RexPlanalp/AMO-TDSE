#pragma once

enum class RunMode;

#include "mpi.h"
#include "petscis.h"
#include "simulation.h"

//////////////////////////
// Petsc IS Wrapper     //
//////////////////////////


class PetscIS
{
    public:

        // Default Constructor
        PetscIS();

        // Explicit Constructor
        PetscIS(int size, int start, int step_size, RunMode mode);

        // Destructor
        ~PetscIS();

        // Internal IS
        IS is;

        // Internal Communicator
        MPI_Comm comm;
};