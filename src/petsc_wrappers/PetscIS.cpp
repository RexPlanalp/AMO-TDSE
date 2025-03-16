#include "petsc_wrappers/PetscIS.h"
#include "utility.h"

PetscIS::PetscIS() : is(nullptr), comm(MPI_COMM_NULL){}

PetscIS::PetscIS(int size, int start, int step_size, RunMode mode)
{
    PetscErrorCode ierr;

    switch(mode)
    {
        case RunMode::SEQUENTIAL:
            comm = PETSC_COMM_SELF;

            ierr = ISCreateStride(comm, size, start, step_size, &is); checkErr(ierr, "Error creating IS");
            break;
        case RunMode::PARALLEL:
            comm = PETSC_COMM_WORLD;

            ierr = ISCreateStride(comm, size, start, step_size, &is); checkErr(ierr, "Error creating IS");
            break;
    }
}

PetscIS::~PetscIS()
{
    if (is) 
    {
        PetscErrorCode ierr;
        ierr = ISDestroy(&is); checkErr(ierr, "Error destroying IS");

        is = nullptr;
    }
}