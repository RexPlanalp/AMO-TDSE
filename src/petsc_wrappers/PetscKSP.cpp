#include <slepceps.h>
#include "simulation.h"
#include "petsc_wrappers/PetscMatrix.h"
#include "petsc_wrappers/PetscVector.h"
#include "mpi.h"
#include "petsc_wrappers/PetscKSP.h"
#include "utility.h"

PetscKSP::PetscKSP(RunMode run) 
{   
    PetscErrorCode ierr;

    switch(run)
    {
        case RunMode::SEQUENTIAL:
            comm = PETSC_COMM_SELF;
            break;
        case RunMode::PARALLEL:
            comm = PETSC_COMM_WORLD;
            break;
    }

    ierr = KSPCreate(comm, &ksp); checkErr(ierr,"Error creating KSP");
}  


PetscKSP::~PetscKSP()
{
    if (ksp) 
    {
        KSPDestroy(&ksp);
        ksp = nullptr;
    }
}

void PetscKSP::setConvergenceParams(const simulation& sim)
{   
    PetscErrorCode ierr;
    ierr = KSPSetTolerances(ksp, sim.schrodinger_params.tdse_tol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); checkErr(ierr,"Error setting tolerances");
    ierr = KSPSetFromOptions(ksp); checkErr(ierr,"Error setting options");
}

void PetscKSP::setOperators(const PetscMatrix& L)
{   
    PetscErrorCode ierr;
    ierr = KSPSetOperators(ksp, L.matrix, L.matrix); checkErr(ierr,"Error setting operators");
    ierr = KSPSetReusePreconditioner(ksp, PETSC_TRUE); checkErr(ierr,"Error setting reuse preconditioner");
}

