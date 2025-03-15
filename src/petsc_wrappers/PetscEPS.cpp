#include "petsc_wrappers/PetscEPS.h"
#include "utility.h"
#include "petsc_wrappers/PetscVector.h"
#include "petsc_wrappers/PetscMatrix.h"
#include "mpi.h"
#include "simulation.h"

//////////////////////////
// Petsc EPS Wrapper   //
//////////////////////////

PetscEPS::PetscEPS(RunMode run) 
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

    ierr = EPSCreate(comm, &eps); checkErr(ierr, "Error creating EPS object");
    ierr = EPSSetProblemType(eps,EPS_GNHEP); checkErr(ierr, "Error setting problem type");
    ierr = EPSSetWhichEigenpairs(eps,EPS_SMALLEST_REAL); checkErr(ierr, "Error setting which eigenpairs");
    ierr = EPSSetType(eps,EPSKRYLOVSCHUR); checkErr(ierr, "Error setting type");
}   

PetscEPS::~PetscEPS()
{
    if (eps) 
    {
        EPSDestroy(&eps);
        eps = nullptr;
    }
}

void PetscEPS::setConvergenceParams(const simulation& sim)
{
    PetscErrorCode ierr;
    ierr = EPSSetTolerances(eps,sim.schrodinger_params.tise_tol,sim.schrodinger_params.tise_max_iter); checkErr(ierr, "Error setting tolerances");
}

void PetscEPS::setSolverParams(int requested_pairs)
{
    PetscErrorCode ierr;
    ierr = EPSSetDimensions(eps,requested_pairs,PETSC_DEFAULT,PETSC_DEFAULT); checkErr(ierr, "Error setting dimensions");
    ierr = EPSSetFromOptions(eps); checkErr(ierr, "Error setting options");
}

void PetscEPS::setOperators(const PetscMatrix& H, const PetscMatrix& S)
{
    PetscErrorCode ierr;
    ierr = EPSSetOperators(eps,H.matrix,S.matrix); checkErr(ierr, "Error setting operators");
    ierr = EPSSetUp(eps); checkErr(ierr, "Error setting up EPS");
}

int PetscEPS::solve()
{
    int nconv = 0;
    PetscErrorCode ierr;
    ierr = EPSSolve(eps); checkErr(ierr, "Error solving EPS");
    ierr = EPSGetConverged(eps,&nconv); checkErr(ierr, "Error getting number of converged eigenvalues");
    return nconv;
}

std::complex<double> PetscEPS::getEigenvalue(int i)
{
    PetscErrorCode ierr;
    std::complex<double> eigenvalue;
    ierr = EPSGetEigenvalue(eps,i,&eigenvalue,NULL); checkErr(ierr, "Error getting eigenvalue");
    return eigenvalue;
}

Wavefunction PetscEPS::getEigenvector(int i, const PetscMatrix& S)
{
    PetscErrorCode ierr;
    Wavefunction eigenvector;

    ierr = MatCreateVecs(S.matrix,&eigenvector.vector,NULL); checkErr(ierr, "Error creating vector");
    ierr = EPSGetEigenvector(eps, i, eigenvector.vector, NULL);
    checkErr(ierr, "Error getting eigenvector");

    return eigenvector;
}







