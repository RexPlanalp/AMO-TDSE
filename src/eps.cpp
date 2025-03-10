#include "simulation.h"
#include <slepceps.h>

#include "eps.h"
#include "utility.h"
#include "matrix.h"
#include "vector.h"


////////////////////////////
//  EPS Baseclass         //
////////////////////////////

// Constructor

PetscEPS::PetscEPS(const simulation& sim)
{   
    PetscErrorCode ierr;
    ierr = EPSCreate(PETSC_COMM_WORLD, &eps); checkError(ierr,"Error creating EPS object");
    ierr = EPSSetProblemType(eps, EPS_GNHEP); checkError(ierr,"Error setting problem type");
    ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); checkError(ierr,"Error setting which eigenpairs");
    ierr = EPSSetType(eps,EPSKRYLOVSCHUR); checkError(ierr,"Error setting type");
    ierr = EPSSetFromOptions(eps); checkError(ierr,"Error setting options");
    ierr = EPSSetUp(eps); checkError(ierr,"Error setting up EPS");
    ierr = EPSSetTolerances(eps,sim.schrodinger_params.tise_tol,sim.schrodinger_params.tise_max_iter); checkError(ierr,"Error setting tolerances");
}

// Destructor

PetscEPS::~PetscEPS()
{
    EPSDestroy(&eps);
}

// Method: set parameters
void PetscEPS::setParameters(int num_of_energies)
{
    PetscErrorCode ierr;
    ierr = EPSSetDimensions(eps,num_of_energies,PETSC_DEFAULT,PETSC_DEFAULT); checkError(ierr,"Error setting dimensions");
}

// Method: solve eigenvalue problem
void PetscEPS::setOperators(PetscMatrix& H, PetscMatrix& S)
{
    PetscErrorCode ierr;
    ierr = EPSSetOperators(eps,H.getMatrix(),S.getMatrix()); checkError(ierr,"Error setting operators");
}

// Method: solve eigenvalue problem

void PetscEPS::solve(int& nconv)
{
    PetscErrorCode ierr;
    ierr = EPSSolve(eps); checkError(ierr,"Error solving EPS");
    ierr = EPSGetConverged(eps,&nconv); checkError(ierr,"Error getting number of converged eigenvalues");
}

// Method: get eigenvalue
void PetscEPS::getEigenvalue(int i,std::complex<double>& eigenvalue)
{
    PetscErrorCode ierr;
    ierr = EPSGetEigenvalue(eps,i,&eigenvalue,NULL); checkError(ierr,"Error getting eigenvalue");
}


//Method: get eigenvector

void PetscEPS::getNormalizedEigenvector(PetscVector& eigenvector, int i, PetscMatrix& S)
{   
    PetscErrorCode ierr;

    ierr = MatCreateVecs(S.getMatrix(),&eigenvector.getVector(), NULL); checkError(ierr,"Error creating vector");
    ierr = EPSGetEigenvector(eps,i,eigenvector.getVector(),NULL); checkError(ierr,"Error getting eigenvector");

    std::complex<double> norm;
    eigenvector.computeNorm(norm,S);
    eigenvector.scale(1.0/norm.real());
}


