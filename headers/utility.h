#include <petscsys.h>

#include "petsc_wrappers/PetscMatrix.h"
#include "petsc_wrappers/PetscVector.h"

void checkErr(PetscErrorCode ierr, const char* msg);

PetscMatrix KroneckerProduct(const PetscMatrix& A, const PetscMatrix& B);

double f(int l, int m);

double g(int l, int m);

double a(int l, int m);

double atilde(int l, int m);

double b(int l, int m);

double btilde(int l, int m);

double d(int l, int m);

double dtilde(int l, int m);

double c(int l, int m);

double ctilde(int l, int m);

PetscErrorCode project_out_bound(const PetscMatrix& S, PetscVector& state, const simulation& sim);

std::complex<double> compute_Ylm(int l, int m, double theta, double phi);
