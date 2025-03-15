#include <petscsys.h>
#include "petsc_wrappers/PetscMatrix.h"

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
