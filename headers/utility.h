#include <petscsys.h>
#include "petsc_wrappers/PetscMatrix.h"

void checkErr(PetscErrorCode ierr, const char* msg);
PetscMatrix KroneckerProduct(const PetscMatrix& A, const PetscMatrix& B);
double f(int l, int m);
double g(int l, int m);