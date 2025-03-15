#include <petscsys.h>
#include "petsc_wrappers/PetscMatrix.h"

void checkErr(PetscErrorCode ierr, const char* msg);
PetscMatrix KroneckerProduct(const PetscMatrix& A, const PetscMatrix& B);