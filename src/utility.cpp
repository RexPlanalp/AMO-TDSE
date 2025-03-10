#include <petscsys.h>
#include <iostream>
#include <stdexcept>

// Custom error checker
void checkError(PetscErrorCode ierr, const char* msg = "") 
{
    if (ierr) 
    {
        std::cerr << "PETSc Error: " << msg << " (Code: " << ierr << ")" << std::endl;
        PetscError(PETSC_COMM_SELF, __LINE__, __func__, __FILE__, ierr, PETSC_ERROR_INITIAL, msg);
        throw std::runtime_error("PETSc encountered an error.");
    }
}