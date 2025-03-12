#include <petscvec.h>
#include "petsc_wrappers/PetscVector.h"
#include <complex>
#include "utility.h"

//////////////////////////
// Petsc Vector Wrapper //
//////////////////////////


PetscVector::PetscVector(const PetscVector& other)
{
    VecDuplicate(other.vector, &vector);
}

PetscVector& PetscVector::operator=(const PetscVector& other)
{
    if (this != &other)  
    {
        if (vector) {
            VecDestroy(&vector);
        }
        VecDuplicate(other.vector, &vector);
    }
    return *this;
}

PetscVector::~PetscVector()
{
    VecDestroy(&vector);
}

void PetscVector::assemble()
{
    VecAssemblyBegin(vector);
    VecAssemblyEnd(vector);
}

//////////////////////////
// Wavefunction Subclass//
//////////////////////////


std::complex<double> Wavefunction::computeNorm(const PetscMatrix& S)
{
    PetscErrorCode ierr;
    std::complex<double> norm = 0;
    PetscVector temp(*this);
    ierr = MatMult(S.matrix, vector,temp.vector); checkErr(ierr, "Error in MatMult");
    ierr = VecDot(vector, temp.vector, &norm); checkErr(ierr, "Error in VecDot");
    return norm;
}



