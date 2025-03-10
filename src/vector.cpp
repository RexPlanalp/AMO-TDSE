#include <petscvec.h>
#include "vector.h"
#include "utility.h"
#include "matrix.h"

////////////////////////////
//    Vector Baseclass    //
////////////////////////////

// Constructor: default
PetscVector::PetscVector(int size, VectorType type)
{
    PetscErrorCode ierr;

    switch(type)
    {
        case VectorType::SEQUENTIAL:
            ierr = VecCreateSeq(PETSC_COMM_WORLD,size,&vector); checkError(ierr,"Error creating vector");
            break;
        case VectorType::PARALLEL:
            ierr = VecCreate(PETSC_COMM_WORLD,&vector); checkError(ierr,"Error creating vector");
            ierr = VecSetSizes(vector,PETSC_DECIDE,size); checkError(ierr,"Error setting vector size");
            ierr = VecSetFromOptions(vector); checkError(ierr,"Error setting vector options");
            break;
    }
}

// Destructor: destroy the vector
PetscVector::~PetscVector()
{
    VecDestroy(&getVector());
}

// Method: get the vector
Vec& PetscVector::getVector()
{
    return vector;
}

// Method: set the value of the vector

void PetscVector::setValue(int i, std::complex<double> value)
{
    PetscErrorCode ierr;
    ierr = VecSetValue(vector,i,value,INSERT_VALUES); checkError(ierr,"Error setting value");
}

// Method: assemble the vector
void PetscVector::assemble()
{
    PetscErrorCode ierr;
    ierr = VecAssemblyBegin(vector); checkError(ierr,"Error assembling vector");
    ierr = VecAssemblyEnd(vector); checkError(ierr,"Error assembling vector");
}

// Method:: compute the norm

void PetscVector::computeNorm(std::complex<double>& norm, PetscMatrix& S)
{
    PetscErrorCode ierr;
    PetscVector temp_vector;

    ierr = VecDuplicate(vector,&temp_vector.getVector()); checkError(ierr,"Error duplicating vector");
    ierr = MatMult(S.getMatrix(),vector,temp_vector.getVector()); checkError(ierr,"Error multiplying matrix");
    ierr = VecDot(vector,temp_vector.getVector(),&norm); checkError(ierr,"Error computing dot product");
    norm = std::sqrt(norm);
}

// Method: scale the vector

template <typename T>
void PetscVector::scale(T factor)
{
    VecScale(vector,factor);
}





template void PetscVector::scale<double>(double);
template void PetscVector::scale<std::complex<double>>(std::complex<double>);
template void PetscVector::scale<int>(int);