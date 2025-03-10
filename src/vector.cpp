#include <petscvec.h>
#include "vector.h"
#include "utility.h"
#include "matrix.h"

////////////////////////////
//    Vector Baseclass    //
////////////////////////////

// Default Constructor: Creates a parallel vector
PetscVector::PetscVector()
{
    PetscErrorCode ierr;
    ierr = VecCreate(PETSC_COMM_WORLD, &vector); checkError(ierr, "Error creating default parallel vector");
    ierr = VecSetFromOptions(vector); checkError(ierr, "Error setting vector options");
}

// Constructor: Allows specifying size and type
PetscVector::PetscVector(int size, VectorType type)
{
    PetscErrorCode ierr;

    switch(type)
    {
        case VectorType::SEQUENTIAL:
            ierr = VecCreateSeq(PETSC_COMM_SELF, size, &vector); checkError(ierr, "Error creating sequential vector");
            break;
        case VectorType::PARALLEL:
            ierr = VecCreate(PETSC_COMM_WORLD, &vector); checkError(ierr, "Error creating parallel vector");
            ierr = VecSetSizes(vector, PETSC_DECIDE, size); checkError(ierr, "Error setting vector size");
            ierr = VecSetFromOptions(vector); checkError(ierr, "Error setting vector options");
            break;
    }
}

// Copy Constructor
PetscVector::PetscVector(const Vec& existingVec)
{
    PetscErrorCode ierr;
    ierr = VecDuplicate(existingVec, &vector); checkError(ierr, "Error duplicating vector");
    ierr = VecCopy(existingVec, vector); checkError(ierr, "Error copying vector");
}

// Destructor: Destroys the vector
PetscVector::~PetscVector()
{
    PetscErrorCode ierr;
    ierr = VecDestroy(&vector); checkError(ierr, "Error destroying vector");
}

// Method: Get the PETSc vector
Vec& PetscVector::getVector()
{
    return vector;
}

// Method: Get the size of the vector
PetscInt PetscVector::getSize()
{
    PetscInt size;
    PetscErrorCode ierr = VecGetSize(vector, &size);
    checkError(ierr, "Error getting vector size");
    return size;
}

// Method: Set the value of the vector
void PetscVector::setValue(int i, std::complex<double> value)
{
    PetscErrorCode ierr;
    ierr = VecSetValue(vector, i, value, INSERT_VALUES);
    checkError(ierr, "Error setting value");
}

// Method: Assemble the vector
void PetscVector::assemble()
{
    PetscErrorCode ierr;
    ierr = VecAssemblyBegin(vector); checkError(ierr, "Error assembling vector");
    ierr = VecAssemblyEnd(vector); checkError(ierr, "Error assembling vector");
}

// Method: Compute the norm
void PetscVector::computeNorm(std::complex<double>& norm, PetscMatrix& S)
{
    PetscErrorCode ierr;
    PetscVector temp_vector(getSize(), VectorType::PARALLEL); // Ensure parallel vector

    ierr = VecDuplicate(vector, &temp_vector.getVector()); checkError(ierr, "Error duplicating vector");
    ierr = MatMult(S.getMatrix(), vector, temp_vector.getVector()); checkError(ierr, "Error multiplying matrix");
    ierr = VecDot(vector, temp_vector.getVector(), &norm); checkError(ierr, "Error computing dot product");
    norm = std::sqrt(norm);
}

// Method: Scale the vector
template <typename T>
void PetscVector::scale(T factor)
{
    PetscErrorCode ierr;
    ierr = VecScale(vector, factor);
    checkError(ierr, "Error scaling vector");
}

// Explicit template instantiations
template void PetscVector::scale<double>(double);
template void PetscVector::scale<std::complex<double>>(std::complex<double>);
template void PetscVector::scale<int>(int);
