#include "matrix.h"
#include "utility.h"

////////////////////////////
//  Matrix Baseclass //
////////////////////////////

// Constructor: default

// Destructory: destroy the matrix
PetscMatrix::~PetscMatrix()
{
    MatDestroy(&getMatrix());
};

// Method: get the matrix

Mat& PetscMatrix::getMatrix()
{
    return matrix;
};

////////////////////////////
// Radial Matrix Subclass //
////////////////////////////

// Constructor: create a matrix
RadialMatrix::RadialMatrix(int size, int nnz, RadialMatrixType matrixType)
{   
    PetscErrorCode ierr;

    switch(matrixType)
    {
        case RadialMatrixType::SEQUENTIAL:
        ierr = MatCreate(PETSC_COMM_SELF, &getMatrix()); checkError(ierr,"Error Creating Matrix");
        ierr = MatSetSizes(getMatrix(), PETSC_DECIDE, PETSC_DECIDE, size, size); checkError(ierr,"Error Setting Matrix Size");
        ierr = MatSetFromOptions(getMatrix()); checkError(ierr,"Error Setting Matrix Options");
        ierr = MatSeqAIJSetPreallocation(getMatrix(), nnz, NULL); checkError(ierr,"Error Preallocating Matrix");
        ierr = MatSetUp(getMatrix()); checkError(ierr,"Error Setting Up Matrix");
       
        local_start = 0;
        local_end = size;
        break;

        case RadialMatrixType::PARALLEL:
        ierr = MatCreate(PETSC_COMM_WORLD, &getMatrix()); checkError(ierr, "Error Creating Matrix");
        ierr = MatSetSizes(getMatrix(), PETSC_DECIDE, PETSC_DECIDE, size, size); checkError(ierr, "Error Setting Matrix Size");
        ierr = MatSetFromOptions(getMatrix()); checkError(ierr, "Error Setting Matrix Options");
        ierr = MatMPIAIJSetPreallocation(getMatrix(), nnz, NULL, nnz, NULL); checkError(ierr, "Error Preallocating Matrix");
        ierr = MatSetUp(getMatrix()); checkError(ierr, "Error Setting Up Matrix");
        ierr = MatGetOwnershipRange(getMatrix(), &local_start, &local_end); checkError(ierr, "Error Getting Ownership Range");
       
        break;
    }

}