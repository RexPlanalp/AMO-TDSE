#include "matrix.h"

// Constructor: default

// Destructory: destroy the matrix
PetscMatrix::~PetscMatrix()
{
    MatDestroy(&matrix);
};

// Method:: get the matrix

Mat PetscMatrix::getMatrix()
{
    return matrix;
};