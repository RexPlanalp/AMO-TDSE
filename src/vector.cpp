#include <petscvec.h>
#include "vector.h"


////////////////////////////
//    Vector Baseclass    //
////////////////////////////

// Constructor: default

// Destructor: destroy the vector
PetscVector::~PetscVector()
{
    VecDestroy(&getVector());
}