#pragma once

#include <petscmat.h>
#include "simulation.h"
#include "bsplines.h"

namespace tise 
{
    PetscErrorCode solve_tise(const simulation& sim,int rank);
    PetscErrorCode prepare_matrices(const simulation& sim,int rank);
}

