#pragma once

#include <petscmat.h>
#include "simulation.h"
#include "bsplines.h"

namespace tise 
{
    PetscErrorCode solve_tise(simulation& sim);
}

