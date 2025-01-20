#pragma once

#include <petscmat.h>
#include "simulation.h"

namespace tdse 
{
    
PetscErrorCode load_starting_state(const simulation& sim);
};