#pragma once

#include <petscmat.h>
#include "simulation.h"
#include "bsplines.h"

namespace tise 
{
    void solve_tise(const simulation& sim,int rank);
    void prepare_matrices(const simulation& sim,int rank);
}

