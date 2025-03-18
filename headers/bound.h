#pragma once

#include "petsc_wrappers/PetscVector.h"
#include "simulation.h"

namespace bound
{
    double computeBoundPopulation(int n_bound, int l_bound, const PetscVector& state, const simulation& sim);
    void computeBoundDistribution(int rank, const simulation& sim);
}