#pragma once

#include <petscmat.h>
#include "simulation.h"
#include "bsplines.h"

class tise
{   
public: 
    static PetscErrorCode construct_overlap(const simulation& sim, Mat& S);
private:
};