#pragma once

#include "block.h"
#include <iostream>
#include <fstream>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>
#include "simulation.h"
#include <map>


namespace block 
{
    PetscErrorCode load_final_state(const char* filename, Vec* state, int total_size);
    PetscErrorCode comptue_norm(Vec& state, Mat& S,int n_blocks,int n_basis, std::map<int,std::pair<int,int>>& block_to_lm);
    void compute_block_distribution(int rank,const simulation& sim);
}