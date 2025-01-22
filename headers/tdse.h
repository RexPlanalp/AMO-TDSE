#pragma once

#include <petscmat.h>
#include "simulation.h"

namespace tdse 
{
    
PetscErrorCode load_starting_state(const simulation& sim, Vec& tdse_state);
PetscErrorCode KroneckerProductParallel(Mat A, PetscInt nnz_A, Mat B, PetscInt nnz_B, Mat *C_out);
PetscErrorCode _construct_S_atomic(const simulation& sim, Mat& S_atomic);
PetscErrorCode _construct_H_atomic(const simulation& sim, Mat& S_atomic);
PetscErrorCode _construct_atomic_interaction(const simulation& sim,Mat& H_atomic,Mat& S_atomic, Mat& atomic_left,Mat& atomic_right);
PetscErrorCode _construct_z_interaction(const simulation& sim, Mat& H_z);
PetscErrorCode solve_tdse(const simulation& sim, int rank);
};