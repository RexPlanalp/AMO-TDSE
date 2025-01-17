#pragma once

#include <string>
#include <petscmat.h>
#include "bsplines.h"



struct matrix
{

    std::string quantum_type;
    std::string matrix_type;
    PetscErrorCode status;
    std::array<int,2> local_range;
    int nnz;
    int dim;

    void set_quantum_type(std::string quantum_type);
    void set_type(std::string type);
    void set_nnz(int nnz);

    PetscErrorCode setup_matrix(bsplines basis);
    std::array<int,2> get_local_range();
    
    Mat petsc_mat;

};