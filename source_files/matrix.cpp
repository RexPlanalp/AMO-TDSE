#include "matrix.h"
#include <iostream>
#include "bsplines.h"


void matrix::set_quantum_type(std::string quantum_type)
{
    quantum_type = quantum_type;
}

void matrix::set_type(std::string type)
{
    matrix_type = type;
}

void matrix::set_nnz(int nnz)
{
    nnz = nnz;
}

std::array<int,2> matrix::get_local_range()
{
    return local_range;
}



PetscErrorCode matrix::setup_matrix(bsplines basis)
{
    if (quantum_type == "angular")
    {
        int dim = basis.angular_data["n_blocks"].get<int>();
    }
    else if (quantum_type == "radial")
    {
        int dim = basis.bspline_data["n_basis"].get<int>();
    }
    else
    {
        std::cout << "Error: quantum type not recognized" << std::endl;
        exit(1);
    }


    status = MatCreate(PETSC_COMM_WORLD, &petsc_mat); 
    CHKERRQ(status);

    status = MatSetSizes(petsc_mat,PETSC_DECIDE,PETSC_DECIDE,dim,dim);
    CHKERRQ(status);

    status = MatSetFromOptions(petsc_mat);
    CHKERRQ(status);

    if (matrix_type == "seq")
    {
        status = MatSeqAIJSetPreallocation(petsc_mat,nnz,NULL);
        CHKERRQ(status);
    }
    else if (matrix_type == "aij")
    {
        status = MatMPIAIJSetPreallocation(petsc_mat,nnz,NULL,nnz,NULL);
        CHKERRQ(status);
    }

    status = MatSetUp(petsc_mat);
    CHKERRQ(status);

    int start_row,end_row;
    status = MatGetOwnershipRange(petsc_mat,&start_row,&end_row);
    CHKERRQ(status);

    local_range = {start_row,end_row};

    return status;  
}

