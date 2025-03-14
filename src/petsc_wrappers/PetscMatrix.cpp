#include "petsc_wrappers/PetscMatrix.h" 

#include "utility.h"
#include "simulation.h"
#include "bsplines.h"
#include "mpi.h"

//////////////////////////
// Petsc Matrix Wrapper //
//////////////////////////

PetscMatrix::PetscMatrix(const PetscMatrix& other)
{   
    PetscErrorCode ierr;
    ierr = MatDuplicate(other.matrix, MAT_COPY_VALUES, &matrix); checkErr(ierr,"Error Copying Matrix"); 
}

PetscMatrix& PetscMatrix::operator=(const PetscMatrix& other)
{
    if (this != &other)  
    {
        if (matrix) {
            MatDestroy(&matrix);
        }
        MatDuplicate(other.matrix,MAT_COPY_VALUES, &matrix);
    }
    return *this;
}



PetscMatrix::~PetscMatrix()
{
    PetscErrorCode ierr;
    ierr = MatDestroy(&matrix); checkErr(ierr,"Error Destroying Matrix");
}

void PetscMatrix::assemble()
{
    PetscErrorCode ierr;
    ierr = MatAssemblyBegin(matrix, MAT_FINAL_ASSEMBLY); checkErr(ierr,"Error Assembling Matrix");
    ierr = MatAssemblyEnd(matrix, MAT_FINAL_ASSEMBLY); checkErr(ierr,"Error Assembling Matrix");
}


//////////////////////////
//    Radial Subclass   //
//////////////////////////

RadialMatrix::RadialMatrix(const simulation& sim, RunMode type) 
{
    int n_basis = sim.bspline_params.n_basis;
    int order = sim.bspline_params.order;

    PetscErrorCode ierr;

    switch(type)
    {
        case RunMode::SEQUENTIAL:
            comm = PETSC_COMM_SELF;


            ierr = MatCreate(comm, &matrix); checkErr(ierr,"Error Creating Matrix");
            ierr = MatSetSizes(matrix,PETSC_DECIDE,PETSC_DECIDE,n_basis,n_basis); checkErr(ierr,"Error Setting Matrix Size");
            ierr = MatSetFromOptions(matrix); checkErr(ierr,"Error Setting Matrix Options");
            ierr = MatSeqAIJSetPreallocation(matrix,2*order+1,NULL); checkErr(ierr,"Error Preallocating Matrix");
            ierr = MatSetUp(matrix); checkErr(ierr,"Error Setting Up Matrix");

            local_start = 0;
            local_end = n_basis;
            break;

        case RunMode::PARALLEL:
            comm = PETSC_COMM_WORLD;

            ierr = MatCreate(comm,&matrix); checkErr(ierr,"Error Creating Matrix");
            ierr = MatSetSizes(matrix,PETSC_DECIDE,PETSC_DECIDE,n_basis,n_basis); checkErr(ierr,"Error Setting Matrix Size");
            ierr = MatSetFromOptions(matrix); checkErr(ierr,"Error Setting Matrix Options");
            ierr = MatMPIAIJSetPreallocation(matrix,2*order+1,NULL,2*order+1,NULL); checkErr(ierr,"Error Preallocating Matrix");
            ierr = MatSetUp(matrix); checkErr(ierr,"Error Setting Up Matrix");
            ierr = MatGetOwnershipRange(matrix,&local_start,&local_end); checkErr(ierr,"Error Getting Ownership Range");
            break;
    }
}

void RadialMatrix::bindElement(radialElement input_element)
{
    element = input_element;
}

void RadialMatrix::populateMatrix(const simulation& sim,ECSMode ecs)
{   
    int n_basis = sim.bspline_params.n_basis;
    int order = sim.bspline_params.order;

    bool use_ecs = false;
    switch(ecs)
    {
        case ECSMode::ON:
        use_ecs = true;
        break;

        case ECSMode::OFF:
        use_ecs = false;
        break;
    }


    PetscErrorCode ierr;
    
    for (int i = local_start; i < local_end; i++) 
    {
        int col_start = std::max(0, i - order + 1);
        int col_end = std::min(n_basis, i + order);

        for (int j = col_start; j < col_end; j++) 
        {
            std::complex<double> result = bsplines::integrate_matrix_element(i, j, element, sim,use_ecs);
            ierr = MatSetValue(matrix, i, j, result, INSERT_VALUES); checkErr(ierr, "Error Setting Matrix Value");
        }
    }
}

//////////////////////////
//    Angular Subclass  //
//////////////////////////






