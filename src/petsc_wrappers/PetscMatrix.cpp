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

    comm = other.comm;
    local_start = other.local_start;
    local_end = other.local_end;
}

PetscMatrix::PetscMatrix(int size, int nnz, RunMode type) 
{
    PetscErrorCode ierr;

    switch(type)
    {
        case RunMode::SEQUENTIAL:
            comm = PETSC_COMM_SELF;

            ierr = MatCreate(comm, &matrix); checkErr(ierr,"Error Creating Matrix");
            ierr = MatSetSizes(matrix,PETSC_DECIDE,PETSC_DECIDE,size,size); checkErr(ierr,"Error Setting Matrix Size");
            ierr = MatSetFromOptions(matrix); checkErr(ierr,"Error Setting Matrix Options");
            ierr = MatSeqAIJSetPreallocation(matrix,nnz,NULL); checkErr(ierr,"Error Preallocating Matrix");
            ierr = MatSetUp(matrix); checkErr(ierr,"Error Setting Up Matrix");

            local_start = 0;
            local_end = size;
            break;

        case RunMode::PARALLEL:
            comm = PETSC_COMM_WORLD;

            ierr = MatCreate(comm,&matrix); checkErr(ierr,"Error Creating Matrix");
            ierr = MatSetSizes(matrix,PETSC_DECIDE,PETSC_DECIDE,size,size); checkErr(ierr,"Error Setting Matrix Size");
            ierr = MatSetFromOptions(matrix); checkErr(ierr,"Error Setting Matrix Options");
            ierr = MatMPIAIJSetPreallocation(matrix,nnz,NULL,nnz,NULL); checkErr(ierr,"Error Preallocating Matrix");
            ierr = MatSetUp(matrix); checkErr(ierr,"Error Setting Up Matrix");
            ierr = MatGetOwnershipRange(matrix,&local_start,&local_end); checkErr(ierr,"Error Getting Ownership Range");
            break;
    }
}

PetscMatrix& PetscMatrix::operator=(const PetscMatrix& other)
{
    if (this != &other)  
    {
        Mat tempMatrix;
        PetscErrorCode ierr = MatDuplicate(other.matrix, MAT_COPY_VALUES, &tempMatrix);
        checkErr(ierr, "Error duplicating matrix");

        if (matrix) {
            MatDestroy(&matrix);
        }
        matrix = tempMatrix;

        comm = other.comm;
        local_start = other.local_start;
        local_end = other.local_end;
    }
    return *this;
}


PetscMatrix::~PetscMatrix()
{   

    if (matrix) 
    {
        PetscErrorCode ierr;
        ierr = MatDestroy(&matrix); checkErr(ierr,"Error Destroying Matrix");
        matrix = nullptr;
    }
    
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

RadialMatrix::RadialMatrix(const simulation& sim, RunMode run, ECSMode ecs)
    : PetscMatrix(sim.bspline_params.n_basis, 2*sim.bspline_params.degree+1, run), ecs(ecs)
{

}

void RadialMatrix::populateMatrix(const simulation& sim, radialElement integrand)
{   
    int n_basis = sim.bspline_params.n_basis;
    int order = sim.bspline_params.order;


    PetscErrorCode ierr;

    bool use_ecs = (ecs == ECSMode::ON);
    
    for (int i = local_start; i < local_end; i++) 
    {
        int col_start = std::max(0, i - order + 1);
        int col_end = std::min(n_basis, i + order);

        for (int j = col_start; j < col_end; j++) 
        {
            std::complex<double> result = bsplines::integrate_matrix_element(i, j, integrand, sim,use_ecs);
            ierr = MatSetValue(matrix, i, j, result, INSERT_VALUES); checkErr(ierr, "Error Setting Matrix Value");
        }
    }
    assemble();
}

//////////////////////////
//    Angular Subclass  //
//////////////////////////



AngularMatrix::AngularMatrix(const simulation& sim, RunMode run, AngularMatrixType type)
    : PetscMatrix(sim.angular_params.n_blocks, 2, run), type(type)
{

}

void AngularMatrix::populateMatrix(const simulation& sim)
{

    int n_blocks = sim.angular_params.n_blocks;

    PetscErrorCode ierr;
    for (int i = local_start; i < local_end; ++i)
        {
            std::pair<int,int> lm_pair = sim.angular_params.block_to_lm.at(i);
            int l = lm_pair.first;
            int m = lm_pair.second;
            for (int j = 0; j < n_blocks; ++j)
            {
                std::pair<int,int> lm_pair_prime = sim.angular_params.block_to_lm.at(j);
                int lprime = lm_pair_prime.first;
                int mprime = lm_pair_prime.second;

                switch(type)
                {
                    case AngularMatrixType::Z_INT_1:
                        if ((l == lprime+1) && (m == mprime))
                        {
                            ierr = MatSetValue(matrix, i, j, -PETSC_i * g(l,m), INSERT_VALUES); checkErr(ierr,"Error Setting Matrix Value");
                        }
                        else if ((l == lprime-1)&&(m == mprime))
                        {
                            ierr = MatSetValue(matrix, i, j, -PETSC_i * f(l,m), INSERT_VALUES); checkErr(ierr,"Error Setting Matrix Value");
                        }
                    break;
                    case AngularMatrixType::Z_INT_2:
                    if ((l == lprime+1) && (m == mprime))
                        {
                            ierr = MatSetValue(matrix, i, j, -PETSC_i * g(l,m) * (-l), INSERT_VALUES); checkErr(ierr,"Error Setting Matrix Value");
                        }
                        else if ((l == lprime-1)&&(m == mprime))
                        {
                            ierr = MatSetValue(matrix, i, j, -PETSC_i * f(l,m) * (l+1), INSERT_VALUES); checkErr(ierr,"Error Setting Matrix Value");
                        }
                }  
            }
        }
    assemble();
}






