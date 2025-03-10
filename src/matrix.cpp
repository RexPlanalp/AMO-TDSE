#include "matrix.h"
#include "utility.h"
#include "simulation.h"
#include "bsplines.h"
#include "viewer.h"

////////////////////////////
//  Matrix Baseclass //
////////////////////////////

// Constructor: default

// Destructory: destroy the matrix
PetscMatrix::~PetscMatrix()
{
    MatDestroy(&getMatrix());
};

// Method: get the matrix

Mat& PetscMatrix::getMatrix()
{
    return matrix;
};

void PetscMatrix::saveMatrix(const char* filename)
{
    PetscSaverBinary saver(filename);
    saver.saveMatrix(*this);
}

////////////////////////////
// Radial Matrix Subclass //
////////////////////////////

// Constructor: create a matrix
RadialMatrix::RadialMatrix(const simulation& sim,RadialMatrixType matrixType)
{   
    int n_basis = sim.bspline_params.n_basis;
    int order = sim.bspline_params.order;


    PetscErrorCode ierr;

    switch(matrixType)
    {
        case RadialMatrixType::SEQUENTIAL:
        ierr = MatCreate(PETSC_COMM_SELF, &getMatrix()); checkError(ierr,"Error Creating Matrix");
        ierr = MatSetSizes(getMatrix(), PETSC_DECIDE, PETSC_DECIDE, n_basis, n_basis); checkError(ierr,"Error Setting Matrix Size");
        ierr = MatSetFromOptions(getMatrix()); checkError(ierr,"Error Setting Matrix Options");
        ierr = MatSeqAIJSetPreallocation(getMatrix(), 2*order+1, NULL); checkError(ierr,"Error Preallocating Matrix");
        ierr = MatSetUp(getMatrix()); checkError(ierr,"Error Setting Up Matrix");
       
        local_start = 0;
        local_end = n_basis;
        break;

        case RadialMatrixType::PARALLEL:
        ierr = MatCreate(PETSC_COMM_WORLD, &getMatrix()); checkError(ierr, "Error Creating Matrix");
        ierr = MatSetSizes(getMatrix(), PETSC_DECIDE, PETSC_DECIDE, n_basis, n_basis); checkError(ierr, "Error Setting Matrix Size");
        ierr = MatSetFromOptions(getMatrix()); checkError(ierr, "Error Setting Matrix Options");
        ierr = MatMPIAIJSetPreallocation(getMatrix(), 2*order+1, NULL, 2*order+1, NULL); checkError(ierr, "Error Preallocating Matrix");
        ierr = MatSetUp(getMatrix()); checkError(ierr, "Error Setting Up Matrix");
        ierr = MatGetOwnershipRange(getMatrix(), &local_start, &local_end); checkError(ierr, "Error Getting Ownership Range");
       
        break;
    }
}

// Method: set the integrand function
void RadialMatrix::setIntegrand(radialIntegrand integrand)
{
    integrand_func = integrand;
}

// Method: populate the matrix
void RadialMatrix::populateMatrix(const simulation& sim,ECSMode ecs)
{   
    int n_basis = sim.bspline_params.n_basis;
    int order = sim.bspline_params.order;

    bool use_ecs;
    switch(ecs)
    {
        case ECSMode::ON:
        use_ecs = true;
        break;

        case ECSMode::OFF:
        use_ecs = false;
    }


    PetscErrorCode ierr;
    
    for (int i = local_start; i < local_end; i++) 
    {
        int col_start = std::max(0, i - order + 1);
        int col_end = std::min(n_basis, i + order);

        for (int j = col_start; j < col_end; j++) 
        {
            std::complex<double> result = bsplines::integrate_matrix_element(i, j, integrand_func, sim,use_ecs);
            ierr = MatSetValue(getMatrix(), i, j, result, INSERT_VALUES); checkError(ierr, "Error Setting Matrix Value");
        }
    }

    ierr = MatAssemblyBegin(getMatrix(), MAT_FINAL_ASSEMBLY); checkError(ierr, "Error Beginning Assembly");
    ierr = MatAssemblyEnd(getMatrix(), MAT_FINAL_ASSEMBLY); checkError(ierr, "Error Ending Assembly");
}