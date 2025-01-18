#include "tise.h"

#include <petscmat.h>
#include "simulation.h"
#include "bsplines.h"
#include "bsplines.h"

PetscErrorCode tise::construct_overlap(const simulation& sim, Mat& S)
{
    PetscErrorCode ierr;
    ierr = MatCreate(PETSC_COMM_WORLD, &S); CHKERRQ(ierr);
    ierr = MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, sim.bspline_data.value("n_basis",0), sim.bspline_data.value("n_basis",0)); CHKERRQ(ierr);
    ierr = MatSetFromOptions(S); CHKERRQ(ierr);
    // Preallocate memory for nonzero entries
    int nnz_per_row = 2 * sim.bspline_data.value("degree",0) + 1;
    ierr = MatMPIAIJSetPreallocation(S, nnz_per_row, NULL, nnz_per_row, NULL); CHKERRQ(ierr);
    // Set up the matrix
    ierr = MatSetUp(S); CHKERRQ(ierr);

    // Get the range of rows owned by the current process
    int start_row, end_row;
    ierr = MatGetOwnershipRange(S, &start_row, &end_row); CHKERRQ(ierr);
    for (int i = start_row; i < end_row; i++) 
    {
        int col_start = std::max(0, i - sim.bspline_data.value("order",0) + 1);
        int col_end = std::min(sim.bspline_data.value("n_basis",0), i + sim.bspline_data.value("order",0));

        for (int j = col_start; j < col_end; j++) 
        {
            std::complex<double> result = bsplines::integrate_matrix_element(i, j, bsplines::overlap_integrand, sim);
            ierr = MatSetValue(S, i, j, result.real(), INSERT_VALUES); CHKERRQ(ierr);
        }
    }
ierr = MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
ierr = MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
}