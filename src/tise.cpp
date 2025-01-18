#include "tise.h"

#include <petscmat.h>
#include "simulation.h"
#include "bsplines.h"
#include "bsplines.h"

PetscErrorCode tise::construct_matrix(const simulation& sim, Mat& M, std::function<std::complex<double>(int, int, std::complex<double>, const simulation&)> integrand)
{
    PetscErrorCode ierr;
    ierr = MatCreate(PETSC_COMM_WORLD, &M); CHKERRQ(ierr);
    ierr = MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, sim.bspline_data.value("n_basis",0), sim.bspline_data.value("n_basis",0)); CHKERRQ(ierr);
    ierr = MatSetFromOptions(M); CHKERRQ(ierr);

    int nnz_per_row = 2 * sim.bspline_data.value("degree",0) + 1;
    ierr = MatMPIAIJSetPreallocation(M, nnz_per_row, NULL, nnz_per_row, NULL); CHKERRQ(ierr);
    ierr = MatSetUp(M); CHKERRQ(ierr);

    int start_row, end_row;
    ierr = MatGetOwnershipRange(M, &start_row, &end_row); CHKERRQ(ierr);

    for (int i = start_row; i < end_row; i++) 
    {
        int col_start = std::max(0, i - sim.bspline_data.value("order",0) + 1);
        int col_end = std::min(sim.bspline_data.value("n_basis",0), i + sim.bspline_data.value("order",0));

        for (int j = col_start; j < col_end; j++) 
        {
            std::complex<double> result = bsplines::integrate_matrix_element(i, j, integrand, sim);
            ierr = MatSetValue(M, i, j, result.real(), INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

    return ierr;
}


PetscErrorCode tise::construct_overlap(const simulation& sim, Mat& S)
{
    return construct_matrix(sim, S, bsplines::overlap_integrand);
}

