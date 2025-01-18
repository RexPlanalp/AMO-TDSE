#include <slepcsys.h>
#include <slepceps.h>
#include <petscmat.h>
#include <petscsys.h>

#include "simulation.h"
#include "bsplines.h"


#include <string>
#include <iostream>

int main(int argc, char **argv) {
    double start = MPI_Wtime();

    PetscErrorCode ierr;
    ierr = SlepcInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    int rank,size;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);

    double end = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Time to initialize petsc %.3f\n",end-start);

    std::string input_file = "input.json";


    simulation sim(input_file);
    sim.save_debug_info(rank);

    bsplines basis;
    basis.save_debug_bsplines(rank,sim);
     start = MPI_Wtime(); 
   Mat S;
ierr = MatCreate(PETSC_COMM_WORLD, &S); CHKERRQ(ierr);
ierr = MatSetSizes(S, PETSC_DECIDE, PETSC_DECIDE, 
                   sim.bspline_data.value("n_basis",0), 
                   sim.bspline_data.value("n_basis",0)); CHKERRQ(ierr);
ierr = MatSetFromOptions(S); CHKERRQ(ierr);

// Preallocate memory for nonzero entries
PetscInt nnz_per_row = 2 * sim.bspline_data.value("degree",0) + 1;
ierr = MatMPIAIJSetPreallocation(S, nnz_per_row, NULL, nnz_per_row, NULL); CHKERRQ(ierr);

// Set up the matrix
ierr = MatSetUp(S); CHKERRQ(ierr);

// Get the range of rows owned by the current process
PetscInt start_row, end_row;
ierr = MatGetOwnershipRange(S, &start_row, &end_row); CHKERRQ(ierr);

// Iterate over locally owned rows
for (PetscInt i = start_row; i < end_row; i++) {
    PetscInt col_start = std::max(static_cast<PetscInt>(0), i - sim.bspline_data.value("order",0) + 1);
    PetscInt col_end = std::min(sim.bspline_data.value("n_basis",0), i + sim.bspline_data.value("order",0));

    for (PetscInt j = col_start; j < col_end; j++) {
        // Compute the matrix element
        std::complex<double> result = basis.integrate_matrix_element(
            i, j,
            [&](int i, int j, std::complex<double> x) { return basis.overlap_integrand(i, j, x, sim); },
            sim
        );

        // Insert only the real part (or full complex value if PETSc supports complex numbers)
        ierr = MatSetValue(S, i, j, result.real(), INSERT_VALUES); CHKERRQ(ierr);
    }
}

// Assemble the matrix
ierr = MatAssemblyBegin(S, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
ierr = MatAssemblyEnd(S, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


   
    end = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Time to construct %.3f\n",(end-start));


    






   
    
   
   
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}


