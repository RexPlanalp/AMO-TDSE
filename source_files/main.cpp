#include <slepcsys.h>
#include <slepceps.h>
#include <petscmat.h>
#include <petscsys.h>

#include "data.h"
#include "bsplines.h"
#include "matrix.h"
#include "tise.h"

#include <string>
#include <iostream>

int main(int argc, char **argv) {
    double start = MPI_Wtime();
    PetscErrorCode ierr;
    ierr = SlepcInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    int rank,size;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    double end = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Time taken: %f\n",end-start);

    //Define input file string
    std::string input_file = "input.json";


    bsplines basis(input_file);
    
    basis.save_debug_info(rank);
    basis.save_debug_info_bsplines(rank);

    tise TISE(input_file);
    TISE.save_debug_info(rank);



    matrix S = TISE.compute_overlap_matrix("mpi",basis);

   
    
   
   
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}


