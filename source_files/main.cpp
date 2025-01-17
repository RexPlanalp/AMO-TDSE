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
    std::cout << "TEST" << std::endl;
    PetscErrorCode ierr;
    ierr = SlepcInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    int rank,size;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    std::cout << "TEST" << std::endl;

    // Define input file string
    // std::string input_file = "input.json";


    // bsplines basis(input_file);
    
    // basis.save_debug_info(rank);
    // basis.save_debug_info_bsplines(rank);

    // tise TISE(input_file);
    // TISE.save_debug_info(rank);



    // matrix S = TISE.compute_overlap_matrix("mpi",basis);

   
  
    Mat S;
    MatCreate(PETSC_COMM_WORLD, &S); 
    MatSetSizes(S,PETSC_DECIDE,PETSC_DECIDE,100000,100000);
    MatSetFromOptions(S);
    MatMPIAIJSetPreallocation(S,1,NULL,1,NULL);
    MatSetUp(S);
    int start_row,end_row;
    double start = MPI_Wtime();
    MatGetOwnershipRange(S,&start_row,&end_row);
    for (int i=start_row; i<end_row; i++) {
        MatSetValue(S,i,i,1.0,INSERT_VALUES);
    }
 
    MatAssemblyBegin(S,MAT_FINAL_ASSEMBLY);
    MatAssemblyEnd(S,MAT_FINAL_ASSEMBLY);
    double end = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Time taken: %f\n",end-start);

 

    

    
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}
