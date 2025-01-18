#include <slepcsys.h>
#include <slepceps.h>
#include <petscmat.h>
#include <petscsys.h>

#include "simulation.h"
#include "bsplines.h"


#include <string>
#include <iostream>

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    ierr = SlepcInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    int rank,size;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);
    
    //Define input file string
    std::string input_file = "input.json";


    simulation sim(input_file);
    sim.save_debug_info(rank);

    bsplines basis;
    basis.save_debug_bsplines(rank,sim);
    






   
    
   
   
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}


