#include <slepcsys.h>
#include <slepceps.h>
#include <petscmat.h>
#include <petscsys.h>

#include "simulation.h"
#include "bsplines.h"
#include "laser.h"  
#include "tise.h"
#include "tdse.h"


#include <string>
#include <iostream>

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    ierr = SlepcInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    int rank,size;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
    MPI_Comm_size(PETSC_COMM_WORLD,&size);

    std::string input_file = "input.json";

    simulation sim(input_file);
    sim.save_debug_info(rank);


    laser::save_debug_laser(rank,sim);
    bsplines::save_debug_bsplines(rank,sim);


    Vec tdse_state;
    ierr = tdse::load_starting_state(sim,tdse_state); CHKERRQ(ierr);

   

    // double start = MPI_Wtime();
    // tise::solve_tise(sim,rank);
    // double end = MPI_Wtime();
    // PetscPrintf(PETSC_COMM_WORLD,"Time to solve TISE %.3f\n",end-start);





    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}

