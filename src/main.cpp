// #include <slepcsys.h>
// #include <slepceps.h>
// #include <petscmat.h>
// #include <petscsys.h>

// #include "simulation.h"
// #include "bsplines.h"
// #include "laser.h"  
// #include "tise.h"
// #include "tdse.h"
// #include "block.h"
// #include <petscviewerhdf5.h>


// #include <string>
// #include <iostream>

// int main(int argc, char **argv) {
//     PetscErrorCode ierr;
//     ierr = SlepcInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
//     int rank,size;
//     MPI_Comm_rank(PETSC_COMM_WORLD,&rank);
//     MPI_Comm_size(PETSC_COMM_WORLD,&size);

//     std::string input_file = "input.json";

//     simulation sim(input_file);

//     sim.save_debug_info(rank);
//     bsplines::save_debug_bsplines(rank,sim);
//     laser::save_debug_laser(rank,sim);


//     // ierr = tise::solve_tise(sim,rank); CHKERRQ(ierr);
//     // ierr = tise::prepare_matrices(sim,rank); CHKERRQ(ierr);

//     // ierr = tdse::solve_tdse(sim,rank); CHKERRQ(ierr);

//     ierr = block::compute_block_distribution(rank,sim); CHKERRQ(ierr);


//     ierr = SlepcFinalize(); CHKERRQ(ierr);
//     return 0;
// }

#include <slepcsys.h>
#include <slepceps.h>
#include <petscmat.h>
#include <petscsys.h>

#include "simulation.h"
#include "bsplines.h"
#include "laser.h"
#include "tise.h"
#include "tdse.h"
#include "block.h"
#include <petscviewerhdf5.h>

#include <string>
#include <iostream>

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    ierr = SlepcInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    
    int rank, size;
    MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
    MPI_Comm_size(PETSC_COMM_WORLD, &size);

    std::string input_file = "input.json";
    simulation sim(input_file);

    sim.save_debug_info(rank);
    bsplines::save_debug_bsplines(rank, sim);
    laser::save_debug_laser(rank, sim);

    // Check for command-line arguments
    bool run_tise = false;
    bool run_tdse = false;
    bool run_block = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--tise") {
            run_tise = true;
        } else if (arg == "--tdse") {
            run_tdse = true;
        } else if (arg == "--block") {
            run_block = true;
        } else if (arg == "--all") {
            run_tise = true;
            run_tdse = true;
            run_block = true;
        }
    }

    // If no flags are provided, default to running everything
    if (!run_tise && !run_tdse && !run_block) {
        run_tise = true;
        run_tdse = true;
        run_block = true;
    }

    // Execute selected computations
    if (run_tise) {
        ierr = tise::solve_tise(sim, rank); CHKERRQ(ierr);
        ierr = tise::prepare_matrices(sim, rank); CHKERRQ(ierr);
    }

    if (run_tdse) {
        ierr = tdse::solve_tdse(sim, rank); CHKERRQ(ierr);
    }

    if (run_block) {
        ierr = block::compute_block_distribution(rank, sim); CHKERRQ(ierr);
    }

    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}
