#include <slepcsys.h>
#include <slepceps.h>
#include <petscmat.h>
#include <petscsys.h>

#include "data.h"
#include "bsplines.h"

#include <string>

int main(int argc, char **argv) {
    PetscErrorCode ierr;
    ierr = SlepcInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);
    int rank,size;
    MPI_Comm_rank(PETSC_COMM_WORLD,&rank);

    // Define input file string
    std::string input_file = "input.json";


    // data sim_data(input_file);
    // sim_data.process_data();
    // sim_data.save_debug_info(rank);

    bsplines basis(input_file);
    basis.process_data();
    basis.setup_bsplines();
    
    basis.save_debug_info(rank);
    basis.save_debug_info_bsplines(rank);



    
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}
