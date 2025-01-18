#include <slepcsys.h>
#include <slepceps.h>
#include <petscmat.h>
#include <petscsys.h>

#include "simulation.h"
#include "bsplines.h"
#include "tise.h"


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

    
    bsplines::save_debug_bsplines(rank,sim);

   

    double start = MPI_Wtime();
    tise::solve_tise(sim);
    double end = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Time to construct overlap matrix %.3f\n",end-start);



    // std::complex<double> value1 = bsplines::integrate_matrix_element(0,0,bsplines::overlap_integrand,sim);
    // PetscPrintf(PETSC_COMM_WORLD,"Overlap matrix element %f %f\n",real(value1),imag(value1));

    // std::complex<double> value2 = bsplines::integrate_matrix_element(99,99,bsplines::overlap_integrand,sim);
    // PetscPrintf(PETSC_COMM_WORLD,"Overlap matrix element %f %f\n",real(value2),imag(value2));

    // Mat S;
    // bsplines::construct_overlap(sim,S,true);
    // bsplines::SaveMatrixToCSV(S,"overlap.csv");
    

    






   
    
   
   
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}


