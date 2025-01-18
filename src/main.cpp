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
    std::complex<double> result = basis.integrate_matrix_element(1000, 1000,[&](int i, int j, std::complex<double> x) {return basis.overlap_integrand(i, j, x, sim);}, sim);
    end = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Time to integrate %.3f\n",(end-start)*1000);


    






   
    
   
   
    ierr = SlepcFinalize(); CHKERRQ(ierr);
    return 0;
}


