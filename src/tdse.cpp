#include "tdse.h"
#include "simulation.h"
#include <iostream>
#include <fstream>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>


PetscErrorCode tdse::load_starting_state(const simulation& sim, Vec& tdse_state)
{   
    PetscErrorCode ierr;
    Vec tise_state;
    PetscViewer viewer;

    int n_basis = sim.bspline_data.value("n_basis",0);
    int n_blocks = sim.angular_data.value("n_blocks",0);
    std::array<int,3> state = sim.state_data.value("state",std::array<int,3>{0,0,0});

    ierr = VecCreate(PETSC_COMM_WORLD,&tdse_state); CHKERRQ(ierr);
    ierr = VecSetSizes(tdse_state,PETSC_DECIDE,n_basis*n_blocks); CHKERRQ(ierr);
    ierr = VecSetFromOptions(tdse_state); CHKERRQ(ierr);

    ierr = VecCreate(PETSC_COMM_WORLD,&tise_state); CHKERRQ(ierr);
    ierr = VecSetSizes(tise_state,PETSC_DECIDE,n_basis); CHKERRQ(ierr);
    ierr = VecSetFromOptions(tise_state); CHKERRQ(ierr);

    std::stringstream ss;
    ss << "eigenvectors/psi_" << state[0] << "_" << state[1];
    std::string state_path = ss.str();
    ierr = PetscObjectSetName((PetscObject)tise_state,state_path.c_str()); CHKERRQ(ierr);

    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"TISE_files/tise_output.h5",FILE_MODE_READ,&viewer); CHKERRQ(ierr);
    ierr = VecLoad(tise_state,viewer); CHKERRQ(ierr);
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

    PetscInt start, end;
    ierr = VecGetOwnershipRange(tdse_state, &start, &end); CHKERRQ(ierr);

    const PetscScalar *tise_state_array;
    ierr = VecGetArrayRead(tise_state, &tise_state_array); CHKERRQ(ierr);
    int offset = sim.qn_map.lm_to_block.at({state[1],state[2]}) * n_basis;
    for (int i = 0; i < n_basis; ++i)
    {   
        int global_index = offset + i;
        if (global_index >= start && global_index < end)
        {
            ierr = VecSetValue(tdse_state,global_index,tise_state_array[i],INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = VecRestoreArrayRead(tise_state, &tise_state_array); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(tdse_state); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tdse_state); CHKERRQ(ierr);
    ierr = VecDestroy(&tise_state); CHKERRQ(ierr);
}