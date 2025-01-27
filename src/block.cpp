#include "block.h"
#include <iostream>
#include <fstream>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>
#include "simulation.h"
#include "bsplines.h"
#include <map>
#include <iomanip>





namespace block 
{

    PetscErrorCode load_final_state(const char* filename, Vec* state, int total_size) 
    {   
        PetscErrorCode ierr;
        PetscViewer viewer;

        ierr = VecCreate(PETSC_COMM_SELF, state); CHKERRQ(ierr);
        ierr = VecSetSizes(*state, PETSC_DECIDE, total_size); CHKERRQ(ierr);
        ierr = VecSetFromOptions(*state); CHKERRQ(ierr);
        ierr = VecSetType(*state, VECMPI); CHKERRQ(ierr);
        ierr = VecSet(*state, 0.0); CHKERRQ(ierr);


        ierr = PetscObjectSetName((PetscObject)*state, "final_state"); CHKERRQ(ierr);
        ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, "TDSE_files/tdse_output.h5", FILE_MODE_READ, &viewer); CHKERRQ(ierr);
        ierr = VecLoad(*state, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);


        return ierr;
    }

    PetscErrorCode project_out_bound(const char* filename, Mat& S, Vec& state, int n_basis, int n_blocks, int nmax, std::map<int, std::pair<int, int>>& block_to_lm)
    {
        PetscErrorCode ierr;
        Vec state_block, tise_state,temp;
        IS is;
        std::complex<double> inner_product;
        PetscBool has_dataset;
        PetscViewer viewer;

        // Open HDF5 file for reading
        ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, filename, FILE_MODE_READ, &viewer); CHKERRQ(ierr);

        const char GROUP_PATH[] = "/eigenvectors";  // Path to the datasets

        for (int idx = 0; idx < n_blocks; ++idx)
        {
            std::pair<int, int> lm_pair = block_to_lm.at(idx);
            int l = lm_pair.first;
            int m = lm_pair.second;

            int start = idx * n_basis;
            ierr = ISCreateStride(PETSC_COMM_SELF, n_basis, start, 1, &is); CHKERRQ(ierr);
            ierr = VecGetSubVector(state, is, &state_block); CHKERRQ(ierr);
            ierr = VecDuplicate(state_block, &temp); CHKERRQ(ierr);

          

            for (int n = 0; n <= nmax; ++n)
            {
                std::ostringstream dataset_name;
                dataset_name << GROUP_PATH << "/psi_" << n << "_" << l;
                ierr = PetscViewerHDF5HasDataset(viewer, dataset_name.str().c_str(), &has_dataset); CHKERRQ(ierr);
                if (has_dataset)
                {   
                    ierr = VecDuplicate(state_block, &tise_state); CHKERRQ(ierr);
                    ierr = VecSet(tise_state, 0.0); CHKERRQ(ierr);

                    ierr = PetscObjectSetName((PetscObject)tise_state, dataset_name.str().c_str()); CHKERRQ(ierr);
                    ierr = VecLoad(tise_state, viewer); CHKERRQ(ierr);

                    ierr = MatMult(S,state_block,temp); CHKERRQ(ierr);
                    ierr = VecDot(tise_state,temp,&inner_product); CHKERRQ(ierr);
                    ierr = VecAXPY(state_block,-std::conj(inner_product),tise_state); CHKERRQ(ierr); // Subtract projection 
                }
            }
            
            

            ierr = VecRestoreSubVector(state, is, &state_block); CHKERRQ(ierr);
            ierr = ISDestroy(&is); CHKERRQ(ierr);
            ierr = VecDestroy(&temp); CHKERRQ(ierr);
        }

        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
        ierr = VecDestroy(&tise_state); CHKERRQ(ierr);

        return ierr;
    }

    PetscErrorCode compute_probabilities(Vec& state, Mat& S,int n_blocks,int n_basis, std::map<int,std::pair<int,int>>& block_to_lm)
    {
        PetscErrorCode ierr;
        Vec state_block,temp;
        IS is;
        std::complex<double> block_norm;

        std::ofstream file("TDSE_files/block_norms.txt");
        file << std::fixed << std::setprecision(15);

        for (int idx = 0; idx<n_blocks; ++idx)
        {   
            std::cout << "Computing norm for block " << idx << std::endl;   
            std::pair<int,int> lm_pair = block_to_lm.at(idx);
            int l = lm_pair.first;
            int m = lm_pair.second;

            int start = idx*n_basis;
            ierr = ISCreateStride(PETSC_COMM_SELF, n_basis, start, 1, &is); CHKERRQ(ierr);
            ierr = VecGetSubVector(state, is,&state_block); CHKERRQ(ierr); CHKERRQ(ierr);
            ierr = VecDuplicate(state_block,&temp); CHKERRQ(ierr);
            ierr = MatMult(S,state_block,temp); CHKERRQ(ierr);
            ierr = VecDot(state_block,temp,&block_norm); CHKERRQ(ierr);
            file << block_norm.real() << " " << block_norm.imag() << "\n";
            ierr = VecRestoreSubVector(state, is, &state_block); CHKERRQ(ierr);

        }
        file.close();

        ierr = VecDestroy(&state_block); CHKERRQ(ierr);
        ierr = VecDestroy(&temp); CHKERRQ(ierr);
        ierr = ISDestroy(&is); CHKERRQ(ierr);
        return ierr;
    }

    PetscErrorCode compute_block_distribution(int rank,const simulation& sim)
    {   
        if (rank!=0)
        {
            return 0;
        }
        PetscErrorCode ierr;
        Mat S;
        Vec state;

        int n_blocks = sim.angular_data.at("n_blocks").get<int>();
        int n_basis = sim.bspline_data.at("n_basis").get<int>();  
        int total_size = n_basis*n_blocks;  
        int nmax = sim.angular_data.at("nmax").get<int>();
        std::map<int,std::pair<int,int>> block_to_lm =  sim.block_to_lm;
        
        std::cout << "Constructing Overlap Matrix" << std::endl;
        ierr = bsplines::construct_overlap(sim,S,false,false); CHKERRQ(ierr);

        std::cout << "Loading Final State" << std::endl;
        ierr = load_final_state("TDSE_files/tdse_output.h5", &state, total_size); CHKERRQ(ierr);

        if (sim.observable_data.at("CONT").get<int>())
        {
            ierr = project_out_bound("TISE_files/tise_output.h5", S, state, n_basis, n_blocks, nmax, block_to_lm); CHKERRQ(ierr);
        }

        std::cout << "Computing Distribution" << std::endl;
        ierr = compute_probabilities(state, S, n_blocks, n_basis, block_to_lm); CHKERRQ(ierr);
        return ierr;
    }

}