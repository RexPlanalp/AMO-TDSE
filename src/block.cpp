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
#include "matrix.h"





namespace block 
{   
    PetscErrorCode load_final_state(std::string filename, Vec* state, int total_size) 
    {   
        PetscErrorCode ierr;
        PetscViewer viewer;

        ierr = VecCreate(PETSC_COMM_SELF, state); CHKERRQ(ierr);
        ierr = VecSetSizes(*state, PETSC_DECIDE, total_size); CHKERRQ(ierr);
        ierr = VecSetFromOptions(*state); CHKERRQ(ierr);
        ierr = VecSetType(*state, VECMPI); CHKERRQ(ierr);
        ierr = VecSet(*state, 0.0); CHKERRQ(ierr);


        ierr = PetscObjectSetName((PetscObject)*state, "final_state"); CHKERRQ(ierr);
        ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, filename.c_str(), FILE_MODE_READ, &viewer); CHKERRQ(ierr);
        ierr = VecLoad(*state, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);


        return ierr;
    }

    PetscErrorCode project_out_bound(std::string filename, RadialMatrix& S, Vec& state, const simulation& sim)
    {
        PetscErrorCode ierr;
        Vec state_block, tise_state,temp;
        IS is;
        std::complex<double> inner_product;
        PetscBool has_dataset;
        PetscViewer viewer;

        // Open HDF5 file for reading
        ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, filename.c_str(), FILE_MODE_READ, &viewer); CHKERRQ(ierr);

        const char GROUP_PATH[] = "/eigenvectors";  // Path to the datasets

        for (int idx = 0; idx < sim.angular_params.n_blocks; ++idx)
        {
            std::pair<int, int> lm_pair = sim.angular_params.block_to_lm.at(idx);
            int l = lm_pair.first;
            

            int start = idx * sim.bspline_params.n_basis;
            ierr = ISCreateStride(PETSC_COMM_SELF, sim.bspline_params.n_basis, start, 1, &is); CHKERRQ(ierr);
            ierr = VecGetSubVector(state, is, &state_block); CHKERRQ(ierr);
            ierr = VecDuplicate(state_block, &temp); CHKERRQ(ierr);

          

            for (int n = 0; n <= sim.angular_params.nmax; ++n)
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

                    ierr = MatMult(S.getMatrix(),state_block,temp); CHKERRQ(ierr);
                    ierr = VecDot(temp,tise_state,&inner_product); CHKERRQ(ierr);
                    ierr = VecAXPY(state_block,-inner_product,tise_state); CHKERRQ(ierr); // Subtract projection 
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

    PetscErrorCode compute_probabilities(const simulation& sim)
    {
        PetscErrorCode ierr;

        std::ofstream file(sim.block_output_path+"/block_norms.txt");
        file << std::fixed << std::setprecision(15);

        std::cout << "Constructing Overlap Matrix" << std::endl;
        RadialMatrix S(sim,RadialMatrixType::SEQUENTIAL);
        S.setIntegrand(bsplines::overlap_integrand);
        S.populateMatrix(sim,ECSMode::OFF);

        std::cout << "Loading Final State" << std::endl;
        Vec state;
        ierr = load_final_state(sim.tdse_output_path+"/tdse_output.h5", &state, sim.angular_params.n_blocks*sim.bspline_params.n_basis); CHKERRQ(ierr);

        if (sim.observable_params.cont)
        {
            ierr = project_out_bound(sim.tise_output_path+"/tise_output.h5", S, state,sim); CHKERRQ(ierr);
        }
        
        Vec state_block,temp;
        std::complex<double> block_norm;
        IS is;
        for (int idx = 0; idx<sim.angular_params.n_blocks; idx++)
        {   
            std::cout << "Computing norm for block " << idx << std::endl;   
            
           
            int start = idx*sim.bspline_params.n_basis;
            ierr = ISCreateStride(PETSC_COMM_SELF, sim.bspline_params.n_basis, start, 1, &is); CHKERRQ(ierr);
            ierr = VecGetSubVector(state, is,&state_block); CHKERRQ(ierr); CHKERRQ(ierr);
            ierr = VecDuplicate(state_block,&temp); CHKERRQ(ierr);
            ierr = MatMult(S.getMatrix(),state_block,temp); CHKERRQ(ierr);
            ierr = VecDot(state_block,temp,&block_norm); CHKERRQ(ierr);
            file << block_norm.real() << " " << block_norm.imag() << "\n";
            ierr = VecRestoreSubVector(state, is, &state_block); CHKERRQ(ierr);

        }
        file.close();

        std::ofstream map_file(sim.block_output_path+"/lm_to_block.txt");
        if (!map_file.is_open())
        {   
            throw std::runtime_error(std::string("Unable to open file for writing: ") + sim.block_output_path + "/lm_to_block.txt");
        }

        for (const auto& pair : sim.angular_params.lm_to_block)
        {
            map_file << pair.first.first << " " << pair.first.second << " " 
                    << pair.second << "\n";
        }
        map_file.close();

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

        create_directory(rank,sim.block_output_path);

        PetscErrorCode ierr;
        std::cout << "Computing Distribution" << std::endl;
        ierr = compute_probabilities(sim); CHKERRQ(ierr);
        return ierr;
    }

}