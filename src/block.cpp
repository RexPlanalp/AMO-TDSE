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
    struct block_context 
    {
        int n_basis;
        int n_blocks;
        int nmax; 
        int CONT;
        std::map<int, std::pair<int, int>> block_to_lm;

        

        static block_context set_config(const simulation& sim) 
        {   
            try {
                block_context config;
                config.n_blocks= sim.angular_data.at("n_blocks").get<int>();
                config.n_basis = sim.bspline_data.at("n_basis").get<int>();
                config.nmax= sim.angular_data.at("nmax").get<int>();
                config.block_to_lm = sim.block_to_lm;
                config.CONT = sim.observable_data.at("CONT").get<int>();

                return config;
            }
            catch (std::exception& e)
            {
                std::cerr << "Error in setting up Photoelectron Spectra context: " << "\n\n" << e.what() << "\n\n";
                throw;
            }
        }
    };

    struct block_filepaths
    {
        static constexpr const char* tdse_output = "TDSE_files/tdse_output.h5";
        static constexpr const char* tise_output = "TISE_files/tise_output.h5";
    };

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
        ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, filename, FILE_MODE_READ, &viewer); CHKERRQ(ierr);
        ierr = VecLoad(*state, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);


        return ierr;
    }

    PetscErrorCode project_out_bound(const char* filename, Mat& S, Vec& state,const block_context& config)
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

        for (int idx = 0; idx < config.n_blocks; ++idx)
        {
            std::pair<int, int> lm_pair = config.block_to_lm.at(idx);
            int l = lm_pair.first;
            

            int start = idx * config.n_basis;
            ierr = ISCreateStride(PETSC_COMM_SELF, config.n_basis, start, 1, &is); CHKERRQ(ierr);
            ierr = VecGetSubVector(state, is, &state_block); CHKERRQ(ierr);
            ierr = VecDuplicate(state_block, &temp); CHKERRQ(ierr);

          

            for (int n = 0; n <= config.nmax; ++n)
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

    PetscErrorCode compute_probabilities(const block_context& config,const block_filepaths& filepaths, const simulation& sim)
    {
        PetscErrorCode ierr;

        std::ofstream file("BLOCK_files/block_norms.txt");
        file << std::fixed << std::setprecision(15);

        std::cout << "Constructing Overlap Matrix" << std::endl;
        Mat S;
        ierr = bsplines::construct_matrix(sim,S,bsplines::overlap_integrand,false,false); CHKERRQ(ierr);

        std::cout << "Loading Final State" << std::endl;
        Vec state;
        ierr = load_final_state(filepaths.tdse_output, &state, config.n_blocks*config.n_basis); CHKERRQ(ierr);

        if (config.CONT)
        {
            ierr = project_out_bound(filepaths.tise_output, S, state, config); CHKERRQ(ierr);
        }
        
        Vec state_block,temp;
        std::complex<double> block_norm;
        IS is;
        for (int idx = 0; idx<config.n_blocks; ++idx)
        {   
            std::cout << "Computing norm for block " << idx << std::endl;   
            
           
            int start = idx*config.n_basis;
            ierr = ISCreateStride(PETSC_COMM_SELF, config.n_basis, start, 1, &is); CHKERRQ(ierr);
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

        create_directory(rank,"BLOCK_files");

        block_context config = block_context::set_config(sim);
        block_filepaths filepaths = block_filepaths();

        PetscErrorCode ierr;
        std::cout << "Computing Distribution" << std::endl;
        ierr = compute_probabilities(config, filepaths,sim); CHKERRQ(ierr);
        return ierr;
    }

}