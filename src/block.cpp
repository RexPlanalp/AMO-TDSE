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
#include "petsc_wrappers/PetscMatrix.h"
#include "petsc_wrappers/PetscFileViewer.h"
#include "petsc_wrappers/PetscVector.h"
#include "petsc_wrappers/PetscIS.h"
#include "bsplines.h"
#include "utility.h"




namespace block 
{   
    PetscErrorCode project_out_bound(const PetscMatrix& S, PetscVector& state, const simulation& sim)
    {
      
    

        PetscErrorCode ierr;
        std::complex<double> inner_product;
        PetscBool has_dataset;
        

        PetscHDF5Viewer viewEigenvectors((sim.tise_output_path+"/tise_output.h5").c_str(),RunMode::SEQUENTIAL,OpenMode::READ);
        PetscVector state_block;


        for (int block = 0; block < sim.angular_params.n_blocks; block++)
        {
            std::pair<int, int> lm_pair = sim.angular_params.block_to_lm.at(block);
            int l = lm_pair.first;
            int start = block * sim.bspline_params.n_basis;

            PetscIS indexSet(sim.bspline_params.n_basis, start, 1, RunMode::SEQUENTIAL);
            
            ierr = VecGetSubVector(state.vector, indexSet.is, &state_block.vector); checkErr(ierr, "Error in VecGetSubVector");
            PetscVector temp(state_block);
            
            for (int n = 0; n <= sim.angular_params.nmax; ++n)
            {
                std::ostringstream dataset_ss;
                dataset_ss << "/eigenvectors" << "/psi_" << n << "_" << l;
                std::string dataset_name = dataset_ss.str();


                ierr = PetscViewerHDF5HasDataset(viewEigenvectors.viewer, dataset_name.c_str(), &has_dataset); checkErr(ierr, "Error in PetscViewerHDF5HasDataset");
                if (has_dataset)
                {   
                    PetscVector tise_state = viewEigenvectors.loadVector(sim.bspline_params.n_basis, "eigenvectors", dataset_name.c_str());

                    ierr = MatMult(S.matrix,state_block.vector,temp.vector); checkErr(ierr, "Error in MatMult");
                    ierr = VecDot(temp.vector,tise_state.vector,&inner_product); checkErr(ierr, "Error in VecDot");
                    ierr = VecAXPY(state_block.vector,-inner_product,tise_state.vector); checkErr(ierr, "Error in VecAXPY");
                }
            }
            
            

            ierr = VecRestoreSubVector(state.vector, indexSet.is, &state_block.vector); checkErr(ierr, "Error in VecRestoreSubVector");
        }
        return ierr;
    }

    PetscErrorCode compute_probabilities(const simulation& sim)
    {
        PetscErrorCode ierr;

        std::ofstream file(sim.block_output_path+"/block_norms.txt");
        file << std::fixed << std::setprecision(15);

        std::cout << "Constructing Overlap Matrix" << std::endl;
        RadialMatrix S(sim, RunMode::SEQUENTIAL, ECSMode::OFF);
        S.populateMatrix(sim,bsplines::overlap_integrand);

        std::cout << "Loading Final State" << std::endl;
        //Vec state;
        //ierr = load_final_state(sim.tdse_output_path+"/tdse_output.h5", &state, sim.angular_params.n_blocks*sim.bspline_params.n_basis); CHKERRQ(ierr);
        PetscHDF5Viewer finalStateViewer((sim.tdse_output_path+"/tdse_output.h5").c_str(),RunMode::SEQUENTIAL,OpenMode::READ);
        PetscVector state = finalStateViewer.loadVector(sim.angular_params.n_blocks*sim.bspline_params.n_basis,"","final_state");

        if (sim.observable_params.cont)
        {
            ierr = project_out_bound(S, state,sim); checkErr(ierr, "Error in project_out_bound");
        }
        
        Vec state_block,temp;
        std::complex<double> block_norm;
        IS is;
        for (int idx = 0; idx<sim.angular_params.n_blocks; idx++)
        {   
            std::cout << "Computing norm for block " << idx << std::endl;   
            
           
            int start = idx*sim.bspline_params.n_basis;
            ierr = ISCreateStride(PETSC_COMM_SELF, sim.bspline_params.n_basis, start, 1, &is); CHKERRQ(ierr);
            ierr = VecGetSubVector(state.vector, is,&state_block); CHKERRQ(ierr); CHKERRQ(ierr);
            ierr = VecDuplicate(state_block,&temp); CHKERRQ(ierr);
            ierr = MatMult(S.matrix,state_block,temp); CHKERRQ(ierr);
            ierr = VecDot(state_block,temp,&block_norm); CHKERRQ(ierr);
            file << block_norm.real() << " " << block_norm.imag() << "\n";
            ierr = VecRestoreSubVector(state.vector, is, &state_block); CHKERRQ(ierr);

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