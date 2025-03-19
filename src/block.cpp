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
    void compute_block_distribution(int rank,const simulation& sim)
    {

        if (rank!=0)
        {
            return;
        }

        std::cout << "Computing Distribution" << std::endl;

        create_directory(rank,sim.block_output_path);
        std::ofstream file(sim.block_output_path+"/block_norms.txt");
        file << std::fixed << std::setprecision(15);

        std::cout << "Constructing Overlap Matrix" << std::endl;
        RadialMatrix S(sim, RunMode::SEQUENTIAL, ECSMode::OFF);
        S.populateMatrix(sim,bsplines::overlap_integrand);

        std::cout << "Loading Final State" << std::endl;
        PetscHDF5Viewer finalStateViewer((sim.tdse_output_path+"/tdse_output.h5").c_str(),RunMode::SEQUENTIAL,OpenMode::READ);
        PetscVector state = finalStateViewer.loadVector(sim.angular_params.n_blocks*sim.bspline_params.n_basis,"","final_state");

        if (sim.observable_params.cont)
        {
            project_out_bound(S, state,sim); 
        }
        
        PetscVector state_block,temp;
        std::complex<double> block_norm;
        for (int block = 0; block<sim.angular_params.n_blocks; block++)
        {   
            std::cout << "Computing norm for block: " << block << std::endl;   
            
            int start = block*sim.bspline_params.n_basis;
            PetscIS indexSet(sim.bspline_params.n_basis, start, 1, RunMode::SEQUENTIAL);

            VecGetSubVector(state.vector, indexSet.is, &state_block.vector); 

            PetscVector temp(state_block);

            MatMult(S.matrix,state_block.vector,temp.vector); 

            VecDot(state_block.vector,temp.vector,&block_norm); 

            file << block_norm.real() << " " << block_norm.imag() << "\n";
            VecRestoreSubVector(state.vector, indexSet.is, &state_block.vector); 

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
    }
}