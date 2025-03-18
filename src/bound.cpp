#include <petscsys.h>

#include "simulation.h"

#include "petsc_wrappers/PetscFileViewer.h"
#include "petsc_wrappers/PetscMatrix.h"
#include "petsc_wrappers/PetscVector.h"
#include "petsc_wrappers/PetscIS.h"
#include "bsplines.h"


namespace bound
{
    double computeBoundPopulation(int n_bound, int l_bound, const PetscVector& state, const simulation& sim)
    {
        PetscErrorCode ierr;
        PetscBool has_dataset;
        double probability = 0;

        PetscHDF5Viewer viewBoundState((sim.tise_output_path+"/tise_output.h5").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscVector state_block;

        std::cout << "Constructing Overlap Matrix" << std::endl;
        RadialMatrix S(sim, RunMode::SEQUENTIAL, ECSMode::OFF);
        S.populateMatrix(sim,bsplines::overlap_integrand);

        std::ostringstream dataset_ss;
        dataset_ss << "/eigenvectors" << "/psi_" << n_bound << "_" << l_bound;
        std::string dataset_name = dataset_ss.str();


        ierr = PetscViewerHDF5HasDataset(viewBoundState.viewer, dataset_name.c_str(), &has_dataset); checkErr(ierr, "Error in PetscViewerHDF5HasDataset");

        if (!(has_dataset))
        {   
            return 0;
        }
        else
        {
            PetscVector bound_state = viewBoundState.loadVector(sim.bspline_params.n_basis, "eigenvectors", dataset_name.c_str());
            for (int block = 0; block < sim.angular_params.n_blocks; block++)
            {
                std::pair<int, int> lm_pair = sim.angular_params.block_to_lm.at(block);
                int l = lm_pair.first;
                

                if (!(l == l_bound))
                {   
                    continue;
                }

                int start = block * sim.bspline_params.n_basis;
                PetscIS indexSet(sim.bspline_params.n_basis, start, 1, RunMode::SEQUENTIAL); 

                ierr = VecGetSubVector(state.vector, indexSet.is, &state_block.vector);
                PetscVector temp(state_block);

                std::complex<double> inner_product = 0;

                ierr = MatMult(S.matrix,state_block.vector,temp.vector); checkErr(ierr, "Error in MatMult");
                ierr = VecDot(temp.vector, bound_state.vector, &inner_product);
                probability += std::norm(inner_product);

                ierr = VecRestoreSubVector(state.vector, indexSet.is, &state_block.vector);
            }
        }

        
        return probability;

    }

    void computeBoundDistribution(int rank, const simulation& sim)
    {
        if (rank != 0)
        {
            return;
        }

        create_directory(rank,sim.bound_output_path);
        std::ofstream file(sim.bound_output_path+"/bound_pops.txt");
        file << std::fixed << std::setprecision(15);

        PetscHDF5Viewer finalStateViewer((sim.tdse_output_path+"/tdse_output.h5").c_str(),RunMode::SEQUENTIAL,OpenMode::READ);
        PetscVector state = finalStateViewer.loadVector(sim.angular_params.n_blocks*sim.bspline_params.n_basis,"","final_state");

        for (int n = 1; n <= sim.angular_params.nmax; n++)
        {
            for (int l = 0; l < n; l++)
            {   
                std::cout << "Computing Bound Population for n = " << n << " and l = " << l << std::endl;
                double pop = computeBoundPopulation(n, l, state, sim);
                file << n << " " << l << " " << pop << std::endl;
            }
        }
        file.close();
    }
}