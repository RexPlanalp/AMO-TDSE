#include <iostream>
#include <fstream>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>
#include "simulation.h"
#include "bsplines.h"
#include <map>
#include <iomanip>
#include <algorithm>
#include <complex>





namespace pes
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

    double H(double r)
    {
        return -1.0/(r+1E-25);
    }

    void scale_vector(std::vector<double>& vec, double scale)
    {
        for (double& val : vec)
        {
            val *= scale;
        }
    }

    struct CoulombResult 
    {
        double phase;
        std::vector<double> wave;
    };

    CoulombResult compute_coulomb_wave(double E, int l, int Nr, double dr)
    {
        std::vector<double> wave(Nr, 0.0);

        double dr2  = dr * dr;
        double k    = std::sqrt(2.0 * E);
        int   lterm = l * (l + 1);

        wave[0] = 1.0;

        
        double r1    = dr;  
        double term1 = dr2 * (lterm/(r1*r1) + 2.0*H(r1) - 2.0*E);
        wave[1]      = (term1 + 2.0) * wave[0];

    
        for (int idx = 2; idx < Nr; ++idx)
        {
            double r_val = idx*dr; 
            double term   = dr2 * (lterm/(r_val*r_val) + 2.0*H(r_val) - 2.0*E);

            wave[idx] = wave[idx - 1] * (term + 2.0) - wave[idx - 2];

            if (std::abs(wave[idx]) > 1e10)
            {
                scale_vector(wave, 1e-10);  
            }
        }

        double r_end     = (Nr - 2)*dr;
        double wave_end  = wave[Nr - 2];
        double dwave_end = (wave[Nr - 1] - wave[Nr - 3]) / (2.0 * dr);


        double denom   = (k + 1.0 / (k * r_end));
        double termPsi = wave_end * wave_end;          
        double termDer = (dwave_end / denom)*(dwave_end / denom); 
        double normVal = std::sqrt(termPsi + termDer);

        if (normVal != 0.0)
        {
            for (auto& val : wave) {
                val /= normVal;
            }
        }

        std::complex<double> numerator(0.0, wave_end); 
        numerator += dwave_end / denom;  

        double scale = 2.0*k*r_end;
        double ln_s  = std::log(scale);
        std::complex<double> denomC  = std::exp(std::complex<double>(0.0, 1.0/k) * ln_s ); 

        std::complex<double> fraction = numerator / denomC;
        double phase  = std::arg(fraction) - k*r_end + l*M_PI/2.0;

        return CoulombResult{phase, wave};

    }
    
    PetscErrorCode expand_state(Vec& state,std::vector<std::complex<double>>& expanded_state,int Nr, int n_blocks,int n_basis, int degree, double dr, const std::vector<std::complex<double>>& knots, std::map<int, std::pair<int, int>>& block_to_lm)
    {   
        PetscErrorCode ierr;
        const std::complex<double>* array;
        ierr =  VecGetArrayRead(state, reinterpret_cast<const PetscScalar**>(&array)); CHKERRQ(ierr);

        for (int idx = 0; idx < n_basis; ++idx)
        {
            std::complex<double> start = knots[idx];
            std::complex<double> end = knots[idx+degree+1];

            std::vector<std::complex<double>> basis_eval;
            std::vector<int> basis_indices;

            for (int i = 0; i < Nr; ++i)
            {   
                
                std::complex<double> r = i*dr;
                if (r.real() >= start.real() && r.real() < end.real())
                {
                    std::complex<double> val = bsplines::B(idx,degree,r,knots);
                    basis_eval.push_back(val);
                    basis_indices.push_back(i);
                }
            }

            for (int block = 0; block < n_blocks; ++block)
            {   
                int global_idx = block*n_basis + idx;
                std::complex<double> coeff = array[global_idx];
                for (int index = 0; index < basis_eval.size(); ++index)
                {
                    expanded_state[block*Nr + basis_indices[index]] += coeff*basis_eval[index];
                }
            }
        }
    }

    int compute_pes(int rank,const simulation& sim)
    {   
        if (rank!=0)
        {
            return 0;
        }

        // double E = 0.5;
        // int l = 0;
        // int Nr = 1000;
        // double dr = 0.01;
        // int n_blocks = 1;

        // CoulombResult result = compute_coulomb_wave(E, l, Nr, dr);
        // std::cout << "Phase: " << result.phase << std::endl;

        // std::ofstream outFile("wave.txt");
        // if (!outFile.is_open())
        // {
        //     std::cerr << "couldnt open wave.txt";
        //     return 1;
        // }

        // for (int idx = 0; idx < Nr; ++idx)
        // {
        //     double rVal = idx*dr;
        //     outFile << rVal << " " << result.wave[idx] << "\n";
        // }

        // outFile.close();
        // std::cout << "wrote wave to txt" << std::endl;

        int Nr = sim.grid_data.at("Nr").get<int>();
        double dr = sim.grid_data.at("grid_spacing").get<double>();
        int n_blocks = sim.angular_data.at("n_blocks").get<int>();
        int n_basis = sim.bspline_data.at("n_basis").get<int>();
        int degree = sim.bspline_data.at("degree").get<int>();
        std::map<int, std::pair<int, int>> block_to_lm = sim.block_to_lm;

        Vec final_state;
        load_final_state("TDSE_files/tdse_output.h5", &final_state, n_blocks*n_basis);


        std::vector<std::complex<double>> expanded_state (Nr * n_blocks,0.0);
        expand_state(final_state,expanded_state,Nr,n_blocks,n_basis,degree,dr,sim.knots,block_to_lm);

        return 0;
    }


}