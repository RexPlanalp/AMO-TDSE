#include <iostream>
#include <fstream>
#include <map>
#include <algorithm>
#include <complex>

#include <gsl/gsl_sf_legendre.h>
#include <sys/stat.h>
#include <sys/types.h>

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>

#include "simulation.h"
#include "bsplines.h"
#include "misc.h"

using lm_pair = std::pair<int, int>;
using energy_l_pair = std::pair<double, int>;

namespace pes
{

    struct pes_context 
    {
        int Nr;
        int Ne;
        double dr;
        int n_blocks;
        int n_basis;
        int degree;
        int nmax;
        double Emax;
        double dE;
        int lmax;
        int debug;
        std::string SLICE;
        std::map<int, std::pair<int, int>> block_to_lm;
        std::map<std::pair<int, int>, int> lm_to_block;
        std::vector<std::complex<double>> knots;

        static pes_context set_config(const simulation& sim) 
        {   
            try {
                pes_context config;
                config.Nr = sim.grid_data.at("Nr").get<int>();
                config.Ne = sim.observable_data.at("Ne").get<int>();
                config.dr = sim.grid_data.at("grid_spacing").get<double>();
                config.n_blocks = sim.angular_data.at("n_blocks").get<int>();
                config.n_basis = sim.bspline_data.at("n_basis").get<int>();
                config.degree = sim.bspline_data.at("degree").get<int>();
                config.nmax = sim.angular_data.at("nmax").get<int>();
                config.Emax = sim.observable_data.at("E").get<std::array<double,2>>()[1];
                config.dE = sim.observable_data.at("E").get<std::array<double,2>>()[0];
                config.lmax = sim.angular_data.at("lmax").get<int>();
                config.SLICE = sim.observable_data.at("SLICE").get<std::string>();
                config.block_to_lm = sim.block_to_lm;
                config.lm_to_block = sim.lm_to_block;
                config.knots = sim.knots;
                config.debug = sim.debug;
                return config;
            }
            catch (std::exception& e)
            {
                std::cerr << "Error in setting up Photoelectron Spectra context: " << "\n\n" << e.what() << "\n\n";
                throw;
            }
        }
    };

    struct pes_filepaths
    {
        static constexpr const char* tdse_output = "TDSE_files/tdse_output.h5";
        static constexpr const char* tise_output = "TISE_files/tise_output.h5";
    };

    struct coulomb_wave 
    {
        double phase;
        std::vector<double> wave;
    };

    PetscErrorCode load_final_state(const char* filename, Vec* state, const pes_context& config) 
    {   
        PetscErrorCode ierr;
        PetscViewer viewer;

        ierr = VecCreate(PETSC_COMM_SELF, state); CHKERRQ(ierr);
        ierr = VecSetSizes(*state, PETSC_DECIDE, config.n_basis*config.n_blocks); CHKERRQ(ierr);
        ierr = VecSetFromOptions(*state); CHKERRQ(ierr);
        ierr = VecSetType(*state, VECMPI); CHKERRQ(ierr);
        ierr = VecSet(*state, 0.0); CHKERRQ(ierr);

        ierr = PetscObjectSetName((PetscObject)*state, "final_state"); CHKERRQ(ierr);
        ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, filename, FILE_MODE_READ, &viewer); CHKERRQ(ierr);
        ierr = VecLoad(*state, viewer); CHKERRQ(ierr);
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

        return ierr;
    }

    PetscErrorCode project_out_bound(const char* filename,Vec& state, const pes_context& config,const simulation& sim)
    {
        PetscErrorCode ierr;
        Vec state_block, tise_state,temp;
        IS is;
        std::complex<double> inner_product;
        PetscBool has_dataset;
        PetscViewer viewer;

        // Open HDF5 file for reading
        ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, filename, FILE_MODE_READ, &viewer); CHKERRQ(ierr);

        Mat S;
        ierr = bsplines::construct_matrix(sim,S,bsplines::overlap_integrand,false,false); CHKERRQ(ierr);

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
                    ierr = VecAXPY(state_block,-inner_product,tise_state); CHKERRQ(ierr); 
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

    std::complex<double> compute_Ylm(int l, int m, double theta, double phi) {

    
    // Using m parity identity to evaluate for negative m
    if (m < 0) 
    {
        int abs_m = -m;
        std::complex<double> Ylm = 
                                  gsl_sf_legendre_sphPlm(l, abs_m, std::cos(theta)) * 
                                  std::exp(std::complex<double>(0, -abs_m * phi));

        return std::pow(-1.0, abs_m) * std::conj(Ylm);
    }
    
    // For positive m evaluate as normal
    return  
           gsl_sf_legendre_sphPlm(l, m, std::cos(theta)) * 
           std::exp(std::complex<double>(0, m * phi));
}

    coulomb_wave compute_coulomb_wave(double E, int l, int Nr, double dr) {
    std::vector<double> wave(Nr, 0.0);
    const double dr2 = dr * dr;
    const double k = std::sqrt(2.0 * E);
    const int lterm = l * (l + 1);

    wave[0] = 0.0;
    wave[1] = 1.0;

    for (int idx = 2; idx < Nr; ++idx) {
        const double r_val = idx * dr;
        const double term = dr2 * (lterm/(r_val*r_val) + 2.0*H(r_val) - 2.0*E);
        wave[idx] = wave[idx - 1] * (term + 2.0) - wave[idx - 2];

        // Match Python's overflow handling
        if (std::abs(wave[idx]) > 1E10) {
            double max_val = *std::max_element(wave.begin(), wave.end(), 
                [](double a, double b) { return std::abs(a) < std::abs(b); });
            scale_vector(wave, 1.0/max_val);
        }
    }

    const double r_end = (Nr - 2) * dr;
    const double wave_end = wave[Nr - 2];
    const double dwave_end = (wave[Nr - 1] - wave[Nr - 3]) / (2.0 * dr);
    
    // Match Python's normalization approach
    const double denom = k + 1.0/(k * r_end);
    const double termPsi = std::abs(wave_end) * std::abs(wave_end);
    const double termDer = std::abs(dwave_end/denom) * std::abs(dwave_end/denom);
    const double normVal = std::sqrt(termPsi + termDer);

    if (normVal > 0.0) {
        scale_vector(wave, 1.0/normVal);
    }

    std::complex<double> numerator(0.0, wave_end);
    numerator += dwave_end / denom;

    const double scale = 2.0 * k * r_end;
    const std::complex<double> denomC = std::exp(std::complex<double>(0.0, 1.0/k) * std::log(scale));
    const std::complex<double> fraction = numerator / denomC;
    const double phase = std::arg(fraction) - k * r_end + l * M_PI/2.0;

    return coulomb_wave{phase, wave};
}
    
    PetscErrorCode expand_state(Vec& state,std::vector<std::complex<double>>& expanded_state,const pes_context& config)
    {   
        // Load the state into easily accessible array (avoid VecGetValue calls)
        PetscErrorCode ierr;
        const std::complex<double>* state_array;
        ierr =  VecGetArrayRead(state, reinterpret_cast<const PetscScalar**>(&state_array)); CHKERRQ(ierr);

        // Loop over all bspline basis function
        for (int bspline_idx = 0; bspline_idx < config.n_basis; ++bspline_idx)
        {   
            // Get the start and end of the bspline basis function
            std::complex<double> start = config.knots[bspline_idx];
            std::complex<double> end = config.knots[bspline_idx+config.degree+1];

            // Initialize vectors to store the evaluation of the bspline basis function and the corresponding indices
            std::vector<std::complex<double>> bspline_eval;
            std::vector<int> bspline_eval_indices;

            // Loop over all grid points
            for (int r_idx = 0; r_idx < config.Nr; ++r_idx)
            {   
                std::complex<double> r = r_idx*config.dr;
                if (r.real() >= start.real() && r.real() < end.real())
                {
                    std::complex<double> val = bsplines::B(bspline_idx,config.degree,r,config.knots);
                    bspline_eval.push_back(val);
                    bspline_eval_indices.push_back(r_idx);
                }
            }

            // Loop over each block 
            for (int block = 0; block < config.n_blocks; ++block)
            {   
                std::complex<double> coeff = state_array[block*config.n_basis + bspline_idx];
                // Loop over all grid points and add contribution to the expanded state for this block
                for (size_t r_sub_idx = 0; r_sub_idx < bspline_eval.size(); ++r_sub_idx)
                {
                    expanded_state[block*config.Nr + bspline_eval_indices[r_sub_idx]] += coeff*bspline_eval[r_sub_idx];
                }
            }
        }
        return ierr;
    }

    void compute_partial_spectra(const std::vector<std::complex<double>>& expanded_state,const pes_context& config,std::map<lm_pair,std::vector<std::complex<double>>>& partial_spectra,std::map<energy_l_pair,double> phases)
    {
        for (int block = 0; block < config.n_blocks; ++block)
        {
            
            lm_pair lm_pair = config.block_to_lm.at(block);
            int l = lm_pair.first;
            int m = lm_pair.second;
            partial_spectra[std::make_pair(l, m)].reserve(config.Ne); 
        }

        for (int E_idx = 1; E_idx <= config.Ne; ++E_idx)
        {

            if (config.debug)
            {
                std::cout << "Computing Partial Spectrum for E = " << (E_idx*config.dE) << "\n\n";
            }

            for (int block = 0; block < config.n_blocks; ++block)
            {
                
                lm_pair lm_pair = config.block_to_lm.at(block);
                int l = lm_pair.first;
                int m = lm_pair.second;
                coulomb_wave coulomb_result = compute_coulomb_wave(E_idx*config.dE, l, config.Nr, config.dr);
                
                phases[std::make_pair(E_idx*config.dE,l)] = coulomb_result.phase;


                auto start = expanded_state.begin() + config.Nr*block;  
                auto end = expanded_state.begin() + config.Nr*(block+1);    

                std::vector<std::complex<double>> block_vector(start, end);  

                std::vector<std::complex<double>> result;
                pes_pointwise_mult(coulomb_result.wave,block_vector,result);
                std::complex<double> I = pes_simpsons_method(result,config.dr);   
                partial_spectra[std::make_pair(l,m)].push_back(I);

            }
        }
        return;
    }

    void compute_angle_integrated(const std::map<lm_pair,std::vector<std::complex<double>>>& partial_spectra,const pes_context& config)
    {   

        std::ofstream pesFiles("PES_files/pes.txt", std::ios::app);
        std::vector<std::complex<double>> pes(config.Ne,0.0);
       
        for (int block = 0; block < config.n_blocks; ++block)
        {   
            lm_pair lm_pair = config.block_to_lm.at(block);
            int l = lm_pair.first;
            int m = lm_pair.second;

            std::vector<std::complex<double>> magsq(config.Ne,0.0);
            
            pes_pointwise_magsq(partial_spectra.at(std::make_pair(l,m)),magsq);
            pes_pointwise_add(pes,magsq,pes);
        }
        

        for (size_t idx = 0; idx < pes.size(); ++idx)
        {   
            if (config.debug)
            {
                std::cout << "Computing Angle Integrated Spectrum for E = " << (config.dE+1)*idx << "\n\n";
            }
            std::complex<double> val = pes[idx];
            val /= ((2*M_PI)*(2*M_PI)*(2*M_PI));
            pesFiles << (idx*config.dE+1) << " " << val.real() << " " << "\n";
        }

        pesFiles.close();
    }

    void compute_angle_resolved(const std::map<lm_pair,std::vector<std::complex<double>>>& partial_spectra,const pes_context& config,std::map<energy_l_pair,double> phases)
    {
        std::ofstream padFiles("PES_files/pad.txt", std::ios::app);
        std::vector<double> theta_range;
        std::vector<double> phi_range;

        if (config.SLICE == "XZ")
        {
            for (double theta = 0; theta <= M_PI; theta += 0.01) 
            {
                theta_range.push_back(theta);
            }

            phi_range  = {0.0,M_PI};
        }
        if (config.SLICE == "XY")
        {
            theta_range = {M_PI/ 2.0};

            for (double phi = 0; phi < 2.0*M_PI; phi += 0.01) 
            {
                phi_range.push_back(phi);
            }

        }
        if (config.SLICE != "XZ" && config.SLICE != "XY") {
            throw std::invalid_argument("Invalid SLICE value: " + config.SLICE);
        }

        for (int E_idx = 1; E_idx <= config.Ne; ++E_idx)
        {   

            double E = E_idx*config.dE;
            double k = std::sqrt(2.0*E);

            if (config.debug)
            {
                std::cout << "Computing Angle Resolved Spectrum for E = " << E << "\n\n";
            }

            for (auto& theta : theta_range)
            {
                for (auto& phi : phi_range)
                {   
                    std::complex<double> pad_amplitude {};
                    for (const auto& pair: partial_spectra)
                    {
                        int l = pair.first.first;
                        int m = pair.first.second;

                        std::complex<double> sph_term = compute_Ylm(l,m,theta,phi);

                        double partial_phase = phases[std::make_pair(E,l)];
                        std::complex<double> partial_amplitude = pair.second[E_idx - 1];

                        std::complex<double> phase_factor = std::exp(std::complex<double>(0.0,3*l*M_PI/2.0 + partial_phase));

                        pad_amplitude += sph_term*phase_factor*partial_amplitude;

                    }

                    double pad_prob = std::norm(pad_amplitude);
                    pad_prob/=((2*M_PI)*(2*M_PI)*(2*M_PI));
                    pad_prob/=k;

                    padFiles << E << " " << theta << " " << phi << " " << pad_prob << "\n";
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
        
        create_directory(rank,"PES_files");


        pes_context config = pes_context::set_config(sim);
        pes_filepaths filepaths = pes_filepaths();

        PetscErrorCode ierr;
        Vec final_state;
        ierr = load_final_state(filepaths.tdse_output, &final_state, config);

        

        ierr = project_out_bound(filepaths.tise_output,final_state, config,sim); CHKERRQ(ierr);

        std::vector<std::complex<double>> expanded_state (config.Nr * config.n_blocks,0.0);
        expand_state(final_state,expanded_state,config);

        
        std::map<energy_l_pair,double> phases;
        std::map<lm_pair,std::vector<std::complex<double>>> partial_spectra;

        compute_partial_spectra(expanded_state,config,partial_spectra,phases);

        compute_angle_integrated(partial_spectra,config);

        compute_angle_resolved(partial_spectra,config,phases);

      

        

        return 0;
    }
}