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

namespace pes
{
    struct pes_context;
    struct pes_filepaths;
    struct coulomb_wave;

    PetscErrorCode load_final_state(const char* filename, Vec* state, const pes_context& config);
    PetscErrorCode project_out_bound(const char* filename, Mat& S, Vec& state, const pes_context& config);
    std::complex<double> compute_Ylm(int l, int m, double theta, double phi);
    coulomb_wave compute_coulomb_wave(double E, int l, int Nr, double dr);
    PetscErrorCode expand_state(Vec& state,std::vector<std::complex<double>>& expanded_state,const pes_context& config);
    std::map<std::pair<int,int>,std::vector<std::complex<double>>> compute_partial_spectra(const std::vector<std::complex<double>>& expanded_state,const pes_context& config,std::map<std::pair<double,int>,double> phases);
    void compute_angle_integrated(const std::map<std::pair<int,int>,std::vector<std::complex<double>>>& partial_spectra,const pes_context& config);
    void compute_angle_resolved(const std::map<std::pair<int,int>,std::vector<std::complex<double>>>& partial_spectra,const pes_context& config,std::map<std::pair<double,int>,double> phases);
    int compute_pes(int rank,const simulation& sim);
}


