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
    PetscErrorCode load_final_state(const char* filename, Vec* state, int total_size);
    PetscErrorCode project_out_bound(const char* filename, Mat& S, Vec& state, int n_basis, int n_blocks, int nmax, std::map<int, std::pair<int, int>>& block_to_lm);
    double H(double r);
    void scale_vector(std::vector<double>& vec, double scale);
    struct CoulombResult;
    CoulombResult compute_coulomb_wave(double E, int l, int Nr, double dr);
    int compute_pes(int rank,const simulation& sim);
}


