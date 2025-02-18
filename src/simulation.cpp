#include <iostream>
#include <fstream>
#include <complex>
#include <sys/stat.h>
#include <sys/types.h>

#include <nlohmann/json.hpp>

#include "simulation.h"
#include "misc.h"

using json = nlohmann::json;
using Vec3 = std::array<double,3>;
using lm_dict = std::map<std::pair<int,int>,int>;
using block_dict = std::map<int,std::pair<int,int>>;
using int_tuple = std::pair<int,int>;
using Complex = std::complex<double>;




simulation::simulation()
{
    // Step 1: Read in input parameters
    std::ifstream input_file(input_path);
    if (!input_file.is_open())
    {
        throw std::runtime_error("Could not open input file: " + input_path);
    }
    try
    {
        input_file >> processed_input_par;
    }
    catch(const std::exception& e)
    {
        throw std::runtime_error("Error parsing JSON file: " + std::string(e.what()));
    }

    // Step 2: Process input parameters
    process_input_data();




}

void simulation::process_input_data()
{   
    // First set all basic parameters from JSON
    
    // 1. Set grid basic parameters
    grid_params.N = processed_input_par["grid"].at("N").get<double>();
    grid_params.dt = processed_input_par["grid"].at("time_spacing").get<double>();
    grid_params.dr = processed_input_par["grid"].at("grid_spacing").get<double>();
    grid_params.grid_size = processed_input_par["grid"].at("grid_size").get<double>();

    // 2. Set bspline basic parameters
    bspline_params.n_basis = processed_input_par["bsplines"].at("n_basis").get<int>();
    bspline_params.order = processed_input_par["bsplines"].at("order").get<int>();
    bspline_params.R0_input = processed_input_par["bsplines"].at("R0_input").get<double>();
    bspline_params.eta = processed_input_par["bsplines"].at("eta").get<double>();
    bspline_params.debug = processed_input_par["bsplines"].at("debug").get<int>();

    // 3. Set laser basic parameters
    laser_params.w = processed_input_par["laser"].at("w").get<double>();
    laser_params.I = processed_input_par["laser"].at("I").get<double>();
    laser_params.ell = processed_input_par["laser"].at("ell").get<double>();
    laser_params.cep = processed_input_par["laser"].at("CEP").get<double>();

    // 4. set angular basic parameters
    angular_params.lmax = processed_input_par["angular"].at("lmax").get<int>();
    angular_params.nmax = processed_input_par["angular"].at("nmax").get<int>();
    angular_params.mmin = processed_input_par["angular"].at("mmin").get<int>();
    angular_params.mmax = processed_input_par["angular"].at("mmax").get<int>();

    // 5. Set observable basic parameters
    std::array<double,2> E_array = processed_input_par["observables"].at("E").get<std::array<double,2>>();
    observable_params.dE = E_array[0];
    observable_params.Emax = E_array[1];

    observable_params.hhg = processed_input_par["observables"].at("HHG").get<int>();
    observable_params.cont = processed_input_par["observables"].at("CONT").get<int>();
    observable_params.SLICE = processed_input_par["observables"].at("SLICE").get<std::string>();

    // 6. Set TISE and TDSE basic parameters

    schrodinger_params.tise_tol = processed_input_par["TISE"].at("tolerance").get<double>();
    schrodinger_params.tise_max_iter = processed_input_par["TISE"].at("max_iter").get<int>();
    schrodinger_params.tdse_tol = processed_input_par["TDSE"].at("tolerance").get<double>();
    schrodinger_params.state = processed_input_par["TDSE"].at("state").get<std::array<int,3>>();

    


    // Then compute derived values
    compute_spacetime();
    compute_degree();
    compute_knots();
    compute_R0();
    compute_complex_knots();
    compute_gauss();
    compute_amplitude();
    compute_laser_vectors();
    compute_nonzero_components();
    compute_lm_expansion();
    invert_lm_expansion();
    compute_n_blocks();
    compute_energy();

    debug = processed_input_par["debug"].get<int>();

}

Complex simulation::ecs_x(double x) const
{
    if (x < bspline_params.R0)
    {
        return Complex(x, 0.0);
    }
    else
    {
        return bspline_params.R0 +
               (x - bspline_params.R0) *
               std::exp(Complex(0, M_PI * bspline_params.eta));
    }
}

Complex simulation::ecs_w(double x, double w) const
{
    if (x < bspline_params.R0)
    {
        return Complex(w, 0.0);
    }
    else 
    {
        return w * std::exp(Complex(0, M_PI * bspline_params.eta));
    }
}




void simulation::compute_spacetime()
{
    double w = processed_input_par["laser"].at("w").get<double>();
    
    grid_params.time_size = processed_input_par["grid"]["time_size"] = grid_params.N * 2 * M_PI / w;
    grid_params.Nt = processed_input_par["grid"]["Nt"] = std::nearbyint(grid_params.N * 2 * M_PI / (w * grid_params.dt));
    grid_params.Nr = processed_input_par["grid"]["Nr"] = std::nearbyint(grid_params.grid_size / grid_params.dr);
}

void simulation::compute_degree()
{   
    bspline_params.degree = processed_input_par["bsplines"]["degree"] = bspline_params.order - 1;
}

void simulation::compute_R0()
{
    double min_val = std::abs(bspline_params.knots[0] - bspline_params.R0_input);
    double knot_val = bspline_params.knots[0].real();

    for (size_t idx = 1; idx < bspline_params.knots.size(); ++idx)
    {
        double diff = std::abs(bspline_params.knots[idx] - bspline_params.R0_input);
        if (diff < min_val)
        {
            min_val = diff;
            knot_val = bspline_params.knots[idx].real();
        }
    }

    bspline_params.R0 = processed_input_par["bsplines"]["R0"] = knot_val;
}

void simulation::compute_knots()
{
    int n_basis = bspline_params.n_basis;
    int order = bspline_params.order;
    double grid_size = grid_params.grid_size;

    int N_knots = n_basis + order;
    int N_middle = N_knots - 2 * (order - 2);
    
    double step_size  = grid_size / (N_middle-1);

    std::vector<Complex> knots_middle;
    knots_middle.reserve(N_middle);
    for (int idx = 0; idx < N_middle; idx++)
    {
        knots_middle.push_back(idx*step_size);
    }

    std::vector<Complex> knots_start(order - 2, 0.0);
    std::vector<Complex> knots_end(order-2, grid_size);

    knots_start.insert(knots_start.end(),knots_middle.begin(),knots_middle.end());
    knots_start.insert(knots_start.end(),knots_end.begin(),knots_end.end());

    bspline_params.knots = knots_start;
}

void simulation::compute_complex_knots()
{   
    bspline_params.complex_knots.reserve(bspline_params.knots.size());
    for (const auto& knot_val : bspline_params.knots)
    {
        bspline_params.complex_knots.push_back(ecs_x(knot_val.real()));
    }
}

void simulation::compute_gauss()
{   
    int order = bspline_params.order;

    if (gauss.find(order) != gauss.end()) {
        std::pair<std::vector<double>, std::vector<double>> values = gauss[order];

        bspline_params.roots = values.first;
        bspline_params.weights = values.second;
    } else {
        throw std::runtime_error("Gauss-Legendre quadrature not implemented for order " + std::to_string(order));
    }
}

void simulation::compute_amplitude()
{
    double I = laser_params.I;
    double w = laser_params.w;
    double A_0 = std::sqrt(I/I_au) / w;
    laser_params.A_0 = processed_input_par["laser"]["A_0"] = A_0;
}

void simulation::compute_laser_vectors()
{   
    Vec3 polarization = processed_input_par["laser"].at("polarization").get<Vec3>();
    Vec3 poynting = processed_input_par["laser"].at("poynting").get<Vec3>();
    Vec3 ellipticity = {};

    cross_product(polarization,poynting,ellipticity);
    normalize_array(ellipticity);
    normalize_array(polarization);
    normalize_array(poynting);

    laser_params.ellipticity = processed_input_par["laser"]["ellipticity"] = ellipticity;
    laser_params.polarization = processed_input_par["laser"]["polarization"] = polarization;
    laser_params.poynting = processed_input_par["laser"]["poynting"] = poynting;
}

void simulation::compute_nonzero_components()
{    
    Vec3 components = {};

    for (int idx = 0; idx < 3; idx++)
    {
        components[idx] = (laser_params.polarization[idx] != 0 || laser_params.ell*laser_params.ellipticity[idx] != 0) ? 1 : 0;
    }
    laser_params.components = processed_input_par["laser"]["components"] = components;
}



void simulation::compute_lm_expansion()
{   

    Vec3 components = laser_params.components; 
    if (components[2] && !(components[1] || components[0]))
    {
        z_expansion();
    }
    else if ((components[0] || components[1]) && !components[2])
    {
        xy_expansion();
    }
    else
    {
        zxy_expansion();
    }
    
}

void simulation::invert_lm_expansion()
{
    for (const auto& pair : angular_params.lm_to_block) 
    {
        angular_params.block_to_lm[pair.second] = pair.first; 
    }
}

void simulation::compute_n_blocks()
{
    angular_params.n_blocks = processed_input_par["angular"]["n_blocks"] = angular_params.lm_to_block.size();
}

void simulation::z_expansion()
{   
    int lmax = angular_params.lmax;

    for (int l = 0; l<=lmax;l++)
    {
        angular_params.lm_to_block[int_tuple(l,0)] = l;
    }
}

void simulation::xy_expansion()
{    
    int lmax = angular_params.lmax;

    int block_idx = 0;
    for (int l = 0; l<=lmax; l++)
    {
        int temp_idx = 0;
        for (int m = -l; m<=l; ++m)
        {
            if (temp_idx%2 == 0)
            {
                angular_params.lm_to_block[int_tuple(l,m)] = block_idx;
                block_idx++;
            }
            temp_idx++;
        }

    }
}

void simulation::zxy_expansion()
{   
    int lmax = angular_params.lmax;

    int block_idx = 0;
    for (int l = 0; l<=lmax; l++)
    {
        for (int m = -l; m<=l; ++m)
        {
            angular_params.lm_to_block[int_tuple(l,m)] = block_idx;
            block_idx++;
        }
    }
}





void simulation::compute_energy()
{
    double dE = observable_params.dE;
    double Emax = observable_params.Emax;

    observable_params.Ne = processed_input_par["observables"]["Ne"] = std::nearbyint(Emax/dE);
}



void simulation::save_debug_info(int rank)
{
   if (!debug) return;

   if (rank == 0)
   {
       try {
           create_directory(rank,"debug");

           std::string filename = "debug/processed_input.json";
           std::ofstream file(filename);
           if (!file.is_open())
           {
               throw std::runtime_error("Unable to open file for writing: " + filename);
           }

           file << std::setw(4) << processed_input_par;
           file.close();
           std::cout << "Debug information saved to " << filename << "\n";

           // Save lm expansion mapping
           std::ofstream map_file("debug/lm_to_block.txt");
           if (!map_file.is_open())
           {
               throw std::runtime_error(std::string("Unable to open file for writing: ") + "debug/lm_to_block.txt");
           }

           for (const auto& pair : angular_params.lm_to_block)
           {
               map_file << "(" << pair.first.first << "," << pair.first.second << ") -> " 
                       << pair.second << "\n";
           }
           map_file.close();
           std::cout << "LM mapping saved to " << filename << "\n";
       }
       catch (const std::exception& e)
       {
           throw std::runtime_error("Error in save_debug_info: " + std::string(e.what()));
       }
   }
}




