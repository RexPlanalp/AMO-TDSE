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

simulation::simulation(const std::string& filename)
{
    json input_par = read_json(filename);
    _read_input_par(input_par);

    _process_bspline_data();
    _process_grid_data();
    _process_laser_data();
    _process_angular_data();

}
void simulation::_read_input_par(const nlohmann::json& input_par)
{
    try 
    {
        bspline_data = input_par.at("bsplines");
        grid_data = input_par.at("grid");
        angular_data = input_par.at("angular");
        tdse_data = input_par.at("TDSE");
        tise_data = input_par.at("TISE");
        laser_data = input_par.at("laser");   
        observable_data = input_par.at("observables");
        debug = input_par.value("debug",0);
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error parsing JSON:" + std::string(e.what()));
    }
}
void simulation::save_debug_info(int rank)
{
    if (!debug) return; // Only save if debugging is enabled

    if (rank == 0)
    {
        if (mkdir("debug", 0777) == 0) 
        {
            std::cout << "Debug Directory Created" << "\n";
        } 
        else 
        {
            std::cout << "Error Creating Debug Directory" << "\n";
        }

        nlohmann::json debug_info;
        debug_info["bspline_data"] = bspline_data;
        debug_info["grid_data"] = grid_data;
        debug_info["angular_data"] = angular_data;
        debug_info["tise_data"] = tise_data;
        debug_info["tdse_data"] = tdse_data;
        debug_info["laser_data"] = laser_data;

        std::string filename = "debug/debug.json";

        std::ofstream file(filename);
        if (file.is_open())
        {
            file << std::setw(4) << debug_info; // Pretty-print JSON with indentation
            file.close();
            std::cout << "Debug information saved to " << filename << std::endl;
        }
        else
        {
            std::cerr << "Error: Unable to open file for writing: " << filename << std::endl;
        }


        save_lm_expansion(lm_to_block, "debug/lm_to_block.txt");
    }
    
}




void simulation::_process_bspline_data()
{
    _compute_degree();
    _compute_knots();
    _compute_R0();
    _compute_complex_knots();
}

void simulation::_compute_degree()
{   
    bspline_data["degree"] = bspline_data.at("order").get<int>() - 1;
}

void simulation::_compute_knots()
{
    int N_knots = bspline_data.at("n_basis").get<int>() + bspline_data.at("order").get<int>();
    int N_middle = N_knots - 2 * (bspline_data.at("order").get<int>() - 2);
    
    double step_size  = grid_data.at("grid_size").get<double>() / (N_middle-1);

    std::vector<std::complex<double>> knots_middle;
    for (int idx = 0; idx < N_middle; ++idx)
    {
        knots_middle.push_back(idx*step_size);
    }

    std::vector<std::complex<double>> knots_start(bspline_data.at("order").get<int>() - 2, 0.0);
    std::vector<std::complex<double>> knots_end(bspline_data.at("order").get<int>()-2,grid_data.at("grid_size").get<double>() );

    knots_start.insert(knots_start.end(),knots_middle.begin(),knots_middle.end());
    knots_start.insert(knots_start.end(),knots_end.begin(),knots_end.end());

    knots = knots_start;
}

void simulation::_compute_R0()
{
    double R0_input = bspline_data.at("R0_input").get<double>();

    double min_val = std::abs(knots[0]-R0_input);
    double knot_val = knots[0].real();

    for (int idx = 1; idx<knots.size(); ++idx)
    {
        double diff = std::abs(knots[idx]-R0_input);
        if (diff < min_val)
        {
            min_val = diff;
            knot_val = knots[idx].real();
        }
    }

    bspline_data["R0"] = knot_val;
}

std::complex<double> simulation::ecs_x(double x) const
{
    if (x < bspline_data.at("R0").get<double>())
    {
        return std::complex<double>(x, 0.0);
    }
    else
    {
        return bspline_data.at("R0").get<double>() +
               (x - bspline_data.at("R0").get<double>()) *
               std::exp(std::complex<double>(0, M_PI * bspline_data.at("eta").get<double>()));
    }
}

std::complex<double> simulation::ecs_w(double x, double w) const
{
    if (x < bspline_data.at("R0").get<double>())
    {
        return std::complex<double>(w, 0.0);
    }
    else 
    {
        return w * std::exp(std::complex<double>(0, M_PI * bspline_data.at("eta").get<double>()));
    }
}

void simulation::_compute_complex_knots()
{   
    for (int i = 0; i < knots.size(); i++)  
    {
        complex_knots.push_back(ecs_x(knots[i].real())); 
    }
}

void simulation::_compute_gauss()
{
    if (gauss.find(bspline_data.at("order").get<int>()) != gauss.end()) {
        std::pair<std::vector<double>, std::vector<double>> values = gauss[bspline_data.at("order").get<int>()];

        // Accessing the first vector
        std::vector<double> points = values.first;
        std::vector<double> weights = values.second;

        
    } else {
        std::cout << "Key not found!\n";
    }
}


void simulation::_process_laser_data()
{
    _compute_laser_vectors();
    _compute_amplitude();
    _compute_nonzero_components();
}

void simulation::_compute_laser_vectors()
{   
    Vec3 polarization = laser_data.at("polarization").get<Vec3>();
    Vec3 poynting = laser_data.at("poynting").get<Vec3>();
    Vec3 elliptiity = {};
    Vec3 components = {};

    cross_product(polarization,poynting,elliptiity);
    normalize_array(elliptiity);
    normalize_array(polarization);
    normalize_array(poynting);

    laser_data["ellipticity"] = elliptiity;
    laser_data["polarization"] = polarization;
    laser_data["poynting"] = poynting;
}

void simulation::_compute_nonzero_components()
{    
    Vec3 result = {};

    double ell = laser_data.at("ell").get<double>();
    for (int idx = 0; idx < 3; ++idx)
    {
        result[idx] = (laser_data.at("polarization").get<Vec3>()[idx] != 0 || ell*laser_data.at("ellipticity").get<Vec3>()[idx] != 0) ? 1 : 0;
    }
    laser_data["components"] = result;
}

void simulation::_compute_amplitude()
{
    double I = laser_data.at("I").get<double>();
    double w = laser_data.at("w").get<double>();
    double A_0 = std::sqrt(I/I_au) / w;
    laser_data["A_0"] = A_0;
}




void simulation::_compute_spacetime()
{
    double N = grid_data.at("N").get<double>();
    double w = laser_data.at("w").get<double>();
    double dt = grid_data.at("time_spacing").get<double>();
    grid_data["time_size"] = N * 2 * M_PI / w;
    grid_data["Nt"] = static_cast<int>(std::nearbyint(N * 2 * M_PI / (w*dt)));

    double dr = grid_data.at("grid_spacing").get<double>();
    double grid_size  = grid_data.at("grid_size").get<double>();
    grid_data["Nr"] = static_cast<int>(std::nearbyint(grid_size/dr));
}

void simulation::_process_grid_data()
{
    _compute_spacetime();
}


void simulation::_process_angular_data()
{
    _compute_lm_expansion();
}

void simulation::_compute_lm_expansion()
{   

    Vec3 components = laser_data.at("components").get<Vec3>(); 
    if (components[2] && !(components[1] || components[0]))
    {
        _z_expansion();
    }
    else if ((components[0] || components[1]) && !components[2])
    {
        _xy_expansion();
    }
    else
    {
        _zxy_expansion();
    }
    
}

void simulation::_z_expansion()
{   
    int lmax = angular_data.at("lmax").get<int>();

    for (int l = 0; l<=lmax;++l)
    {
        lm_to_block[int_tuple(l,0)] = l;
    }
}

void simulation::_xy_expansion()
{   
    int block_idx = 0;
    for (int l = 0; l<=angular_data.at("lmax").get<int>(); ++l)
    {
        int temp_idx = 0;
        for (int m = -l; m<=l; ++m)
        {
            if (temp_idx/2 == 0)
            {
                lm_to_block[int_tuple(l,m)] = block_idx;
                block_idx++;
            }
            temp_idx++;
        }

    }
}

void simulation::_zxy_expansion()
{
    int block_idx = 0;
    for (int l = 0; l<=angular_data.at("lmax").get<int>(); ++l)
    {
        for (int m = -l; m<=l; ++m)
        {
            lm_to_block[int_tuple(l,m)] = block_idx;
            block_idx++;
        }
    }
}







