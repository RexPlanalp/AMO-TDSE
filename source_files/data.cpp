#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <cmath>
#include <map>
#include <utility>
#include "data.h"
#include <iostream>



data::data(std::string& filename)
{
    std::ifstream file(filename);
    nlohmann::json input_par;
    file >> input_par;
    file.close();

    bspline_data = input_par["bsplines"];
    grid_data = input_par["grid"];
    angular_data = input_par["angular"];
    tise_data = input_par["TISE"];
    tdse_data = input_par["TDSE"];
    laser_data = input_par["laser"];
    state_data = input_par["state"];
    misc_data = input_par["misc"];

    _process_bspline_data();
    _process_laser_data();
    _process_grid_data();
    _process_angular_data();
}



void data::_process_bspline_data()
{   
    _set_degree();
    _set_knots();
    _set_R0();
    
}

void data::_set_knots() 
{	
    int  N_knots = bspline_data["n_basis"].get<int>() + bspline_data["order"].get<int>();

    int N_middle = N_knots - 2 * (bspline_data["order"].get<int>() - 2);
    double step_size = grid_data["grid_size"].get<double>() / (N_middle-1);
    std::vector<double> knots_middle;
    for (int idx = 0; idx < N_middle; ++idx) 
    {
        knots_middle.push_back(idx * step_size);
    }
    knots_middle.back() = grid_data["grid_size"].get<double>();

    std::vector<double> knots_start(bspline_data["order"].get<int>() - 2, 0.0);
    std::vector<double> knots_end(bspline_data["order"].get<int>() - 2, grid_data["grid_size"].get<double>());

    knots.insert(knots.end(), knots_start.begin(), knots_start.end());
    knots.insert(knots.end(), knots_middle.begin(), knots_middle.end());
    knots.insert(knots.end(), knots_end.begin(), knots_end.end());
}

void data::_set_degree()
{
    int degree = bspline_data["order"].get<int>() - 1;
    bspline_data["degree"] = degree;
}

void data::_set_R0()
{   
    double R0 = bspline_data["R0"].get<double>();

    double min_val = std::abs(knots[0] - R0);
    double knot_val = knots[0];


    for (int idx = 1; idx < knots.size(); ++ idx)
    {   

        double diff = std::abs(knots[idx] - R0);
        if (diff < min_val)
        {
            min_val = diff;
            knot_val = knots[idx];
        }
    }
    bspline_data["R0"] = knot_val;
}

void data::_process_laser_data()
{
    _set_poynting();
    _set_polarization();
    _set_ellipticity();
    _set_amplitude();
    _set_components();
}

void data::_set_poynting()
{
    std::array<double,3> poynting =  laser_data["poynting"].get<std::array<double,3>>(); 
    double norm = std::sqrt(poynting[0]*poynting[0] + poynting[1]*poynting[1] + poynting[2]*poynting[2]);
    for (int idx = 0; idx < 3; ++idx)
    {
        poynting[idx] = poynting[idx]/norm;
    }
    laser_data["poynting"] = poynting;
}

void data::_set_polarization()
{
    std::array<double,3> polarization =  laser_data["polarization"].get<std::array<double,3>>(); 
    double norm = std::sqrt(polarization[0]*polarization[0] + polarization[1]*polarization[1] + polarization[2]*polarization[2]);
    for (int idx = 0; idx < 3; ++idx)
    {
        polarization[idx] = polarization[idx]/norm;
    }
    laser_data["polarization"] = polarization;
}

void data::_set_ellipticity()
{
    std::array<double,3> ellipticity;
    std::array<double,3> poynting =  laser_data["poynting"].get<std::array<double,3>>();
    std::array<double,3> polarization =  laser_data["polarization"].get<std::array<double,3>>();
    ellipticity[0] = polarization[1]*poynting[2] - polarization[2]*poynting[1];
    ellipticity[1] = polarization[2]*poynting[0] - polarization[0]*poynting[2];
    ellipticity[2] = polarization[0]*poynting[1] - polarization[1]*poynting[0];
    laser_data["ellipticity"] = ellipticity;
}

void data::_set_amplitude()
{
    double I = laser_data["I"].get<double>();
    double I_au = I / 3.5094457e16;
    double E_0 = std::sqrt(I_au);
    double A_0 = E_0 / laser_data["w"].get<double>();
    laser_data["A_0"] = A_0;
}

void data::_set_components()
{
    std::array<double,3> components;
    for (int idx = 0; idx < 3; ++idx)
    {   

        if (laser_data["polarization"].get<std::array<double,3>>()[idx] != 0 || laser_data["ell"].get<double>() * laser_data["ellipticity"].get<std::array<double,3>>()[idx] != 0)
        {
            components[idx] = 1;
        }
        else
        {
            components[idx] = 0;
        }
        
    }
    laser_data["components"] = components;
}

void data::_process_grid_data()
{
    _set_time();
    _set_space();
}

void data::_set_time()
{   
    double N = grid_data["N"].get<double>();
    double w = laser_data["w"].get<double>();

    double dt = grid_data["time_spacing"].get<double>();
    double time_size = N * 2 * M_PI / w;
    int Nt = std::nearbyint(time_size / dt);

    grid_data.erase("time_spacing");
    std::array<double,3> time_range = {0.0,time_size,dt};
    grid_data["time_range"] = time_range;
    grid_data["Nt"] = Nt;
}

void data::_set_space()
{   
    double grid_size = grid_data["grid_size"].get<double>();
    double dx = grid_data["grid_spacing"].get<double>();
    int Nr = std::nearbyint(grid_size / dx);

    grid_data.erase("grid_spacing");
    grid_data.erase("grid_size");
    std::array<double,3> space_range = {dx,grid_size,dx};
    grid_data["space_range"] = space_range;
    grid_data["Nr"] = Nr;
}

void data::_process_angular_data()
{
    _lm_expansion();
}

void data::_lm_expansion()
{   

    std::array<double,3> components = laser_data["components"].get<std::array<double,3>>();
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

void data::_z_expansion()
{
    int lmax = angular_data["lmax"].get<int>();

    for (int l = 0; l<=lmax;++l)
    {
        lm_to_block[std::make_pair(l,0)] = l;
    }
}

void data::_xy_expansion()
{
    int block_idx = 0;
    for (int l = 0; l<=angular_data["lmax"].get<int>(); ++l)
    {
        int temp_idx = 0;
        for (int m = -l; m<=l; ++m)
        {
            if (temp_idx/2 == 0)
            {
                lm_to_block[std::make_pair(l,m)] = block_idx;
                block_idx++;
            }
            temp_idx++;
        }

    }
}

void data::_zxy_expansion()
{
    int block_idx = 0;
    for (int l = 0; l<=angular_data["lmax"].get<int>(); ++l)
    {
        for (int m = -l; m<=l; ++m)
        {
            lm_to_block[std::make_pair(l,m)] = block_idx;
            block_idx++;
        }
    }
}

void data::save_debug_info(int rank)
{
    if (!misc_data["debug"].get<int>()) return; // Only save if debugging is enabled

    if (rank == 0)
    {
        nlohmann::json debug_info;

        // Store various data objects, excluding knots and lm_block
        debug_info["bspline_data"] = bspline_data;
        debug_info["grid_data"] = grid_data;
        debug_info["angular_data"] = angular_data;
        debug_info["tise_data"] = tise_data;
        debug_info["tdse_data"] = tdse_data;
        debug_info["laser_data"] = laser_data;
        debug_info["state_data"] = state_data;
        debug_info["misc_data"] = misc_data;
        

        // Construct the filename using the rank
        std::string filename = "debug_info_" + std::to_string(rank) + ".json";

        // Save to file
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
    }
    
}
