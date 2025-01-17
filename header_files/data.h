#pragma once

#include <nlohmann/json.hpp>
#include <fstream>
#include <string>
#include <vector>
#include <complex>
#include <cmath>
#include <map>
#include <utility>


struct data
{
    nlohmann::json bspline_data;
    nlohmann::json laser_data;
    nlohmann::json grid_data;
    nlohmann::json species_data;
    nlohmann::json angular_data;
    nlohmann::json tise_data;
    nlohmann::json tdse_data;
    nlohmann::json state_data;
    nlohmann::json misc_data;

    std::vector<std::complex<double>> knots;
    std::map<std::pair<int,int>,int> lm_to_block;

    data(std::string& filename);

    void process_data();
    


    void _process_bspline_data();
    

    void _set_knots();
    

    void _set_degree();
    

    void _set_R0();
    


    void _process_laser_data();
    

    void _set_poynting();
    

    void _set_polarization();
    

    void _set_ellipticity();
   

    void _set_amplitude();
    

    void _set_components();
    

    void _process_grid_data();
    

    void _set_time();
    

    void _set_space();
    

    void _process_angular_data();
    

    void _lm_expansion();
    

    void _z_expansion();
    void _xy_expansion();
    void _zxy_expansion();

    void save_debug_info(int rank, bool debug);
    
};