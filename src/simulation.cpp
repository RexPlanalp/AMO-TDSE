#include <iostream>
#include <fstream>
#include <complex>
#include <nlohmann/json.hpp>

#include "simulation.h"

simulation::simulation(const std::string& filename)
{
    std::ifstream file(filename);
    if (!file.is_open())
    {
        throw std::runtime_error("Could not open input file:" + filename);
    }

    nlohmann::json input_par;
    try 
    {
        file >> input_par;
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error parsing JSON:" + std::string(e.what()));
    }

    _parse_json(input_par);

    _process_bspline_data();
    _process_grid_data();
    _process_laser_data();
    _process_angular_data();

}

void simulation::_parse_json(const nlohmann::json& input_par)
{
    try 
    {
        this->bspline_data = input_par.at("bsplines");
        this->grid_data = input_par.at("grid");
        this->angular_data = input_par.at("angular");
        this->tdse_data = input_par.at("TDSE");
        this->tise_data = input_par.at("TISE");
        this->laser_data = input_par.at("laser");   
        this->state_data = input_par.at("state");
        this->misc_data = input_par.at("misc");
    }
    catch (const std::exception& e)
    {
        throw std::runtime_error("Error parsing JSON:" + std::string(e.what()));
    }
}


void simulation::_set_degree()
{   
    int order = this->bspline_data.value("order",0);

    this->bspline_data["degree"] = order - 1;
}

void simulation::_set_knots()
{
    int n_basis = this->bspline_data.value("n_basis",0);
    int order = this->bspline_data.value("order",0);
    double grid_size = this->grid_data.value("grid_size",0.0);

    int N_knots = n_basis + order;
    int N_middle = N_knots - 2 * (order - 2);
    
    double step_size  = grid_size / (N_middle-1);

    std::vector<double> knots_middle;
    for (int idx = 0; idx < N_middle; ++idx)
    {
        knots_middle.push_back(idx*step_size);
    }

    std::vector<double> knots_start(order - 2, 0.0);
    std::vector<double> knots_end(order-2,grid_size);

    knots_start.insert(knots_start.end(),knots_middle.begin(),knots_middle.end());
    knots_start.insert(knots_start.end(),knots_end.begin(),knots_end.end());

    this->knots = knots_start;
}

void simulation::_set_R0()
{
    double R0_input = this->bspline_data.value("R0_input", 0.0);

    double min_val = std::abs(this->knots[0]-R0_input);
    double knot_val = this->knots[0];

    for (int idx = 1; idx<this->knots.size(); ++idx)
    {
        double diff = std::abs(this->knots[idx]-R0_input);
        if (diff < min_val)
        {
            min_val = diff;
            knot_val = this->knots[idx];
        }
    }

    this->bspline_data["R0"] = knot_val;
}

std::complex<double> simulation::ecs_x(double x) const
{
    if (x < this->bspline_data.value("R0",0.0))
    {
        return std::complex<double>(x, 0.0);
    }
    else
    {
        return this->bspline_data.value("R0",0.0) +
               (x - this->bspline_data.value("R0",0.0)) *
               std::exp(std::complex<double>(0, M_PI * this->bspline_data.value("eta",0.0)));
    }
}

std::complex<double> simulation::ecs_w(double x, double w) const
{
    if (x < this->bspline_data.value("R0",0.0))
    {
        return std::complex<double>(w, 0.0);
    }
    else 
    {
        return w * std::exp(std::complex<double>(0, M_PI * this->bspline_data.value("eta",0.0)));
    }
}

void simulation::_compute_complex_knots()
{   
    for (int i = 0; i < this->knots.size(); i++)  
    {
        this->complex_knots.push_back(this->ecs_x(this->knots[i])); // Use 'this->' to access inherited members
    }
}

void simulation::_process_bspline_data()
{
    this->_set_degree();
    this->_set_knots();
    this->_set_R0();
    this->_compute_complex_knots();
}


void simulation::_normalize_array(std::array<double,3>& vec)
{      
    double norm = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);
    
    for (int idx = 0; idx< vec.size(); ++idx)
    {
        vec[idx] /= norm;
    }
}

void simulation::_cross_product(const std::array<double,3>& vec1, const std::array<double,3>& vec2, std::array<double,3>& result)
{
    result[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
    result[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
    result[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
}

void simulation::_set_laser_vectors()
{
    std::array<double,3> polarization = this->laser_data.value("polarization", std::array<double,3>{0,0.0,0.0});
    std::array<double,3> poynting = this->laser_data.value("poynting", std::array<double,3>{0.0,0.0,0});
    std::array<double,3> elliptiity = {};
    std::array<double,3> components = {};

    _cross_product(polarization,poynting,elliptiity);
    _normalize_array(elliptiity);
    _normalize_array(polarization);
    _normalize_array(poynting);
    _components(polarization,elliptiity,components);

    this->laser_data["ellipticity"] = elliptiity;
    this->laser_data["polarization"] = polarization;
    this->laser_data["poynting"] = poynting;
    this->laser_data["components"] = components;
}

void simulation::_components(const std::array<double,3>& polarization, const std::array<double,3>& ellipticity, std::array<double,3>& result)
{   
    double ell = this->laser_data["ell"];
    for (int idx = 0; idx < 3; ++idx)
    {
        result[idx] = (polarization[idx] != 0 || ell*ellipticity[idx] != 0) ? 1 : 0;
    }
}

void simulation::_set_amplitude()
{
    constexpr double I_au = 3.51E16;
    double I = this->laser_data.value("I",0.0);
    double w = this->laser_data.value("w",0.0);
    double A_0 = std::sqrt(I/I_au) / w;
    this->laser_data["A_0"] = A_0;
}

void simulation::_process_laser_data()
{
    this->_set_laser_vectors();
    this->_set_amplitude();
}


void simulation::_set_spacetime()
{
    double N = this->grid_data.value("N",0.0);
    double w = this->laser_data.value("w",0.0);
    double dt = this->grid_data.value("time_spacing",0.0);
    double time_size = N * 2 * M_PI / w;

    double dr = this->grid_data.value("grid_spacing",0.0);
    double grid_size  = this->grid_data.value("grid_size",0.0);

    std::array<double,3> time = {0.0,time_size,dt};
    std::array<double,3> space = {dr,grid_size,dr};

    int Nt = std::nearbyint(time_size/dt);
    int Nr = std::nearbyint(grid_size/dr);

    this->grid_data["time"] = time;
    this->grid_data["space"] = space;
    this->grid_data["Nt"] = Nt;
    this->grid_data["Nr"] = Nr;
}

void simulation::_process_grid_data()
{
    this->_set_spacetime();
}


void simulation::_process_angular_data()
{
    this->_lm_expansion();
    this->qn_map.set_block_to_lm();
}

void simulation::_lm_expansion()
{   

    std::array<double,3> components = this->laser_data.value("components",std::array<double,3>{0,0,0});
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
    int lmax = this->angular_data.value("lmax",0);

    for (int l = 0; l<=lmax;++l)
    {
        this->qn_map.lm_to_block[std::make_pair(l,0)] = l;
    }
}

void simulation::_xy_expansion()
{
    int block_idx = 0;
    for (int l = 0; l<=this->angular_data.value("lmax",0); ++l)
    {
        int temp_idx = 0;
        for (int m = -l; m<=l; ++m)
        {
            if (temp_idx/2 == 0)
            {
                this->qn_map.lm_to_block[std::make_pair(l,m)] = block_idx;
                block_idx++;
            }
            temp_idx++;
        }

    }
}

void simulation::_zxy_expansion()
{
    int block_idx = 0;
    for (int l = 0; l<=this->angular_data.value("lmax",0); ++l)
    {
        for (int m = -l; m<=l; ++m)
        {
            this->qn_map.lm_to_block[std::make_pair(l,m)] = block_idx;
            block_idx++;
        }
    }
}

void simulation::save_debug_info(int rank)
{
    if (!this->misc_data.value("debug",0)) return; // Only save if debugging is enabled

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
        std::string filename = "output.json";

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

