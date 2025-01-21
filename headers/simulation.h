#pragma once


#include <vector>
#include <complex>
#include <map>

#include <nlohmann/json.hpp>
#include "misc.h"


class simulation 
{

    public:

    simulation( const std::string& filename);
    void save_debug_info(int rank);
    std::complex<double> ecs_x(double x) const;
    std::complex<double> ecs_w(double x, double w) const;

    nlohmann::json bspline_data;
    nlohmann::json grid_data;
    nlohmann::json angular_data;
    nlohmann::json tise_data;
    nlohmann::json tdse_data;
    nlohmann::json laser_data;
    nlohmann::json observable_data;
    int debug;

    double I_au = 3.51E16;

    std::vector<std::complex<double>> knots;
    std::vector<std::complex<double>> complex_knots;
    std::map<std::pair<int,int>,int> lm_to_block;
    std::map<int,std::pair<int,int>> block_to_lm;

    std::unordered_map<int, std::pair<std::vector<double>, std::vector<double>>> gauss = {
        {2, {{-0.57735027, 0.57735027}, {1, 1}}},
        {3, {{-0.77459667, 0.0, 0.77459667}, {0.55555556, 0.88888889, 0.55555556}}},
        {4, {{-0.86113631, -0.33998104, 0.33998104, 0.86113631}, {0.34785485, 0.65214515, 0.65214515, 0.34785485}}},
        {5, {{-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985}, {0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689}}},
        {6, {{-0.93246951, -0.66120939, -0.23861919, 0.23861919, 0.66120939, 0.93246951}, {0.17132449, 0.36076157, 0.46791393, 0.46791393, 0.36076157, 0.17132449}}},
        {7, {{-0.94910791, -0.74153119, -0.40584515, 0, 0.40584515, 0.74153119, 0.94910791}, {0.12948497, 0.27970539, 0.38183005, 0.41795918, 0.38183005, 0.27970539, 0.12948497}}},
        {8, {{-0.96028986, -0.79666648, -0.52553241, -0.18343464, 0.18343464, 0.52553241, 0.79666648, 0.96028986}, {0.10122854, 0.22238103, 0.31370665, 0.36268378, 0.36268378, 0.31370665, 0.22238103, 0.10122854}}}
    };

    std::vector<double> roots;
    std::vector<double> weights;



    
    

    private:
    void _read_input_par(const nlohmann::json& input_par);

    void _process_bspline_data();
    void _compute_degree();
    void _compute_knots();
    void _compute_R0();
    void _compute_complex_knots();
    void _compute_gauss();
    

    void _compute_laser_vectors();
    void _compute_nonzero_components();
    void _compute_amplitude();
    void _process_laser_data();

    void _compute_spacetime();
    void _process_grid_data();

    void _compute_lm_expansion();
    void _z_expansion();
    void _xy_expansion();
    void _zxy_expansion();
    void _process_angular_data();

   



};