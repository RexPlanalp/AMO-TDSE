#pragma once


#include <vector>
#include <nlohmann/json.hpp>
#include "misc.h"

class simulation 
{

    public:
    simulation( const std::string& filename);

    nlohmann::json bspline_data;
    nlohmann::json grid_data;
    nlohmann::json angular_data;
    nlohmann::json tise_data;
    nlohmann::json tdse_data;
    nlohmann::json laser_data;
    nlohmann::json state_data;
    nlohmann::json misc_data;

    qn_maps qn_map;
    std::vector<double> knots;
    std::vector<std::complex<double>> complex_knots;

    std::complex<double> ecs_x(double x) const;
    std::complex<double> ecs_w(double x, double w) const;

     void save_debug_info(int rank);
    

    private:
    void _parse_json(const nlohmann::json& input_par);

    void _set_degree();
    void _set_knots();
    void _set_R0();
    void _compute_complex_knots();
    void _process_bspline_data();

    void _normalize_array(std::array<double,3>& vec);
    void _cross_product(const std::array<double,3>& vec1, const std::array<double,3>& vec2, std::array<double,3>& result);
    void _set_laser_vectors();
    void _components(const std::array<double,3>& polarization, const std::array<double,3>& ellipticity, std::array<double,3>& result);
    void _set_amplitude();
    void _process_laser_data();

    void _set_spacetime();
    void _process_grid_data();

    void _lm_expansion();
    void _z_expansion();
    void _xy_expansion();
    void _zxy_expansion();
    void _process_angular_data();

   



};