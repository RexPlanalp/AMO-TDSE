#pragma once


#include <vector>
#include <complex>
#include <map>
#include <string>

#include <nlohmann/json.hpp>
#include "misc.h"


class simulation 
{

    public:

    std::string input_path = "input.json";
    std::string tise_output_path = "TISE_files";
    std::string tdse_output_path = "TDSE_files";


    nlohmann::json processed_input_par;
    int debug;
    simulation();
    void process_input_data();


    struct BSplineParams 
    {
        int n_basis;
        int order;
        int degree;
        double R0_input;
        double R0;
        double eta;
        std::vector<std::complex<double>> knots;
        std::vector<std::complex<double>> complex_knots;
        std::vector<double> roots;
        std::vector<double> weights;
        int debug;
    };
    
    BSplineParams bspline_params;  

    struct GridParams
    {
        double N;
        double dt;
        double dr;
        double grid_size;
        double Nt;
        double Nr;
        double time_size;
    }; 
    
    GridParams grid_params;  

    struct LaserParams
    {
        double w;
        double I;
        double ell; 
        double cep; 
        double A_0;
        std::array<double,3> polarization;
        std::array<double,3> poynting;
        std::array<double,3> ellipticity;
        std::array<double,3> components;

    }; 
    
    LaserParams laser_params;  

    struct AngularParams
    {
       int lmax;
       int nmax;
       int mmin;
       int mmax;
       int n_blocks;
       std::map<std::pair<int,int>,int> lm_to_block;
       std::map<int,std::pair<int,int>> block_to_lm;
    }; 
    
    AngularParams angular_params;  

    struct ObservableParams
    {
       double Emax;
       double dE;
       int Ne;
       int hhg; 
       int cont;
       std::string SLICE;
    }; 
    
    ObservableParams observable_params;  


  
    



    void save_debug_info(int rank);
    std::complex<double> ecs_x(double x) const;
    std::complex<double> ecs_w(double x, double w) const;

  
    

    double I_au = 3.51E16;

  

    std::unordered_map<int, std::pair<std::vector<double>, std::vector<double>>> gauss = {
        {2, {{-0.57735027, 0.57735027}, {1, 1}}},
        {3, {{-0.77459667, 0.0, 0.77459667}, {0.55555556, 0.88888889, 0.55555556}}},
        {4, {{-0.86113631, -0.33998104, 0.33998104, 0.86113631}, {0.34785485, 0.65214515, 0.65214515, 0.34785485}}},
        {5, {{-0.90617985, -0.53846931, 0.0, 0.53846931, 0.90617985}, {0.23692689, 0.47862867, 0.56888889, 0.47862867, 0.23692689}}},
        {6, {{-0.93246951, -0.66120939, -0.23861919, 0.23861919, 0.66120939, 0.93246951}, {0.17132449, 0.36076157, 0.46791393, 0.46791393, 0.36076157, 0.17132449}}},
        {7, {{-0.94910791, -0.74153119, -0.40584515, 0, 0.40584515, 0.74153119, 0.94910791}, {0.12948497, 0.27970539, 0.38183005, 0.41795918, 0.38183005, 0.27970539, 0.12948497}}},
        {8, {{-0.96028986, -0.79666648, -0.52553241, -0.18343464, 0.18343464, 0.52553241, 0.79666648, 0.96028986}, {0.10122854, 0.22238103, 0.31370665, 0.36268378, 0.36268378, 0.31370665, 0.22238103, 0.10122854}}}
    };

    



    
    

    private:


    void compute_spacetime();
    void compute_degree();
    void compute_knots();
    void compute_R0();
    void compute_complex_knots();
    void compute_gauss();
    void compute_laser_vectors();
    void compute_nonzero_components();
    void compute_amplitude();
    void compute_lm_expansion();
    void invert_lm_expansion();
    void compute_n_blocks();
    void z_expansion();
    void xy_expansion();
    void zxy_expansion();
    void compute_energy();



   



};