#include "laser.h"
#include <fstream>
#include <iostream>

double laser::sin2_envelope(double t, const simulation& sim)
{   
    double argument = sim.laser_params.w*t / (2*sim.grid_params.N);
    double value = sim.laser_params.A_0*std::sin(argument)*std::sin(argument);
    return value;
}

double laser::A(double t, const simulation& sim, int idx)
{
    std::array<double,3> polarization = sim.laser_params.polarization;
    std::array<double,3> ellipticity = sim.laser_params.ellipticity;
    double ell = sim.laser_params.ell;
    double w = sim.laser_params.w;
    double CEP = sim.laser_params.cep;
    double N = sim.grid_params.N;

    double prefactor = laser::sin2_envelope(t,sim)/std::sqrt(1+ell*ell);
    double term1 = polarization[idx]*std::sin(w *t + CEP - N*M_PI);
    double term2 = ell*ellipticity[idx]*std::cos(w*t + CEP - N*M_PI);
    return prefactor*(term1 + term2);
}

void laser::save_debug_laser(int rank, const simulation& sim) 
{
    if (!sim.debug) return;

    if (rank == 0)
    {
        int Nt = sim.grid_params.Nt;
        double dt = sim.grid_params.dt;

        try 
        {
            std::string filename = "debug/laser.txt";
            std::ofstream file(filename);
            if (!file.is_open())
            {
                throw std::runtime_error("Unable to open file for writing: " + filename);
            }
            for (int idx = 0; idx < Nt; ++idx) 
            {
                double t = idx * dt;
                double A_x = laser::A(t, sim, 0);
                double A_y = laser::A(t, sim, 1);
                double A_z = laser::A(t, sim, 2);
    
                file << t << " " << A_x << " " << A_y << " " << A_z << std::endl;
            }
            file.close();
        }
        catch (const std::exception& e)
        {
            throw std::runtime_error("Error in saving laser debug info: " + std::string(e.what()));
        }
    }
}
    