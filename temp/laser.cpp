#include "laser.h"
#include <fstream>
#include <iostream>

double laser::sin2_envelope(double t, const simulation& sim)
{   
    double argument = sim.laser_data.value("w",0.0)*t / (2*sim.grid_data.value("N",0.0));
    double value = sim.laser_data.value("A_0",0.0)*std::sin(argument)*std::sin(argument);
    return value;
}

double laser::A(double t, const simulation& sim, int idx)
{
    std::array<double,3> polarization = sim.laser_data.value("polarization", std::array<double,3>{0,0.0,0.0});
    std::array<double,3> ellipticity = sim.laser_data.value("ellipticity", std::array<double,3>{0.0,0.0,0});
    double ell = sim.laser_data.value("ell",0.0);
    double w = sim.laser_data.value("w",0.0);
    double CEP = sim.laser_data.value("CEP",0.0);

    double prefactor = laser::sin2_envelope(t,sim)/std::sqrt(1+ell*ell);
    double term1 = polarization[idx]*std::sin(w *t + CEP);
    double term2 = ell*ellipticity[idx]*std::cos(w*t + CEP);
    return prefactor*(term1 + term2);
}

void laser::save_debug_laser(int rank, const simulation& sim) {
    if (!sim.misc_data.value("debug", 0)) return;


    std::string filename = "laser.txt";
    int Nt = sim.grid_data.value("Nt", 0);
    double dt = sim.grid_data.value("time_spacing", 0.0);

    if (rank == 0) {
        std::ofstream file(filename, std::ios::app);
        if (!file.is_open()) {
            std::cerr << "Error: Could not open file " << filename << std::endl;
            return;
        }

        for (int idx = 0; idx < Nt; ++idx) {
            double t = idx * dt;
            double A_x = laser::A(t, sim, 0);
            double A_y = laser::A(t, sim, 1);
            double A_z = laser::A(t, sim, 2);

            file << t << " " << A_x << " " << A_y << " " << A_z << std::endl;
        }

        file.close();
    }
}
    