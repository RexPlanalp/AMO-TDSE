#pragma once

#include "simulation.h"

namespace laser 
{
    double sin2_envelope(double t, const simulation& sim);
    double A(double t, const simulation& sim,int idx);
    void save_debug_laser(int rank, const simulation& sim);
}