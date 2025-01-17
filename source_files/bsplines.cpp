#include "bsplines.h"
#include <cmath>
#include "gauss.h"

bsplines::bsplines(std::string& filename) : data(filename)
{
    complex_knots = _compute_complex_knots();
    complex_weights = _compute_complex_weights();
}

std::complex<double> bsplines::ecs_x(double x)
{
    if (x < bspline_data["R0"].get<double>())
    {
        return std::complex<double>(x, 0.0);
    }
    else
    {
        return bspline_data["R0"].get<double>() +
               (x - bspline_data["R0"].get<double>()) *
               std::exp(std::complex<double>(0, M_PI * bspline_data["eta"].get<double>()));
    }
}

std::complex<double> bsplines::ecs_w(double x, double w)
{
    if (x < bspline_data["R0"].get<double>())
    {
        return std::complex<double>(w, 0.0);
    }
    else 
    {
        return w * std::exp(std::complex<double>(0, M_PI * bspline_data["eta"].get<double>()));
    }
}

std::vector<std::complex<double>> bsplines::_compute_complex_knots()
{
    std::vector<std::complex<double>> complex_knots;
    
    for (int i = 0; i < this->knots.size(); i++)  
    {
        complex_knots.push_back(this->ecs_x(this->knots[i])); // Use 'this->' to access inherited members
    }

    return complex_knots;
}

std::vector<std::complex<double>> bsplines::_compute_complex_weights()
{
    std::vector<std::complex<double>> complex_weights;
    std::vector<double> weights = gauss::get_weights(bspline_data["order"].get<int>());
    
    for (int i = 0; i < weights.size(); i++)
    {
        complex_weights.push_back(this->ecs_w(this->knots[i], weights[i])); // Use 'this->' to access inherited members
    }

    return complex_weights;
}

