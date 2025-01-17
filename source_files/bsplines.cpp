#include "bsplines.h"
#include <cmath>
#include "gauss.h"

bsplines::bsplines(std::string& filename) : data(filename)
{
    complex_knots = _compute_complex_knots();
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

std::complex<double> bsplines::B(int i, int degree, std::complex<double> x)
{
    if (degree == 0)
    {
        return (complex_knots[i].real() <= x.real() && x.real() < complex_knots[i+1].real() ? 1.0 : 0.0);
    }


    std::complex<double> denom1 = complex_knots[i + degree] - complex_knots[i];
    std::complex<double> denom2 = complex_knots[i + degree + 1] - complex_knots[i + 1];

    std::complex<double> term1 = 0.0;
    std::complex<double> term2 = 0.0;

    if (denom1.real() > 0)
    {
        term1 = (x-complex_knots[i]) / (denom1) * B(i,degree-1,x);
    }
    if (denom2.real()>0)
    {
        term2 = (complex_knots[i+degree+1] -x ) / (denom2) * B(i+1,degree-1,x);
    }

    return term1+term2;


}



