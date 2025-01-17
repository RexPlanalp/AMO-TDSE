#include "data.h"
#include <vector>
#include <string>

struct bsplines : public data 
{   


    bsplines::bsplines(std::string& filename);
    std::complex<double> bsplines::ecs_x(double x);
    std::complex<double> bsplines::ecs_w(double x, double w);

    std::vector<std::complex<double>> complex_knots;
    std::vector<std::complex<double>> complex_weights;
    std::vector<std::complex<double>> _compute_complex_knots();
    std::vector<std::complex<double>> bsplines::_compute_complex_weights();



};