#include "data.h"
#include <vector>
#include <string>

struct bsplines : public data 
{   


    bsplines(std::string& filename);
    std::complex<double> ecs_x(double x);
    std::complex<double> ecs_w(double x, double w);

    std::vector<std::complex<double>> complex_knots;
    std::vector<std::complex<double>> complex_weights;
    std::vector<std::complex<double>> _compute_complex_knots();
    std::vector<std::complex<double>> _compute_complex_weights();

    std::complex<double> B(int i, int degree, std::complex<double> x);



};