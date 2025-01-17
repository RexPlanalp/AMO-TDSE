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
    void _compute_complex_knots();

   


    std::complex<double> B(int i, int degree, std::complex<double> x);
    std::complex<double> dB(int i, int degree, std::complex<double> x);
    
    void save_debug_info_bsplines(int rank);



};