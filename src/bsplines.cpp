#include <cmath>
#include <fstream>
#include <iostream>
#include <functional>
#include <algorithm> 

#include "bsplines.h"
#include "simulation.h"
#include "misc.h"

namespace bsplines 
{


void save_debug_bsplines(int rank, const simulation& sim)
{
    if (!sim.debug or !sim.bspline_params.debug) return; 

    if (rank == 0)
    {
        std::ofstream file1("debug/bsplines.txt");
        std::ofstream file2("debug/dbsplines.txt");

        if (!file1.is_open())
        {
            std::cerr << "Error: could not open file bsplines.txt" << std::endl;
            return;
        }
        if (!file2.is_open())
        {
            std::cerr << "Error: could not open file dbsplines.txt" << std::endl;
            return;
        }

        

        for (int i = 0; i < sim.bspline_params.n_basis; i++)
        {
            for (int idx = 0; idx < sim.grid_params.Nr; ++idx)
            {


                
                double x_val = idx * sim.grid_params.dr;

                if (x_val > sim.bspline_params.knots[i].real() && x_val < sim.bspline_params.knots[i+sim.bspline_params.degree+1].real())
                {
                    std::complex<double> x = sim.ecs_x(x_val);
                    std::complex<double> val_B = B(i, sim.bspline_params.degree, x, sim.bspline_params.complex_knots);
                    std::complex<double> val_dB = dB(i, sim.bspline_params.degree, x, sim.bspline_params.complex_knots);
                    
                    file1 << val_B.real() << "\t" << val_B.imag() << "\n";
                    file2 << val_dB.real() << "\t" << val_dB.imag() << "\n";
                }
                else
                {
                    file1 << 0.0 << "\t" << 0.0 << "\n";
                    file2 << 0.0 << "\t" << 0.0 << "\n";
                }
            }
            file1 << "\n";
            file2 << "\n";
        }

        file1.close();
        file2.close();
    }
}

std::complex<double> B(int i, int degree, std::complex<double> x, const std::vector<std::complex<double>>& knot_vector)
{
    if (degree == 0)
    {
        return (knot_vector[i].real() <= x.real() && x.real() < knot_vector[i + 1].real() ? 1.0 : 0.0);
    }

    std::complex<double> denom1 = knot_vector[i + degree] - knot_vector[i];
    std::complex<double> denom2 = knot_vector[i + degree + 1] - knot_vector[i + 1];

    std::complex<double> term1 = 0.0;
    std::complex<double> term2 = 0.0;

    if (denom1.real() > 0)
    {
        term1 = (x - knot_vector[i]) / denom1 * B(i, degree - 1, x, knot_vector);
    }
    if (denom2.real() > 0)
    {
        term2 = (knot_vector[i + degree + 1] - x) / denom2 * B(i + 1, degree - 1, x, knot_vector);
    }

    return term1 + term2;
}

std::complex<double> dB(int i, int degree, std::complex<double> x, const std::vector<std::complex<double>>& knot_vector)
{
    if (degree == 0)
    {
        return 0.0;
    }

    std::complex<double> denom1 = knot_vector[i + degree] - knot_vector[i];
    std::complex<double> denom2 = knot_vector[i + degree + 1] - knot_vector[i + 1];

    std::complex<double> term1 = 0.0;
    std::complex<double> term2 = 0.0;

    if (denom1.real() > 0)
    {
        term1 = std::complex<double>(degree) / denom1 * B(i, degree - 1, x, knot_vector);
    }
    if (denom2.real() > 0)
    {
        term2 = -std::complex<double>(degree) / denom2 * B(i + 1, degree - 1, x, knot_vector);
    }

    return term1 + term2;
}

std::complex<double> integrate_matrix_element(int i, int j,std::function<std::complex<double>(int, int, std::complex<double>, int,const std::vector<std::complex<double>>&)> integrand,const simulation& sim,bool use_ecs)
{
    std::complex<double> total = 0.0;
    int lower = std::min(i, j);
    int upper = std::max(i, j);

    for (int k = lower; k <= upper + sim.bspline_params.degree; ++k)
    {
        double a = sim.bspline_params.knots[k].real();
        double b = sim.bspline_params.knots[k + 1].real();


        if (a == b)
        {
            continue;
        }

        double half_b_minus_a = 0.5 * (b - a);
        double half_b_plus_a = 0.5 * (b + a);


        for (size_t r = 0; r < sim.bspline_params.roots.size(); ++r)
        {
            double x_val = half_b_minus_a * sim.bspline_params.roots[r] + half_b_plus_a;
            double weight_val = sim.bspline_params.weights[r];

            if (use_ecs)
            {
                std::complex<double> x = sim.ecs_x(x_val);
                std::complex<double> weight = sim.ecs_w(x_val, weight_val) * half_b_minus_a;
                std::complex<double> integrand_val = integrand(i, j, x, sim.bspline_params.degree,sim.bspline_params.complex_knots);
                total += weight * integrand_val;
            }
            else
            {
                std::complex<double> x = x_val;
                std::complex<double> weight = weight_val* half_b_minus_a;
                std::complex<double> integrand_val = integrand(i, j, x, sim.bspline_params.degree,sim.bspline_params.knots);
                total += weight * integrand_val;
            }
        }
    }

    return total;
}

std::complex<double> overlap_integrand(int i, int j, std::complex<double> x, int degree,const std::vector<std::complex<double>>& knot_vector)
{
    return bsplines::B(i, degree, x, knot_vector) * 
           bsplines::B(j, degree, x, knot_vector);
}

std::complex<double> kinetic_integrand(int i, int j, std::complex<double> x, int degree,const std::vector<std::complex<double>>& knot_vector)
{
    return 0.5*bsplines::dB(i, degree, x, knot_vector) * 
           bsplines::dB(j,degree, x, knot_vector);
}

std::complex<double> invr_integrand(int i, int j, std::complex<double> x, int degree,const std::vector<std::complex<double>>& knot_vector)
{
    return bsplines::B(i, degree, x, knot_vector) * 
           bsplines::B(j, degree, x, knot_vector) /(x + 1E-25);
}

std::complex<double> invr2_integrand(int i, int j, std::complex<double> x, int degree,const std::vector<std::complex<double>>& knot_vector)
{
    return bsplines::B(i, degree, x, knot_vector) * 
           bsplines::B(j, degree, x, knot_vector) /(x*x + 1E-25);
}

std::complex<double> der_integrand(int i, int j, std::complex<double> x,int degree,const std::vector<std::complex<double>>& knot_vector)
{
    return bsplines::B(i, degree, x, knot_vector) * 
           bsplines::dB(j, degree, x, knot_vector);
}

std::complex<double> H_integrand(int i, int j, std::complex<double> x,int degree,const std::vector<std::complex<double>>& knot_vector)
{
    return bsplines::B(i, degree, x, knot_vector) * 
           bsplines::B(j, degree, x, knot_vector) *
           H(x);
}

PetscErrorCode save_matrix(Mat A, const char *filename)
    {
        PetscErrorCode ierr;
        PetscViewer viewer;

        // Open a binary viewer in write mode
        ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer); CHKERRQ(ierr);

        // Write the matrix to the file in parallel
        ierr = MatView(A, viewer); CHKERRQ(ierr);

        // Clean up the viewer
        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);

        return ierr;
    }

PetscErrorCode SaveMatrixToCSV(Mat M, const std::string& filename) {
    PetscErrorCode ierr;
    PetscInt m, n;

    // Get matrix dimensions
    ierr = MatGetSize(M, &m, &n); CHKERRQ(ierr);

    std::ofstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return PETSC_ERR_FILE_OPEN;
    }

    // Write matrix values row by row
    for (PetscInt i = 0; i < m; ++i) {
        for (PetscInt j = 0; j < n; ++j) {
            PetscScalar value;
            ierr = MatGetValues(M, 1, &i, 1, &j, &value); CHKERRQ(ierr);
            file << value;
            if (j < n - 1) {
                file << ", ";  // Add CSV separator
            }
        }
        file << "\n"; // Newline for next row
    }

    file.close();
    return ierr;
}

}
