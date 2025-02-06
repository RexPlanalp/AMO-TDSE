#include <cmath>
#include <fstream>
#include <iostream>
#include <functional>
#include <algorithm> 

#include "bsplines.h"
#include "simulation.h"

namespace bsplines 
{
void save_debug_bsplines(int rank, const simulation& sim)
{
    if (!sim.debug or !sim.bspline_data.at("debug").get<int>()) return; // Only save if debugging is enabled

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

        int n_basis = sim.bspline_data.at("n_basis").get<int>();
        int Nr = sim.grid_data.at("Nr").get<int>();
        double grid_spacing = sim.grid_data.at("grid_spacing").get<double>();
        int degree = sim.bspline_data.at("degree").get<int>();

        for (int i = 0; i < n_basis; i++)
        {
            for (int idx = 0; idx < Nr; ++idx)
            {


                
                double x_val = idx * grid_spacing;

                if (x_val > sim.knots[i].real() && x_val < sim.knots[i+sim.bspline_data.at("degree").get<int>()+1].real())
                {
                    std::complex<double> x = sim.ecs_x(x_val);
                    std::complex<double> val_B = B(i, degree, x, sim.complex_knots);
                    std::complex<double> val_dB = dB(i, degree, x, sim.complex_knots);
                    
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

    for (int k = lower; k <= upper + sim.bspline_data.at("degree").get<int>(); ++k)
    {
        double a = sim.knots[k].real();
        double b = sim.knots[k + 1].real();


        if (a == b)
        {
            continue;
        }

        double half_b_minus_a = 0.5 * (b - a);
        double half_b_plus_a = 0.5 * (b + a);


        for (int r = 0; r < sim.roots.size(); ++r)
        {
            double x_val = half_b_minus_a * sim.roots[r] + half_b_plus_a;
            double weight_val = sim.weights[r];

            if (use_ecs)
            {
                std::complex<double> x = sim.ecs_x(x_val);
                std::complex<double> weight = sim.ecs_w(x_val, weight_val) * half_b_minus_a;
                std::complex<double> integrand_val = integrand(i, j, x, sim.bspline_data.at("degree").get<int>(),sim.complex_knots);
                total += weight * integrand_val;
            }
            else
            {
                std::complex<double> x = x_val;
                std::complex<double> weight = weight_val* half_b_minus_a;
                std::complex<double> integrand_val = integrand(i, j, x, sim.bspline_data.at("degree").get<int>(),sim.knots);
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

PetscErrorCode construct_matrix(const simulation& sim, Mat& M, std::function<std::complex<double>(int, int, std::complex<double>, int,std::vector<std::complex<double>>)> integrand,bool use_mpi,bool use_ecs)
{
    PetscErrorCode ierr;
    int nnz_per_row = 2 * sim.bspline_data.value("degree",0) + 1;

    if (use_mpi)
    {
        ierr = MatCreate(PETSC_COMM_WORLD, &M); CHKERRQ(ierr);
        ierr = MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, sim.bspline_data.value("n_basis",0), sim.bspline_data.value("n_basis",0)); CHKERRQ(ierr);
        ierr = MatSetFromOptions(M); CHKERRQ(ierr);
        ierr = MatMPIAIJSetPreallocation(M, nnz_per_row, NULL, nnz_per_row, NULL); CHKERRQ(ierr);
        ierr = MatSetUp(M); CHKERRQ(ierr);
    }
    else
    {
        ierr = MatCreate(PETSC_COMM_SELF, &M); CHKERRQ(ierr);
        ierr = MatSetSizes(M, PETSC_DECIDE, PETSC_DECIDE, sim.bspline_data.value("n_basis",0), sim.bspline_data.value("n_basis",0)); CHKERRQ(ierr);
        ierr = MatSetFromOptions(M); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(M, nnz_per_row, NULL); CHKERRQ(ierr);
        ierr = MatSetUp(M); CHKERRQ(ierr);
    }
    
    int start_row,end_row;
    if (use_mpi) {
        ierr = MatGetOwnershipRange(M, &start_row, &end_row); CHKERRQ(ierr);
    } else {
        start_row = 0;
        end_row = sim.bspline_data.value("n_basis", 0);
    }

    for (int i = start_row; i < end_row; i++) 
    {
        int col_start = std::max(0, i - sim.bspline_data.value("order",0) + 1);
        int col_end = std::min(sim.bspline_data.value("n_basis",0), i + sim.bspline_data.value("order",0));

        for (int j = col_start; j < col_end; j++) 
        {
            std::complex<double> result = bsplines::integrate_matrix_element(i, j, integrand, sim,use_ecs);
            ierr = MatSetValue(M, i, j, result, INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = MatAssemblyBegin(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    ierr = MatAssemblyEnd(M, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
    return ierr;
}

PetscErrorCode construct_overlap(const simulation& sim, Mat& S,bool use_mpi,bool use_ecs)
{
    return construct_matrix(sim, S, bsplines::overlap_integrand, use_mpi,use_ecs);
}

PetscErrorCode construct_kinetic(const simulation& sim, Mat& K,bool use_mpi,bool use_ecs)
{
    return construct_matrix(sim, K, bsplines::kinetic_integrand, use_mpi,use_ecs);
}

PetscErrorCode construct_invr(const simulation& sim, Mat& Inv_r,bool use_mpi,bool use_ecs)
{
    return construct_matrix(sim, Inv_r, bsplines::invr_integrand, use_mpi,use_ecs);
}

PetscErrorCode construct_invr2(const simulation& sim, Mat& Inv_r2,bool use_mpi,bool use_ecs)
{
    return construct_matrix(sim, Inv_r2, bsplines::invr2_integrand, use_mpi,use_ecs);
}

PetscErrorCode construct_der(const simulation& sim, Mat& D,bool use_mpi,bool use_ecs)
{
    return construct_matrix(sim, D, bsplines::der_integrand, use_mpi,use_ecs);
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
