

#include "matrix.h"
#include "tise.h"

#include "bsplines.h"
#include <complex>
#include <string>
#include <iostream>

tise::tise(std::string& filename) : data(filename)
{
}

matrix tise::compute_overlap_matrix(std::string matrix_type,bsplines basis)
{

    matrix S;
    S.set_nnz(2*bspline_data["degree"].get<int>() +1);
    S.set_quantum_type("radial");
    S.set_type(matrix_type);
    S.setup_matrix(basis);

    for (int i = S.local_range[0]; i < S.local_range[1]; ++i)
    {
        int col_start = std::max(0, i - bspline_data["order"].get<int>() + 1);
        int col_end = std::min(bspline_data["n_basis"].get<int>(), i + bspline_data["order"].get<int>()); // Exclusive

        for (int j = col_start; j<col_end; ++j)
        {
            std::complex<double> matrix_element= basis.integrate_matrix_element
            (i, j, [&basis](int i, int j, std::complex<double> x) -> std::complex<double> 
            { 
                    return basis.overlap_integrand(i, j, x); 
            }
            );
            MatSetValue(S.petsc_mat, i, j, matrix_element, INSERT_VALUES); 
        }
    }

    S.assemble_matrix();
    return S;


}