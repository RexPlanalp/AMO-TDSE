

#include "matrix.h"
#include "tise.h"

#include <string>

matrix tise::compute_overlap_matrix(std::string matrix_type)
{

    matrix S;
    S.set_nnz(2*bspline_data["degree"].get<int>() +1);
    S.set_quantum_type("radial");
    S.set_type(matrix_type);


}