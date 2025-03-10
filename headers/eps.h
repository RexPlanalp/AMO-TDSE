#pragma once

#include "simulation.h"
#include <slepceps.h>
#include "matrix.h"
#include "vector.h"

class PetscEPS
{


    public:
        PetscEPS(const simulation& sim);
        ~PetscEPS();

        void setParameters(int num_of_energies);
        void setOperators(PetscMatrix& H, PetscMatrix& S);
        void solve(int& nconv);

        void getEigenvalue(int i,std::complex<double>& eigenvalue);

        void getNormalizedEigenvector(PetscVector& eigenvector, int i, PetscMatrix& S);




        EPS eps;



};
