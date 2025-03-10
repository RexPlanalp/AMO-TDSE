#pragma once

#include <petscvec.h>
#include "matrix.h"

enum class VectorType
{
    SEQUENTIAL,
    PARALLEL
};

class PetscVector 
{
    public:
        PetscVector(int size, VectorType type);
        PetscVector() = default;
        ~PetscVector();

        Vec& getVector();

        void setValue(int i, std::complex<double> value);

        void assemble();

        void computeNorm(std::complex<double>& norm, PetscMatrix& S);

        template <typename T>
        void scale(T factor);

    private:
        Vec vector;
};






// PetscErrorCode extract_normalized_eigenvector(Vec& eigenvector,const EPS& eps, const Mat& S, int i)
// {   
//     PetscErrorCode ierr;
//     ierr = MatCreateVecs(S,&eigenvector, NULL); CHKERRQ(ierr);
//     ierr = EPSGetEigenvector(eps,i,eigenvector,NULL); CHKERRQ(ierr);

//     std::complex<double> norm;
//     ierr = compute_eigenvector_norm(eigenvector,S,norm); CHKERRQ(ierr);
//     ierr = VecScale(eigenvector,1.0/norm.real()); CHKERRQ(ierr);
//     return ierr;
// }
