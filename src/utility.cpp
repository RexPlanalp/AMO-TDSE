#include <iostream>

#include <petscsys.h>

#include "utility.h"
#include "petsc_wrappers/PetscMatrix.h"
#include "petsc_wrappers/PetscVector.h"
#include "simulation.h"
#include "petsc_wrappers/PetscFileViewer.h"
#include "petsc_wrappers/PetscIS.h"

// Custom Error Checker
void checkErr(PetscErrorCode ierr, const char* msg)
{
    if (ierr)
    {
        std::cerr << "PETSc Error:" << msg << " (Code: " << ierr << ")" << std::endl;
        PetscError(PETSC_COMM_SELF, __LINE__, __func__, __FILE__, ierr, PETSC_ERROR_INITIAL,msg);
        throw std::runtime_error("PETSc Error");
    }
}

// Kronecker Product
PetscMatrix KroneckerProduct(const PetscMatrix& A, const PetscMatrix& B) 
{
    PetscErrorCode ierr;

    // Get matrix dimensions
    PetscInt am, an, bm, bn;
    ierr = MatGetSize(A.matrix, &am, &an); checkErr(ierr, "Error getting matrix size");
    ierr = MatGetSize(B.matrix, &bm, &bn); checkErr(ierr, "Error getting matrix size");


    // Compute dimensions of Kronecker product matrix
    PetscInt cm = am * bm;
    PetscInt cn = an * bn;


    // Create parallel matrix C
    PetscMatrix C;
    ierr = MatCreate(PETSC_COMM_WORLD, &C.matrix); checkErr(ierr, "Error creating matrix");
    ierr = MatSetSizes(C.matrix, PETSC_DECIDE, PETSC_DECIDE, cm, cn); checkErr(ierr, "Error setting matrix sizes");
    ierr = MatSetType(C.matrix, MATMPIAIJ); checkErr(ierr, "Error setting matrix type");
    ierr = MatSetOption(C.matrix, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE); checkErr(ierr, "Error setting matrix option");


    // Get ownership range for rows in C
    PetscInt startC, endC;
    ierr = MatGetOwnershipRange(C.matrix, &startC, &endC); checkErr(ierr, "Error getting ownership range");


    // Access internal data arrays of A and B
    const PetscInt *ai, *aj, *bi, *bj;
    const PetscScalar *aa, *ba;
    ierr = MatGetRowIJ(A.matrix, 0, PETSC_FALSE, PETSC_FALSE, &am, &ai, &aj, NULL); checkErr(ierr, "Error getting row IJ");
    ierr = MatGetRowIJ(B.matrix, 0, PETSC_FALSE, PETSC_FALSE, &bm, &bi, &bj, NULL); checkErr(ierr, "Error getting row IJ");
    ierr = MatSeqAIJGetArrayRead(A.matrix, &aa); checkErr(ierr, "Error getting array");
    ierr = MatSeqAIJGetArrayRead(B.matrix, &ba); checkErr(ierr, "Error getting array");


    // Preallocate matrix C for local rows
    PetscInt local_nnz = 0;
    PetscInt *ci = new PetscInt[endC - startC + 1];
    PetscInt *cj = nullptr;
    PetscScalar *cv = nullptr;


    ci[0] = 0;
    for (PetscInt iC = startC; iC < endC; ++iC) {
        PetscInt iA = iC / bm; // Row index in A
        PetscInt iB = iC % bm; // Row index in B


        for (PetscInt n = ai[iA]; n < ai[iA + 1]; ++n) {
            for (PetscInt q = bi[iB]; q < bi[iB + 1]; ++q) {
                local_nnz++;
            }
        }


        ci[iC - startC + 1] = local_nnz;
    }


    cj = new PetscInt[local_nnz];
    cv = new PetscScalar[local_nnz];


    local_nnz = 0;
    for (PetscInt iC = startC; iC < endC; ++iC) {
        PetscInt iA = iC / bm;
        PetscInt iB = iC % bm;


        for (PetscInt n = ai[iA]; n < ai[iA + 1]; ++n) {
            PetscInt colA = aj[n];
            PetscScalar valA = aa[n];


            for (PetscInt q = bi[iB]; q < bi[iB + 1]; ++q) {
                cj[local_nnz] = colA * bn + bj[q];
                cv[local_nnz] = valA * ba[q];
                local_nnz++;
            }
        }
    }


    ierr = MatMPIAIJSetPreallocationCSR(C.matrix, ci, cj, cv); checkErr(ierr, "Error setting preallocation");


    // Clean up
    delete[] ci;
    delete[] cj;
    delete[] cv;


    ierr = MatRestoreRowIJ(A.matrix, 0, PETSC_FALSE, PETSC_FALSE, &am, &ai, &aj, NULL); checkErr(ierr, "Error restoring row IJ");
    ierr = MatRestoreRowIJ(B.matrix, 0, PETSC_FALSE, PETSC_FALSE, &bm, &bi, &bj, NULL); checkErr(ierr, "Error restoring row IJ");
    ierr = MatSeqAIJRestoreArrayRead(A.matrix, &aa); checkErr(ierr, "Error restoring array");
    ierr = MatSeqAIJRestoreArrayRead(B.matrix, &ba); checkErr(ierr, "Error restoring array");


    // Assemble the matrix
    ierr = MatAssemblyBegin(C.matrix, MAT_FINAL_ASSEMBLY); checkErr(ierr, "Error assembling matrix");
    ierr = MatAssemblyEnd(C.matrix, MAT_FINAL_ASSEMBLY); checkErr(ierr, "Error assembling matrix");

    return C;
}

double f(int l, int m)
{   
    int numerator = (l+1)*(l+1) - m*m;
    int denominator = (2*l + 1)*(2*l+3);
    return sqrt(numerator/double(denominator));
}

double g(int l, int m)
{
    int numerator = l*l - m*m;
    int denominator = (2*l-1)*(2*l+1);
    return sqrt(numerator/double(denominator));
}

double a(int l, int m)
{
    int numerator = (l+m);
    int denominator = (2*l +1) * (2*l-1);
    double f1 = sqrt(numerator/double(denominator));
    double f2 = - m * std::sqrt(l+m-1) - std::sqrt((l-m)*(l*(l-1)-m*(m-1)));
    return f1*f2;
    
}

double atilde(int l, int m)
{
    int numerator = (l-m);
    int denominator = (2*l+1)*(2*l-1);
    double f1 = sqrt(numerator/double(denominator));
    double f2 = - m * std::sqrt(l-m-1) + std::sqrt((l+m)*(l*(l-1)-m*(m+1)));
    return f1*f2;
}

double b(int l, int m)
{
    return -atilde(l+1,m-1);
}

double btilde(int l, int m)
{
    return -a(l+1,m+1);
}

double d(int l, int m)
{
    double numerator = (l-m+1)*(l-m+2);
    double denominator = (2*l+1)*(2*l+3);
    return std::sqrt(numerator/double(denominator));
}

double dtilde(int l, int m)
{
    return d(l,-m);
}

double c(int l, int m)
{
    return dtilde(l-1,m-1);
}

double ctilde(int l, int m)
{
    return d(l-1,m+1);
}

// Function to project out bound states
PetscErrorCode project_out_bound(const PetscMatrix& S, PetscVector& state, const simulation& sim)
    {
      
    

        PetscErrorCode ierr;
        std::complex<double> inner_product;
        PetscBool has_dataset;
        

        PetscHDF5Viewer viewEigenvectors((sim.tise_output_path+"/tise_output.h5").c_str(),RunMode::SEQUENTIAL,OpenMode::READ);
        PetscVector state_block;


        for (int block = 0; block < sim.angular_params.n_blocks; block++)
        {
            std::pair<int, int> lm_pair = sim.angular_params.block_to_lm.at(block);
            int l = lm_pair.first;
            int start = block * sim.bspline_params.n_basis;

            PetscIS indexSet(sim.bspline_params.n_basis, start, 1, RunMode::SEQUENTIAL);
            
            ierr = VecGetSubVector(state.vector, indexSet.is, &state_block.vector); checkErr(ierr, "Error in VecGetSubVector");
            PetscVector temp(state_block);
            
            for (int n = 0; n <= sim.angular_params.nmax; ++n)
            {
                std::ostringstream dataset_ss;
                dataset_ss << "/eigenvectors" << "/psi_" << n << "_" << l;
                std::string dataset_name = dataset_ss.str();


                ierr = PetscViewerHDF5HasDataset(viewEigenvectors.viewer, dataset_name.c_str(), &has_dataset); checkErr(ierr, "Error in PetscViewerHDF5HasDataset");
                if (has_dataset)
                {   
                    PetscVector tise_state = viewEigenvectors.loadVector(sim.bspline_params.n_basis, "eigenvectors", dataset_name.c_str());

                    ierr = MatMult(S.matrix,state_block.vector,temp.vector); checkErr(ierr, "Error in MatMult");
                    ierr = VecDot(temp.vector,tise_state.vector,&inner_product); checkErr(ierr, "Error in VecDot");
                    ierr = VecAXPY(state_block.vector,-inner_product,tise_state.vector); checkErr(ierr, "Error in VecAXPY");
                }
            }
            
            

            ierr = VecRestoreSubVector(state.vector, indexSet.is, &state_block.vector); checkErr(ierr, "Error in VecRestoreSubVector");
        }
        return ierr;
    }