#include <iostream>
#include <fstream>
#include <sys/stat.h>
#include <sys/types.h>

#include <petscmat.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>
#include <petscksp.h>

#include "tdse.h"
#include "simulation.h"
#include "laser.h"

#include "petsc_wrappers/PetscVector.h"
#include "utility.h"
#include "petsc_wrappers/PetscFileViewer.h"
#include "petsc_wrappers/PetscMatrix.h"

using Vec3 = std::array<double, 3>;

namespace tdse 
{

    Wavefunction load_starting_state(const simulation& sim)
    {   
        PetscErrorCode ierr;

        int n_basis = sim.bspline_params.n_basis;
        int n_blocks = sim.angular_params.n_blocks;
        std::array<int, 3> state = sim.schrodinger_params.state;
        int block_idx = sim.angular_params.lm_to_block.at({state[1], state[2]});
        int total_size = n_basis * n_blocks;

        Wavefunction starting_state(total_size,RunMode::PARALLEL);
        ierr = VecSet(starting_state.vector, 0.0); checkErr(ierr,"Error zeroing vector");

        Wavefunction tise_state(n_basis,RunMode::SEQUENTIAL);
        ierr = VecSet(tise_state.vector, 0.0); checkErr(ierr,"Error zeroing vector");

        std::stringstream ss;
        ss << "eigenvectors/psi_" << state[0] << "_" << state[1];
        std::string state_name = ss.str();

        tise_state.setName(state_name.c_str());

        PetscHDF5Viewer TISEViewer((sim.tise_output_path+"/tise_output.h5").c_str(),RunMode::SEQUENTIAL,OpenMode::READ);

        ierr = VecLoad(tise_state.vector,TISEViewer.viewer); checkErr(ierr, "Error loading TISE state");

        const PetscScalar *tise_state_array;
        ierr = VecGetArrayRead(tise_state.vector,&tise_state_array); checkErr(ierr, "Error getting array");

        for (int local_idx = 0; local_idx < n_basis; local_idx++)
        {
            int global_idx = block_idx * n_basis + local_idx;
            if (global_idx >= starting_state.local_start && global_idx < starting_state.local_end)
            {
                ierr = VecSetValue(starting_state.vector,global_idx, tise_state_array[local_idx],INSERT_VALUES); checkErr(ierr, "Error setting value");
            }
        }

        starting_state.assemble();
        ierr = VecRestoreArrayRead(tise_state.vector,&tise_state_array); checkErr(ierr, "Error restoring array");

        return starting_state;
    }

    PetscErrorCode load_matrix(std::string filename, Mat *matrix)
    {
        PetscErrorCode ierr;
        PetscViewer viewer;

        ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,filename.c_str(), FILE_MODE_READ,&viewer); CHKERRQ(ierr);

        ierr = MatCreate(PETSC_COMM_SELF,matrix); CHKERRQ(ierr);
        ierr = MatSetType(*matrix,MATSEQAIJ); CHKERRQ(ierr);

        ierr = MatLoad(*matrix,viewer); CHKERRQ(ierr);

        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
        return ierr;
    }

    PetscErrorCode KroneckerProductParallel(Mat A, Mat B, Mat *C_out) {
        PetscErrorCode ierr;


        // Get matrix dimensions
        PetscInt am, an, bm, bn;
        ierr = MatGetSize(A, &am, &an); CHKERRQ(ierr);
        ierr = MatGetSize(B, &bm, &bn); CHKERRQ(ierr);


        // Compute dimensions of Kronecker product matrix
        PetscInt cm = am * bm;
        PetscInt cn = an * bn;


        // Create parallel matrix C
        Mat C;
        ierr = MatCreate(PETSC_COMM_WORLD, &C); CHKERRQ(ierr);
        ierr = MatSetSizes(C, PETSC_DECIDE, PETSC_DECIDE, cm, cn); CHKERRQ(ierr);
        ierr = MatSetType(C, MATMPIAIJ); CHKERRQ(ierr);
        ierr = MatSetOption(C, MAT_NO_OFF_PROC_ENTRIES, PETSC_TRUE); CHKERRQ(ierr);


        // Get ownership range for rows in C
        PetscInt startC, endC;
        ierr = MatGetOwnershipRange(C, &startC, &endC); CHKERRQ(ierr);


        // Access internal data arrays of A and B
        const PetscInt *ai, *aj, *bi, *bj;
        const PetscScalar *aa, *ba;
        ierr = MatGetRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &am, &ai, &aj, NULL); CHKERRQ(ierr);
        ierr = MatGetRowIJ(B, 0, PETSC_FALSE, PETSC_FALSE, &bm, &bi, &bj, NULL); CHKERRQ(ierr);
        ierr = MatSeqAIJGetArrayRead(A, &aa); CHKERRQ(ierr);
        ierr = MatSeqAIJGetArrayRead(B, &ba); CHKERRQ(ierr);


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


        ierr = MatMPIAIJSetPreallocationCSR(C, ci, cj, cv); CHKERRQ(ierr);


        // Clean up
        delete[] ci;
        delete[] cj;
        delete[] cv;


        ierr = MatRestoreRowIJ(A, 0, PETSC_FALSE, PETSC_FALSE, &am, &ai, &aj, NULL); CHKERRQ(ierr);
        ierr = MatRestoreRowIJ(B, 0, PETSC_FALSE, PETSC_FALSE, &bm, &bi, &bj, NULL); CHKERRQ(ierr);
        ierr = MatSeqAIJRestoreArrayRead(A, &aa); CHKERRQ(ierr);
        ierr = MatSeqAIJRestoreArrayRead(B, &ba); CHKERRQ(ierr);


        // Assemble the matrix
        ierr = MatAssemblyBegin(C, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(C, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


        *C_out = C;
        return ierr;
    }

    PetscErrorCode _construct_S_atomic(const simulation& sim, Mat& S_atomic)
    {
        PetscErrorCode ierr;
        int n_blocks = sim.angular_params.n_blocks;

        PetscBinaryViewer SViewer((sim.tise_output_path+"/S.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix S = SViewer.loadMatrix();

    
        PetscMatrix I;


        ierr = MatCreate(PETSC_COMM_SELF,&I); CHKERRQ(ierr);
        ierr = MatSetSizes(I,PETSC_DECIDE,PETSC_DECIDE,n_blocks,n_blocks); CHKERRQ(ierr);
        ierr = MatSetType(I,MATSEQAIJ); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(I,1,NULL); CHKERRQ(ierr);
        for (int i = 0; i < n_blocks; ++i)
        {
            ierr = MatSetValue(I,i,i,1.0,INSERT_VALUES); CHKERRQ(ierr);
        }
        ierr = MatAssemblyBegin(I,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(I,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        ierr = KroneckerProductParallel(I,S,&S_atomic); CHKERRQ(ierr);

        ierr = MatDestroy(&S); CHKERRQ(ierr);
        ierr = MatDestroy(&I); CHKERRQ(ierr);
        return ierr;
    }

    PetscErrorCode _construct_H_atomic(const simulation& sim, Mat& H_atomic)
    {
        PetscErrorCode ierr; 

        int n_blocks = sim.angular_params.n_blocks;
        int n_basis = sim.bspline_params.n_basis;
        int degree = sim.bspline_params.degree;
        int lmax = sim.angular_params.lmax;

        Mat K,Inv_r2,Inv_r;
        ierr = load_matrix(sim.tise_output_path+"/K.bin",&K); CHKERRQ(ierr);
        ierr = load_matrix(sim.tise_output_path+"/Inv_r2.bin",&Inv_r2); CHKERRQ(ierr);
        ierr = load_matrix(sim.tise_output_path+"/Inv_r.bin",&Inv_r); CHKERRQ(ierr);

        ierr = MatAXPY(K,-1.0,Inv_r,SAME_NONZERO_PATTERN); CHKERRQ(ierr);

        Mat H_partial,temp;

        ierr = MatCreate(PETSC_COMM_WORLD,&H_atomic); CHKERRQ(ierr);
        ierr = MatSetSizes(H_atomic,PETSC_DECIDE,PETSC_DECIDE,n_blocks*n_basis,n_blocks*n_basis); CHKERRQ(ierr);
        ierr = MatSetType(H_atomic,MATMPIAIJ); CHKERRQ(ierr);
        ierr = MatMPIAIJSetPreallocation(H_atomic,2*degree+1,NULL,2*degree+1,NULL); CHKERRQ(ierr);
        ierr = MatAssemblyBegin(H_atomic,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(H_atomic,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        for (int l = 0; l<=lmax; ++l)
        {
            ierr = MatDuplicate(K,MAT_COPY_VALUES,&temp); CHKERRQ(ierr);
            ierr = MatAXPY(temp,l*(l+1)*0.5,Inv_r2,SAME_NONZERO_PATTERN); CHKERRQ(ierr);

            std::vector<int> indices;
            for (int i = 0; i < n_blocks; ++i)
            {   
                auto lm_pair = sim.angular_params.block_to_lm.at(i);  // Retrieve (l, m) pair
                if (lm_pair.first == l) // Compare only 'l' component
                {
                    indices.push_back(i);
                }
            }

            Mat I_partial;
            ierr = MatCreate(PETSC_COMM_SELF,&I_partial); CHKERRQ(ierr);
            ierr = MatSetSizes(I_partial,PETSC_DECIDE,PETSC_DECIDE,n_blocks,n_blocks); CHKERRQ(ierr);
            ierr = MatSetType(I_partial,MATSEQAIJ); CHKERRQ(ierr);
            ierr = MatSeqAIJSetPreallocation(I_partial,1,NULL); CHKERRQ(ierr);

            for (int i = 0; i<n_blocks; ++i)
            {
                if (std::find(indices.begin(),indices.end(),i) != indices.end())
                {
                    ierr = MatSetValue(I_partial,i,i,1.0,INSERT_VALUES); CHKERRQ(ierr);
                }
            }
            ierr = MatAssemblyBegin(I_partial,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
            ierr = MatAssemblyEnd(I_partial,MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

            ierr = KroneckerProductParallel(I_partial,temp,&H_partial); CHKERRQ(ierr);
            ierr = MatDestroy(&temp); CHKERRQ(ierr);
            ierr = MatDestroy(&I_partial); CHKERRQ(ierr);

            ierr = MatAXPY(H_atomic,1.0,H_partial,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
            ierr = MatDestroy(&H_partial); CHKERRQ(ierr);
        }

        return ierr;






    }

    PetscErrorCode _construct_atomic_interaction(const simulation& sim,Mat& H_atomic,Mat& S_atomic, Mat& atomic_left,Mat& atomic_right)
    {   
        double dt = sim.grid_params.dt;
        std::complex<double> alpha  = PETSC_i * (dt / 2.0);
        PetscErrorCode ierr;
        ierr = MatDuplicate(S_atomic,MAT_COPY_VALUES,&atomic_left); CHKERRQ(ierr);
        ierr = MatDuplicate(S_atomic,MAT_COPY_VALUES,&atomic_right); CHKERRQ(ierr);
        ierr = MatAXPY(atomic_left,alpha,H_atomic,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
        ierr = MatAXPY(atomic_right,-alpha,H_atomic,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
        return ierr;
    }

    double _a(int l, int m)
    {
        int numerator = (l+m);
        int denominator = (2*l +1) * (2*l-1);
        double f1 = sqrt(numerator/(double)denominator);
        double f2 = - m * std::sqrt(l+m-1) - std::sqrt((l-m)*(l*(l-1)-m*(m-1)));
        return f1*f2;
        
    }

    double _atilde(int l, int m)
    {
        int numerator = (l-m);
        int denominator = (2*l+1)*(2*l-1);
        double f1 = sqrt(numerator/(double)denominator);
        double f2 = - m * std::sqrt(l-m-1) + std::sqrt((l+m)*(l*(l-1)-m*(m+1)));
        return f1*f2;
    }

    double _b(int l, int m)
    {
        return -_atilde(l+1,m-1);
    }

    double _btilde(int l, int m)
    {
        return -_a(l+1,m+1);
    }

    double _d(int l, int m)
    {
        double numerator = (l-m+1)*(l-m+2);
        double denominator = (2*l+1)*(2*l+3);
        return std::sqrt(numerator/(double)denominator);
    }

    double _dtilde(int l, int m)
    {
        return _d(l,-m);
    }

    double _c(int l, int m)
    {
        return _dtilde(l-1,m-1);
    }

    double _ctilde(int l, int m)
    {
        return _d(l-1,m+1);
    }

    PetscErrorCode _construct_xy_interaction(const simulation& sim, Mat& H_xy, Mat& H_xy_tilde)
    {
        PetscErrorCode ierr;

        int n_blocks = sim.angular_params.n_blocks;

        Mat Der,Inv_r;
        ierr = load_matrix(sim.tise_output_path+"/Der.bin",&Der); CHKERRQ(ierr);
        ierr = load_matrix(sim.tise_output_path+"/Inv_r.bin",&Inv_r); CHKERRQ(ierr);

        Mat H_lm_1;
        ierr = MatCreate(PETSC_COMM_SELF,&H_lm_1); CHKERRQ(ierr);
        ierr = MatSetSizes(H_lm_1,PETSC_DECIDE,PETSC_DECIDE,n_blocks,n_blocks); CHKERRQ(ierr);
        ierr = MatSetFromOptions(H_lm_1); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(H_lm_1,2,NULL); CHKERRQ(ierr);
        ierr = MatSetUp(H_lm_1); CHKERRQ(ierr);
        for (int i = 0; i < n_blocks; ++i)
        {
            std::pair<int,int> lm_pair = sim.angular_params.block_to_lm.at(i);
            int l = lm_pair.first;
            int m = lm_pair.second;
            for (int j = 0; j < n_blocks; ++j)
            {
                std::pair<int,int> lm_pair_prime = sim.angular_params.block_to_lm.at(j);
                int lprime = lm_pair_prime.first;
                int mprime = lm_pair_prime.second;

                if ((l == lprime+1) && (m == mprime+1))
                {
                    ierr = MatSetValue(H_lm_1, i, j, PETSC_i*_a(l,m)/2, INSERT_VALUES); CHKERRQ(ierr);
                }
                else if ((l == lprime-1)&&(m == mprime+1))
                {
                    ierr = MatSetValue(H_lm_1, i, j, PETSC_i*_b(l,m)/2, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(H_lm_1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(H_lm_1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


        Mat H_lm_2;
        ierr = MatCreate(PETSC_COMM_SELF,&H_lm_2); CHKERRQ(ierr);
        ierr = MatSetSizes(H_lm_2,PETSC_DECIDE,PETSC_DECIDE,n_blocks,n_blocks); CHKERRQ(ierr);
        ierr = MatSetFromOptions(H_lm_2); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(H_lm_2,2,NULL); CHKERRQ(ierr);
        ierr = MatSetUp(H_lm_2); CHKERRQ(ierr);
        for (int i = 0; i < n_blocks; ++i)
        {
            std::pair<int,int> lm_pair = sim.angular_params.block_to_lm.at(i);
            int l = lm_pair.first;
            int m = lm_pair.second;
            for (int j = 0; j < n_blocks; ++j)
            {
                std::pair<int,int> lm_pair_prime = sim.angular_params.block_to_lm.at(j);
                int lprime = lm_pair_prime.first;
                int mprime = lm_pair_prime.second;

                if ((l == lprime+1) && (m == mprime+1))
                {
                    ierr = MatSetValue(H_lm_2, i, j, PETSC_i*_c(l,m)/2, INSERT_VALUES); CHKERRQ(ierr);
                }
                else if ((l == lprime-1)&&(m == mprime+1))
                {
                    ierr = MatSetValue(H_lm_2, i, j, -PETSC_i*_d(l,m)/2, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(H_lm_2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(H_lm_2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        Mat H_xy_temp_1;
        ierr = KroneckerProductParallel(H_lm_1,Inv_r,&H_xy); CHKERRQ(ierr);      
        ierr = KroneckerProductParallel(H_lm_2,Der,&H_xy_temp_1); CHKERRQ(ierr);
        ierr = MatAXPY(H_xy,1.0,H_xy_temp_1,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
        ierr = MatDestroy(&H_lm_1); CHKERRQ(ierr);
        ierr = MatDestroy(&H_lm_2); CHKERRQ(ierr);
        ierr = MatDestroy(&H_xy_temp_1); CHKERRQ(ierr);


        Mat H_lm_3;
        ierr = MatCreate(PETSC_COMM_SELF,&H_lm_3); CHKERRQ(ierr);
        ierr = MatSetSizes(H_lm_3,PETSC_DECIDE,PETSC_DECIDE,n_blocks,n_blocks); CHKERRQ(ierr);
        ierr = MatSetFromOptions(H_lm_3); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(H_lm_3,2,NULL); CHKERRQ(ierr);
        ierr = MatSetUp(H_lm_3); CHKERRQ(ierr);
        for (int i = 0; i < n_blocks; ++i)
        {
            std::pair<int,int> lm_pair = sim.angular_params.block_to_lm.at(i);
            int l = lm_pair.first;
            int m = lm_pair.second;
            for (int j = 0; j < n_blocks; ++j)
            {
                std::pair<int,int> lm_pair_prime = sim.angular_params.block_to_lm.at(j);
                int lprime = lm_pair_prime.first;
                int mprime = lm_pair_prime.second;

                if ((l == lprime+1) && (m == mprime-1))
                {
                    ierr = MatSetValue(H_lm_3, i, j, PETSC_i*_atilde(l,m)/2, INSERT_VALUES); CHKERRQ(ierr);
                }
                else if ((l == lprime-1)&&(m == mprime-1))
                {
                    ierr = MatSetValue(H_lm_3, i, j, PETSC_i*_btilde(l,m)/2, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(H_lm_3, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(H_lm_3, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        Mat H_lm_4;
        ierr = MatCreate(PETSC_COMM_SELF,&H_lm_4); CHKERRQ(ierr);
        ierr = MatSetSizes(H_lm_4,PETSC_DECIDE,PETSC_DECIDE,n_blocks,n_blocks); CHKERRQ(ierr);
        ierr = MatSetFromOptions(H_lm_4); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(H_lm_4,2,NULL); CHKERRQ(ierr);
        ierr = MatSetUp(H_lm_4); CHKERRQ(ierr);
        for (int i = 0; i < n_blocks; ++i)
        {
            std::pair<int,int> lm_pair = sim.angular_params.block_to_lm.at(i);
            int l = lm_pair.first;
            int m = lm_pair.second;
            for (int j = 0; j < n_blocks; ++j)
            {
                std::pair<int,int> lm_pair_prime = sim.angular_params.block_to_lm.at(j);
                int lprime = lm_pair_prime.first;
                int mprime = lm_pair_prime.second;

                if ((l == lprime+1) && (m == mprime-1))
                {
                    ierr = MatSetValue(H_lm_4, i, j, -PETSC_i*_ctilde(l,m)/2, INSERT_VALUES); CHKERRQ(ierr);
                }
                else if ((l == lprime-1)&&(m == mprime-1))
                {
                    ierr = MatSetValue(H_lm_4, i, j, PETSC_i*_dtilde(l,m)/2, INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(H_lm_4, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(H_lm_4, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

        Mat H_xy_temp_2;
        ierr = KroneckerProductParallel(H_lm_3,Inv_r,&H_xy_tilde); CHKERRQ(ierr);
        ierr = KroneckerProductParallel(H_lm_4,Der,&H_xy_temp_2); CHKERRQ(ierr);
        ierr = MatAXPY(H_xy_tilde,1.0,H_xy_temp_2,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
        ierr = MatDestroy(&H_lm_3); CHKERRQ(ierr);
        ierr = MatDestroy(&H_lm_4); CHKERRQ(ierr);
        ierr = MatDestroy(&H_xy_temp_2); CHKERRQ(ierr);

        ierr = MatDestroy(&Der); CHKERRQ(ierr);
        ierr = MatDestroy(&Inv_r); CHKERRQ(ierr);
        return ierr;

    }

    double _f(int l, int m)
    {   
        int numerator = (l+1)*(l+1) - m*m;
        int denominator = (2*l + 1)*(2*l+3);
        return sqrt(numerator/(double)denominator);
    }

    double _g(int l, int m)
    {
        int numerator = l*l - m*m;
        int denominator = (2*l-1)*(2*l+1);
        return sqrt(numerator/(double)denominator);
    }

    PetscErrorCode _construct_z_interaction(const simulation& sim, Mat& H_z)
    {
        PetscErrorCode ierr;

        int n_blocks = sim.angular_params.n_blocks;

        Mat H_lm_1;
        ierr = MatCreate(PETSC_COMM_SELF, &H_lm_1); CHKERRQ(ierr);
        ierr = MatSetSizes(H_lm_1, PETSC_DECIDE, PETSC_DECIDE, n_blocks, n_blocks); CHKERRQ(ierr);
        ierr = MatSetFromOptions(H_lm_1); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(H_lm_1, 2, NULL); CHKERRQ(ierr);
        ierr = MatSetUp(H_lm_1); CHKERRQ(ierr);
        for (int i = 0; i < n_blocks; ++i)
        {
            std::pair<int,int> lm_pair = sim.angular_params.block_to_lm.at(i);
            int l = lm_pair.first;
            int m = lm_pair.second;
            for (int j = 0; j < n_blocks; ++j)
            {
                std::pair<int,int> lm_pair_prime = sim.angular_params.block_to_lm.at(j);
                int lprime = lm_pair_prime.first;
                int mprime = lm_pair_prime.second;

                if ((l == lprime+1) && (m == mprime))
                {
                    ierr = MatSetValue(H_lm_1, i, j, -PETSC_i * _g(l,m), INSERT_VALUES); CHKERRQ(ierr);
                }
                else if ((l == lprime-1)&&(m == mprime))
                {
                    ierr = MatSetValue(H_lm_1, i, j, -PETSC_i * _f(l,m), INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(H_lm_1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(H_lm_1, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);




        Mat H_lm_2;
        ierr = MatCreate(PETSC_COMM_SELF, &H_lm_2); CHKERRQ(ierr);
        ierr = MatSetSizes(H_lm_2, PETSC_DECIDE, PETSC_DECIDE, n_blocks, n_blocks); CHKERRQ(ierr);
        ierr = MatSetFromOptions(H_lm_2); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(H_lm_2, 2, NULL); CHKERRQ(ierr);
        ierr = MatSetUp(H_lm_2); CHKERRQ(ierr);
        for (int i = 0; i < n_blocks; ++i)
        {
            std::pair<int,int> lm_pair = sim.angular_params.block_to_lm.at(i);
            int l = lm_pair.first;
            int m = lm_pair.second;
            for (int j = 0; j < n_blocks; ++j)
            {
                std::pair<int,int> lm_pair_prime = sim.angular_params.block_to_lm.at(j);
                int lprime = lm_pair_prime.first;
                int mprime = lm_pair_prime.second;

                if ((l == lprime+1) && (m == mprime))
                {
                    ierr = MatSetValue(H_lm_2, i, j, -PETSC_i * _g(l,m) * (-l), INSERT_VALUES); CHKERRQ(ierr);
                }
                else if ((l == lprime-1)&&(m == mprime))
                {
                    ierr = MatSetValue(H_lm_2, i, j, -PETSC_i * _f(l,m) * (l+1), INSERT_VALUES); CHKERRQ(ierr);
                }
            }
        }
        ierr = MatAssemblyBegin(H_lm_2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
        ierr = MatAssemblyEnd(H_lm_2, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);


        Mat Der,Inv_r;
        ierr = load_matrix(sim.tise_output_path+"/Der.bin",&Der); CHKERRQ(ierr);
        ierr = load_matrix(sim.tise_output_path+"/Inv_r.bin",&Inv_r); CHKERRQ(ierr);

        Mat H_z_2;
        ierr = KroneckerProductParallel(H_lm_1,Der,&H_z); CHKERRQ(ierr);
        ierr = KroneckerProductParallel(H_lm_2,Inv_r,&H_z_2); CHKERRQ(ierr);

        ierr = MatAXPY(H_z,1.0,H_z_2,SAME_NONZERO_PATTERN); CHKERRQ(ierr);

        ierr = MatDestroy(&H_lm_1); CHKERRQ(ierr);
        ierr = MatDestroy(&H_lm_2); CHKERRQ(ierr);
        ierr = MatDestroy(&Der); CHKERRQ(ierr);
        ierr = MatDestroy(&Inv_r); CHKERRQ(ierr);
        ierr = MatDestroy(&H_z_2); CHKERRQ(ierr);
        return ierr;
    }


PetscErrorCode solve_tdse(const simulation& sim, int rank)
{   
    double time_start = MPI_Wtime();    

    PetscPrintf(PETSC_COMM_WORLD, "Loading Information\n\n");
    PetscErrorCode ierr;
    
    Wavefunction state = load_starting_state(sim); CHKERRQ(ierr);
    Vec3 components = sim.laser_params.components;
    double dt = sim.grid_params.dt;
    int Nt = sim.grid_params.Nt;
    std::complex<double> alpha = PETSC_i * (dt / 2.0);

    create_directory(rank, sim.tdse_output_path);

    PetscHDF5Viewer viewTDSE((sim.tdse_output_path+"/tdse_output.h5").c_str(),RunMode::PARALLEL,OpenMode::WRITE);

    


    PetscPrintf(PETSC_COMM_WORLD, "Constructing Atomic Interaction\n\n");
    Mat H_atomic, S_atomic, atomic_left, atomic_right;
    ierr = _construct_S_atomic(sim, S_atomic); CHKERRQ(ierr);
    ierr = _construct_H_atomic(sim, H_atomic); CHKERRQ(ierr);
    ierr = _construct_atomic_interaction(sim, H_atomic, S_atomic, atomic_left, atomic_right); CHKERRQ(ierr);

    Mat H_z;
    if (components[2]) 
    {   
        PetscPrintf(PETSC_COMM_WORLD, "Constructing Z Interaction\n\n");
        ierr = _construct_z_interaction(sim, H_z); CHKERRQ(ierr);
    }
    Mat H_xy, H_xy_tilde;
    if (components[0] || components[1]) 
    {
        PetscPrintf(PETSC_COMM_WORLD, "Constructing XY Interaction\n\n");
        ierr = _construct_xy_interaction(sim, H_xy, H_xy_tilde); CHKERRQ(ierr);
    }

    PetscPrintf(PETSC_COMM_WORLD, "Computing Norm...\n\n");
    std::complex<double> norm;
    Vec y; 
    ierr = VecDuplicate(state.vector, &y); CHKERRQ(ierr);
    ierr = MatMult(S_atomic, state.vector, y); CHKERRQ(ierr);
    ierr = VecDot(state.vector, y, &norm); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Norm of Initial State: (%.4f,%.4f)\n\n", norm.real(), norm.imag());

    PetscPrintf(PETSC_COMM_WORLD, "Setting up Linear Solver\n\n");
    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, sim.schrodinger_params.tdse_tol, PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "Preallocating Temporary Petsc Objects\n\n");
    Vec state_temp;
    Mat atomic_left_temp, atomic_right_temp;
    ierr = MatDuplicate(atomic_left, MAT_COPY_VALUES, &atomic_left_temp); CHKERRQ(ierr);
    ierr = MatDuplicate(atomic_right, MAT_COPY_VALUES, &atomic_right_temp); CHKERRQ(ierr);
    ierr = VecDuplicate(state.vector, &state_temp); CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "Solving TDSE\n\n");
    for (int idx = 0; idx < Nt; ++idx) 
    {   
        if (sim.debug)
        {
            PetscPrintf(PETSC_COMM_WORLD, "Time Step: %d/%d\n", idx,Nt);
        }
        
        double t = idx * dt;

        // Destroy and recreate temp matrices to avoid accumulation of structural changes
        ierr = MatDestroy(&atomic_left_temp); CHKERRQ(ierr);
        ierr = MatDestroy(&atomic_right_temp); CHKERRQ(ierr);
        ierr = MatDuplicate(atomic_left, MAT_COPY_VALUES, &atomic_left_temp); CHKERRQ(ierr);
        ierr = MatDuplicate(atomic_right, MAT_COPY_VALUES, &atomic_right_temp); CHKERRQ(ierr);

        if (components[2]) 
        {

            double laser_val = laser::A(t+dt/2.0, sim, 2);
            ierr = MatAXPY(atomic_left_temp, alpha * laser_val, H_z, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
            ierr = MatAXPY(atomic_right_temp, -alpha * laser_val, H_z, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
        }
        else if (components[0] || components[1])
        {
            std::complex<double> A_tilde = laser::A(t+dt/2.0, sim, 0) + PETSC_i*laser::A(t+dt/2.0, sim, 1);
            std::complex<double> A_tilde_star = laser::A(t+dt/2.0, sim, 0) - PETSC_i*laser::A(t+dt/2.0, sim, 1);

            ierr = MatAXPY(atomic_left_temp,alpha*A_tilde_star,H_xy,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
            ierr = MatAXPY(atomic_right_temp,-alpha*A_tilde_star,H_xy,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);

            ierr = MatAXPY(atomic_left_temp,alpha*A_tilde,H_xy_tilde,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
            ierr = MatAXPY(atomic_right_temp,-alpha*A_tilde,H_xy_tilde,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);

        }

        ierr = MatMult(atomic_right_temp, state.vector, state_temp); CHKERRQ(ierr);

        // Reuse KSP operator
        ierr = KSPSetOperators(ksp, atomic_left_temp, atomic_left_temp); CHKERRQ(ierr);
        ierr = KSPSetReusePreconditioner(ksp, PETSC_TRUE); CHKERRQ(ierr);

        ierr = KSPSolve(ksp, state_temp, state.vector); CHKERRQ(ierr);

        
    }

    PetscPrintf(PETSC_COMM_WORLD, "Computing Norm...\n\n");
    ierr = MatMult(S_atomic, state.vector, y); CHKERRQ(ierr);
    ierr = VecDot(state.vector, y, &norm); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Norm of Final State: (%.15f,%.15f)\n\n", (double)norm.real(), (double)norm.imag());

    ierr = PetscObjectSetName((PetscObject)state.vector,"final_state"); CHKERRQ(ierr);
    ierr = VecView(state.vector, viewTDSE.viewer); CHKERRQ(ierr);

    PetscPrintf(PETSC_COMM_WORLD, "Destroying Petsc Objects\n\n");
    ierr = VecDestroy(&y); CHKERRQ(ierr);
    ierr = VecDestroy(&state_temp); CHKERRQ(ierr);
    ierr = MatDestroy(&atomic_left_temp); CHKERRQ(ierr);
    ierr = MatDestroy(&atomic_right_temp); CHKERRQ(ierr);
    if (components[0] || components[1]) 
    {
        ierr = MatDestroy(&H_xy); CHKERRQ(ierr);
        ierr = MatDestroy(&H_xy_tilde); CHKERRQ(ierr);
    }
    if (components[2]) 
    {
        ierr = MatDestroy(&H_z); CHKERRQ(ierr);
    }
    ierr = MatDestroy(&H_atomic); CHKERRQ(ierr);
    ierr = MatDestroy(&S_atomic); CHKERRQ(ierr);
    ierr = MatDestroy(&atomic_left); CHKERRQ(ierr);
    ierr = MatDestroy(&atomic_right); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
    double time_end = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD,"Time to solve TDSE %.3f\n",time_end-time_start);

    return ierr;
}

}
