#include "tdse.h"
#include "simulation.h"
#include <iostream>
#include <fstream>
#include <petscmat.h>
#include <petscvec.h>
#include <petscviewerhdf5.h>
#include <petscksp.h>
#include "laser.h"

using Vec3 = std::array<double, 3>;

namespace tdse 
{
    PetscErrorCode load_starting_state(const simulation& sim, Vec& tdse_state)
{   
    PetscErrorCode ierr;

    int n_basis = sim.bspline_data.value("n_basis",0);
    int n_blocks = sim.angular_data.value("n_blocks",0);
    std::array<int,3> state = sim.tdse_data.value("state",std::array<int,3>{0,0,0});
    int block_idx = sim.lm_to_block.at({state[1],state[2]});
    int total_size = n_basis*n_blocks;

    ierr = VecCreate(PETSC_COMM_WORLD, &tdse_state); CHKERRQ(ierr);
    ierr = VecSetSizes(tdse_state, PETSC_DECIDE, total_size); CHKERRQ(ierr);
    ierr =  VecSetType(tdse_state, VECMPI); CHKERRQ(ierr);
    ierr = VecSetFromOptions(tdse_state); CHKERRQ(ierr);
    ierr = VecSet(tdse_state, 0.0); CHKERRQ(ierr);

    // Create empty vector for TISE output
    Vec tise_state;
    ierr = VecCreate(PETSC_COMM_SELF, &tise_state); CHKERRQ(ierr);
    ierr = VecSetSizes(tise_state, PETSC_DECIDE, n_basis); CHKERRQ(ierr);
    ierr =  VecSetType(tise_state, VECSEQ); CHKERRQ(ierr);
    ierr = VecSetFromOptions(tise_state); CHKERRQ(ierr);
    ierr = VecSet(tise_state, 0.0); CHKERRQ(ierr);


    // Set the name of the vector to match the dataset in the HDF5 group
    std::stringstream ss;
    ss<< "eigenvectors/psi_" << state[0] << "_" << state[1];
    std::string state_path = ss.str();
    ierr = PetscObjectSetName((PetscObject)tise_state, state_path.c_str()); CHKERRQ(ierr);

    // Open the HDF5 file
    PetscViewer viewer;
    ierr = PetscViewerHDF5Open(PETSC_COMM_SELF, "TISE_files/tise_output.h5", FILE_MODE_READ, &viewer); CHKERRQ(ierr);

    // Load the smaller vector from the HDF5 file
    ierr = VecLoad(tise_state, viewer); CHKERRQ(ierr);

    // Close the HDF5 viewer
    ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);


    PetscInt start, end;
    ierr = VecGetOwnershipRange(tdse_state, &start, &end); CHKERRQ(ierr);



    const PetscScalar *tise_state_array;
    ierr = VecGetArrayRead(tise_state, &tise_state_array); CHKERRQ(ierr);

    for (int i = 0; i < n_basis; ++i)
    {   
        int global_index = block_idx*n_basis + i;
        if (global_index >= start && global_index < end)
        {   
            
            ierr = VecSetValue(tdse_state,global_index,tise_state_array[i],INSERT_VALUES); CHKERRQ(ierr);
        }
    }

    ierr = VecRestoreArrayRead(tise_state, &tise_state_array); CHKERRQ(ierr);
    ierr = VecDestroy(&tise_state); CHKERRQ(ierr);
    ierr = VecAssemblyBegin(tdse_state); CHKERRQ(ierr);
    ierr = VecAssemblyEnd(tdse_state); CHKERRQ(ierr);
    ierr = VecDestroy(&tise_state); CHKERRQ(ierr);
    return ierr; 
}

    PetscErrorCode load_matrix(const char* filename, Mat *matrix)
    {
        PetscErrorCode ierr;
        PetscViewer viewer;

        ierr = PetscViewerBinaryOpen(PETSC_COMM_SELF,filename, FILE_MODE_READ,&viewer); CHKERRQ(ierr);

        ierr = MatCreate(PETSC_COMM_SELF,matrix); CHKERRQ(ierr);
        ierr = MatSetType(*matrix,MATSEQAIJ); CHKERRQ(ierr);

        ierr = MatLoad(*matrix,viewer); CHKERRQ(ierr);

        ierr = PetscViewerDestroy(&viewer); CHKERRQ(ierr);
        return ierr;
    }

    PetscErrorCode KroneckerProductParallel(Mat A, PetscInt nnz_A, Mat B, PetscInt nnz_B, Mat *C_out) {
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
        Mat S;
        ierr = load_matrix("TISE_files/S.bin",&S); CHKERRQ(ierr);

        int n_blocks = sim.angular_data.value("n_blocks",0);
        int degree = sim.bspline_data.value("degree",0);
        
        Mat I; 
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

        ierr = KroneckerProductParallel(I,1,S,2*degree+1,&S_atomic); CHKERRQ(ierr);

        ierr = MatDestroy(&S); CHKERRQ(ierr);
        ierr = MatDestroy(&I); CHKERRQ(ierr);
        return ierr;
    }

    PetscErrorCode _construct_H_atomic(const simulation& sim, Mat& H_atomic)
    {
        PetscErrorCode ierr; 

        int n_blocks = sim.angular_data.value("n_blocks",0);
        int n_basis = sim.bspline_data.value("n_basis",0);
        int degree = sim.bspline_data.value("degree",0);
        int lmax = sim.angular_data.value("lmax",0);

        Mat K,Inv_r2,Inv_r;
        ierr = load_matrix("TISE_files/K.bin",&K); CHKERRQ(ierr);
        ierr = load_matrix("TISE_files/Inv_r2.bin",&Inv_r2); CHKERRQ(ierr);
        ierr = load_matrix("TISE_files/Inv_r.bin",&Inv_r); CHKERRQ(ierr);

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
                auto lm_pair = sim.block_to_lm.at(i);  // Retrieve (l, m) pair
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

            ierr = KroneckerProductParallel(I_partial,1,temp,2*degree+1,&H_partial); CHKERRQ(ierr);
            ierr = MatDestroy(&temp); CHKERRQ(ierr);
            ierr = MatDestroy(&I_partial); CHKERRQ(ierr);

            ierr = MatAXPY(H_atomic,1.0,H_partial,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
            ierr = MatDestroy(&H_partial); CHKERRQ(ierr);
        }

        return ierr;






    }

    PetscErrorCode _construct_atomic_interaction(const simulation& sim,Mat& H_atomic,Mat& S_atomic, Mat& atomic_left,Mat& atomic_right)
    {   
        double dt = sim.grid_data.value("time_spacing",0.0);
        std::complex<double> alpha  = PETSC_i * (dt / 2.0);
        PetscErrorCode ierr;
        ierr = MatDuplicate(S_atomic,MAT_COPY_VALUES,&atomic_left); CHKERRQ(ierr);
        ierr = MatDuplicate(S_atomic,MAT_COPY_VALUES,&atomic_right); CHKERRQ(ierr);
        ierr = MatAXPY(atomic_left,alpha,H_atomic,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
        ierr = MatAXPY(atomic_right,-alpha,H_atomic,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
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

        int n_blocks = sim.angular_data.at("n_blocks").get<int>();
        int degree = sim.bspline_data.at("degree").get<int>();

        Mat H_lm_1;
        ierr = MatCreate(PETSC_COMM_SELF, &H_lm_1); CHKERRQ(ierr);
        ierr = MatSetSizes(H_lm_1, PETSC_DECIDE, PETSC_DECIDE, n_blocks, n_blocks); CHKERRQ(ierr);
        ierr = MatSetFromOptions(H_lm_1); CHKERRQ(ierr);
        ierr = MatSeqAIJSetPreallocation(H_lm_1, 2, NULL); CHKERRQ(ierr);
        ierr = MatSetUp(H_lm_1); CHKERRQ(ierr);
        for (int i = 0; i < n_blocks; ++i)
        {
            std::pair<int,int> lm_pair = sim.block_to_lm.at(i);
            int l = lm_pair.first;
            int m = lm_pair.second;
            for (int j = 0; j < n_blocks; ++j)
            {
                std::pair<int,int> lm_pair_prime = sim.block_to_lm.at(j);
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
            std::pair<int,int> lm_pair = sim.block_to_lm.at(i);
            int l = lm_pair.first;
            int m = lm_pair.second;
            for (int j = 0; j < n_blocks; ++j)
            {
                std::pair<int,int> lm_pair_prime = sim.block_to_lm.at(j);
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
        ierr = load_matrix("TISE_files/Der.bin",&Der); CHKERRQ(ierr);
        ierr = load_matrix("TISE_files/Inv_r.bin",&Inv_r); CHKERRQ(ierr);

        Mat H_z_2;
        ierr = KroneckerProductParallel(H_lm_1,2,Der,2*degree+1,&H_z); CHKERRQ(ierr);
        ierr = KroneckerProductParallel(H_lm_2,2,Inv_r,2*degree+1,&H_z_2); CHKERRQ(ierr);

        ierr = MatAXPY(H_z,1.0,H_z_2,SAME_NONZERO_PATTERN); CHKERRQ(ierr);

        ierr = MatDestroy(&H_lm_1); CHKERRQ(ierr);
        ierr = MatDestroy(&H_lm_2); CHKERRQ(ierr);
        ierr = MatDestroy(&Der); CHKERRQ(ierr);
        ierr = MatDestroy(&Inv_r); CHKERRQ(ierr);
        ierr = MatDestroy(&H_z_2); CHKERRQ(ierr);
        return ierr;
    }

    // PetscErrorCode solve_tdse(const simulation& sim, int rank)
// {
//     PetscErrorCode ierr;
//     Vec state;
//     ierr = load_starting_state(sim, state); CHKERRQ(ierr);

//     Vec3 components = sim.laser_data.at("components").get<Vec3>();
//     std::complex<double> alpha = PETSC_i * (sim.grid_data.at("time_spacing").get<double>() / 2.0);

//     Mat H_atomic, S_atomic, atomic_left, atomic_right;
//     ierr = _construct_S_atomic(sim, S_atomic); CHKERRQ(ierr);
//     ierr = _construct_H_atomic(sim, H_atomic); CHKERRQ(ierr);
//     ierr = _construct_atomic_interaction(sim, H_atomic, S_atomic, atomic_left, atomic_right); CHKERRQ(ierr);

//     Mat H_z;
//     if (false)
//     {   
//         ierr = _construct_z_interaction(sim, H_z); CHKERRQ(ierr);
//     }

//     std::complex<double> norm;
//     Vec y; 
//     ierr = VecDuplicate(state, &y); CHKERRQ(ierr);
//     ierr = MatMult(S_atomic, state, y); CHKERRQ(ierr);
//     ierr = VecDot(state, y, &norm); CHKERRQ(ierr);
//     PetscPrintf(PETSC_COMM_WORLD, "Norm of Initial State: (%.4f,%.4f)\n", norm.real(), norm.imag());

//     int Nt = sim.grid_data.value("Nt", 0);
//     double dt = sim.grid_data.value("time_spacing", 0.0);

//     KSP ksp;
//     ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
//     ierr = KSPSetTolerances(ksp, sim.tdse_data.at("tolerance").get<double>(), PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
//     ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);


//     Vec state_temp;
//     Mat atomic_left_temp, atomic_right_temp;
//     ierr = MatDuplicate(atomic_left, MAT_COPY_VALUES, &atomic_left_temp); CHKERRQ(ierr);
//     ierr = MatDuplicate(atomic_right, MAT_COPY_VALUES, &atomic_right_temp); CHKERRQ(ierr);
//     ierr = VecDuplicate(state, &state_temp); CHKERRQ(ierr);


    
//     double start = MPI_Wtime();
//     for (int idx = 0; idx < Nt; ++idx)
//     {   
//         if (true)
//         {
//             PetscPrintf(PETSC_COMM_WORLD, "Time Step: %d\n", idx);
//         }

//         double t= idx*dt;
        

//         // Faster copy, assumes nonzero structure does not change
//         ierr = MatCopy(atomic_left, atomic_left_temp, SAME_NONZERO_PATTERN); CHKERRQ(ierr);
//         ierr = MatCopy(atomic_right, atomic_right_temp, SAME_NONZERO_PATTERN); CHKERRQ(ierr);

//         if (false)
//         {   
//             double laser_val = laser::A(t,sim,2);
//             ierr = MatAXPY(atomic_left_temp,alpha*laser_val,H_z,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
//             ierr = MatAXPY(atomic_right_temp,-alpha*laser_val,H_z,DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
//         }

//         ierr = KSPSetOperators(ksp, atomic_left_temp, atomic_left_temp); CHKERRQ(ierr); 

//         ierr = MatMult(atomic_right_temp, state, state_temp); CHKERRQ(ierr);

//         // Reuse KSP operator
//         ierr = KSPSetOperators(ksp, atomic_left_temp, atomic_left_temp); CHKERRQ(ierr);
//         ierr = KSPSetReusePreconditioner(ksp, PETSC_TRUE); CHKERRQ(ierr);

//         ierr = KSPSolve(ksp, state_temp, state); CHKERRQ(ierr);
//     }
//     double end = MPI_Wtime();
//     PetscPrintf(PETSC_COMM_WORLD, "Time to solve TDSE %.3f\n", end - start);

//     ierr = MatMult(S_atomic, state, y); CHKERRQ(ierr);
//     ierr = VecDot(state, y, &norm); CHKERRQ(ierr);
//     PetscPrintf(PETSC_COMM_WORLD, "Norm of Final State: (%.15f,%.15f)\n", (double)norm.real(), (double)norm.imag());

//     return ierr;
// }


PetscErrorCode solve_tdse(const simulation& sim, int rank)
{
    PetscErrorCode ierr;
    Vec state;
    ierr = load_starting_state(sim, state); CHKERRQ(ierr);

    Vec3 components = sim.laser_data.at("components").get<Vec3>();
    std::complex<double> alpha = PETSC_i * (sim.grid_data.at("time_spacing").get<double>() / 2.0);

    Mat H_atomic, S_atomic, atomic_left, atomic_right;
    ierr = _construct_S_atomic(sim, S_atomic); CHKERRQ(ierr);
    ierr = _construct_H_atomic(sim, H_atomic); CHKERRQ(ierr);
    ierr = _construct_atomic_interaction(sim, H_atomic, S_atomic, atomic_left, atomic_right); CHKERRQ(ierr);

    Mat H_z;
    if (false) {   
        ierr = _construct_z_interaction(sim, H_z); CHKERRQ(ierr);
    }

    std::complex<double> norm;
    Vec y; 
    ierr = VecDuplicate(state, &y); CHKERRQ(ierr);
    ierr = MatMult(S_atomic, state, y); CHKERRQ(ierr);
    ierr = VecDot(state, y, &norm); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Norm of Initial State: (%.4f,%.4f)\n", norm.real(), norm.imag());

    int Nt = sim.grid_data.value("Nt", 0);
    double dt = sim.grid_data.value("time_spacing", 0.0);

    KSP ksp;
    ierr = KSPCreate(PETSC_COMM_WORLD, &ksp); CHKERRQ(ierr);
    PC pc;
    ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
    ierr = PCSetType(pc, PCBJACOBI); CHKERRQ(ierr);
    ierr = KSPSetTolerances(ksp, sim.tdse_data.at("tolerance").get<double>(), PETSC_DEFAULT, PETSC_DEFAULT, PETSC_DEFAULT); CHKERRQ(ierr);
    ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

    Vec state_temp;
    Mat atomic_left_temp, atomic_right_temp;
    ierr = MatDuplicate(atomic_left, MAT_COPY_VALUES, &atomic_left_temp); CHKERRQ(ierr);
    ierr = MatDuplicate(atomic_right, MAT_COPY_VALUES, &atomic_right_temp); CHKERRQ(ierr);
    ierr = VecDuplicate(state, &state_temp); CHKERRQ(ierr);

    double start = MPI_Wtime();
    for (int idx = 0; idx < Nt; ++idx) 
    {   
        if (true)
        {
            PetscPrintf(PETSC_COMM_WORLD, "Time Step: %d/%d\n", idx,Nt);
        }
        
        double t = idx * dt;

        // Destroy and recreate temp matrices to avoid accumulation of structural changes
        ierr = MatDestroy(&atomic_left_temp); CHKERRQ(ierr);
        ierr = MatDestroy(&atomic_right_temp); CHKERRQ(ierr);
        ierr = MatDuplicate(atomic_left, MAT_COPY_VALUES, &atomic_left_temp); CHKERRQ(ierr);
        ierr = MatDuplicate(atomic_right, MAT_COPY_VALUES, &atomic_right_temp); CHKERRQ(ierr);

        if (false) 
        {

            double laser_val = laser::A(t, sim, 2);
            ierr = MatAXPY(atomic_left_temp, alpha * laser_val, H_z, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
            ierr = MatAXPY(atomic_right_temp, -alpha * laser_val, H_z, DIFFERENT_NONZERO_PATTERN); CHKERRQ(ierr);
        }

        ierr = KSPSetOperators(ksp, atomic_left_temp, atomic_left_temp); CHKERRQ(ierr); 

        ierr = MatMult(atomic_right_temp, state, state_temp); CHKERRQ(ierr);

        // Reuse KSP operator
        ierr = KSPSetOperators(ksp, atomic_left_temp, atomic_left_temp); CHKERRQ(ierr);
        ierr = KSPSetReusePreconditioner(ksp, PETSC_TRUE); CHKERRQ(ierr);

        ierr = KSPSolve(ksp, state_temp, state); CHKERRQ(ierr);

        
    }

    double end = MPI_Wtime();
    PetscPrintf(PETSC_COMM_WORLD, "Time to solve TDSE %.3f\n", end - start);

    ierr = MatMult(S_atomic, state, y); CHKERRQ(ierr);
    ierr = VecDot(state, y, &norm); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, "Norm of Final State: (%.15f,%.15f)\n", (double)norm.real(), (double)norm.imag());

    // Clean up
    ierr = VecDestroy(&y); CHKERRQ(ierr);
    ierr = VecDestroy(&state_temp); CHKERRQ(ierr);
    ierr = MatDestroy(&atomic_left_temp); CHKERRQ(ierr);
    ierr = MatDestroy(&atomic_right_temp); CHKERRQ(ierr);
    ierr = MatDestroy(&H_z); CHKERRQ(ierr);
    ierr = MatDestroy(&H_atomic); CHKERRQ(ierr);
    ierr = MatDestroy(&S_atomic); CHKERRQ(ierr);
    ierr = MatDestroy(&atomic_left); CHKERRQ(ierr);
    ierr = MatDestroy(&atomic_right); CHKERRQ(ierr);
    ierr = VecDestroy(&state); CHKERRQ(ierr);
    ierr = KSPDestroy(&ksp); CHKERRQ(ierr);

    return ierr;
}



}
