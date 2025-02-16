#include <sys/stat.h>
#include <iostream>


#include <petscmat.h>
#include <petscviewerhdf5.h>
#include <slepceps.h>


#include "tise.h"
#include "simulation.h"
#include "bsplines.h"
#include "misc.h"



namespace tise
{

    PetscErrorCode setup_eigenvalue_problem(const simulation& sim, EPS& eps)
    {   
        PetscErrorCode ierr;
        ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
        ierr = EPSSetProblemType(eps, EPS_GNHEP); CHKERRQ(ierr);
        ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
        ierr = EPSSetType(eps,EPSKRYLOVSCHUR); CHKERRQ(ierr);
        ierr = EPSSetTolerances(eps,sim.schrodinger_params.tise_tol,sim.schrodinger_params.tise_max_iter); CHKERRQ(ierr);
        return ierr;
    }

    PetscErrorCode solve_eigenvalue_problem(const Mat& H, const Mat& S, EPS& eps, int num_of_energies, int& nconv)
    {
        PetscErrorCode ierr;
        ierr = EPSSetOperators(eps,H,S); CHKERRQ(ierr);
        ierr = EPSSetDimensions(eps,num_of_energies,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
        ierr = EPSSolve(eps); CHKERRQ(ierr);
        ierr = EPSGetConverged(eps,&nconv); CHKERRQ(ierr);
        return ierr;
    }

    PetscErrorCode compute_eigenvector_norm(const Vec& eigenvector, const Mat& S, std::complex<double>& norm)
    {
        PetscErrorCode ierr;
        Vec temp_vec;
        ierr = VecDuplicate(eigenvector,&temp_vec); CHKERRQ(ierr);
        ierr = MatMult(S,eigenvector,temp_vec); CHKERRQ(ierr);
        ierr = VecDot(eigenvector,temp_vec,&norm); CHKERRQ(ierr);
        norm = std::sqrt(norm);

        ierr = VecDestroy(&temp_vec); CHKERRQ(ierr);
        return ierr;
    }
    
    PetscErrorCode extract_normalized_eigenvector(Vec& eigenvector,const EPS& eps, const Mat& S, int i)
    {   
        PetscErrorCode ierr;
        ierr = MatCreateVecs(S,&eigenvector, NULL); CHKERRQ(ierr);
        ierr = EPSGetEigenvector(eps,i,eigenvector,NULL); CHKERRQ(ierr);

        std::complex<double> norm;
        ierr = compute_eigenvector_norm(eigenvector,S,norm); CHKERRQ(ierr);
        ierr = VecScale(eigenvector,1.0/norm.real()); CHKERRQ(ierr);
        return ierr;
    }

    PetscErrorCode save_eigenvalue(const std::complex<double>& eigenvalue, PetscViewer& viewTISE, int l, int i)
    {   
        PetscErrorCode ierr;
        std::string eigenvalue_name = std::string("E_") + std::to_string(i+l+1) + '_' + std::to_string(l);
        ierr = PetscViewerHDF5PushGroup(viewTISE,"/eigenvalues"); CHKERRQ(ierr);

        Vec eigenvalue_vec;
        ierr = VecCreate(PETSC_COMM_WORLD, &eigenvalue_vec); CHKERRQ(ierr);
        ierr = VecSetSizes(eigenvalue_vec,PETSC_DECIDE,1); CHKERRQ(ierr);
        ierr = VecSetFromOptions(eigenvalue_vec); CHKERRQ(ierr);
        ierr=  VecSetValue(eigenvalue_vec,0,eigenvalue,INSERT_VALUES); CHKERRQ(ierr);
        ierr = VecAssemblyBegin(eigenvalue_vec); CHKERRQ(ierr);
        ierr = VecAssemblyEnd(eigenvalue_vec); CHKERRQ(ierr);
    
        ierr = PetscObjectSetName((PetscObject)eigenvalue_vec,eigenvalue_name.c_str()); CHKERRQ(ierr);
        ierr = VecView(eigenvalue_vec, viewTISE); CHKERRQ(ierr);
        ierr = PetscViewerHDF5PopGroup(viewTISE); CHKERRQ(ierr);
        ierr = VecDestroy(&eigenvalue_vec); CHKERRQ(ierr);

        return ierr;
    }

    PetscErrorCode save_eigenvector(Vec& eigenvector, PetscViewer& viewTISE, int l, int i)
    {
        PetscErrorCode ierr;
        std::string eigenvector_name = std::string("psi_") + std::to_string(i+l+1) + "_" + std::to_string(l);
        ierr = PetscViewerHDF5PushGroup(viewTISE, "/eigenvectors"); CHKERRQ(ierr);
        ierr = PetscObjectSetName((PetscObject)eigenvector,eigenvector_name.c_str()); CHKERRQ(ierr);
        ierr = VecView(eigenvector,viewTISE); CHKERRQ(ierr);
        ierr = PetscViewerHDF5PopGroup(viewTISE); CHKERRQ(ierr);
        return ierr;
    }

    PetscErrorCode solve_eigensystem(const simulation& sim)
    {   
        PetscErrorCode ierr;
        PetscPrintf(PETSC_COMM_WORLD, "Constructing Matrices  \n\n");
        Mat S;
        ierr = bsplines::construct_matrix(sim,S,bsplines::overlap_integrand,true,false); CHKERRQ(ierr);

        Mat K;
        ierr = bsplines::construct_matrix(sim,K,bsplines::kinetic_integrand,true,false); CHKERRQ(ierr);

        Mat Inv_r2;
        ierr = bsplines::construct_matrix(sim,Inv_r2,bsplines::invr2_integrand,true,false); CHKERRQ(ierr);

        Mat Inv_r;
        ierr = bsplines::construct_matrix(sim,Inv_r,bsplines::invr_integrand,true,false); CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD, "Opening HDF5 File  \n\n");
        PetscViewer viewTISE;
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,(sim.tise_output_path+"/tise_output.h5").c_str(), FILE_MODE_WRITE, &viewTISE); CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD, "Setting Up Eigenvalue Problem  \n\n");
        EPS eps;
        ierr = setup_eigenvalue_problem(sim,eps); CHKERRQ(ierr);


        PetscPrintf(PETSC_COMM_WORLD, "Solving TISE  \n\n");
        int nconv;
        Mat temp;
        for (int l = 0; l<= sim.angular_params.lmax; ++l)
        {
            ierr = MatDuplicate(K,MAT_COPY_VALUES,&temp); CHKERRQ(ierr);
            ierr = MatAXPY(temp, l*(l+1)*0.5,Inv_r2,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
            ierr = MatAXPY(temp,-1.0,Inv_r,SAME_NONZERO_PATTERN); CHKERRQ(ierr);

            int num_of_energies = sim.angular_params.nmax - l;
            if (num_of_energies <= 0)
            {
                continue;
            }

            ierr = solve_eigenvalue_problem(temp,S,eps,num_of_energies,nconv); CHKERRQ(ierr);
            PetscPrintf(PETSC_COMM_WORLD, "Eigenvalues Requested %d, Eigenvalues Converged: %d \n\n", num_of_energies,nconv); CHKERRQ(ierr);

            for (int i = 0; i < nconv; ++i)
            {
                std::complex<double> eigenvalue;
                ierr = EPSGetEigenvalue(eps,i,&eigenvalue,NULL); CHKERRQ(ierr);
                

             

                if (eigenvalue.real()>0)
                {
                    continue;
                }

                ierr = save_eigenvalue(eigenvalue,viewTISE,l,i); CHKERRQ(ierr);

                Vec eigenvector;
                ierr = extract_normalized_eigenvector(eigenvector,eps,S,i); CHKERRQ(ierr);
                
                
                
                if (sim.debug)
                {   
                    std::complex<double> norm;
                    ierr = compute_eigenvector_norm(eigenvector,S,norm); CHKERRQ(ierr);
                    PetscPrintf(PETSC_COMM_WORLD,"Eigenvector %d -> Norm(%.4f , %.4f) -> Eigenvalue(%.4f , %.4f)  \n",i+1,norm.real(),norm.imag(),eigenvalue.real(),eigenvalue.imag()); CHKERRQ(ierr);
                }
                ierr = save_eigenvector(eigenvector,viewTISE,l,i); CHKERRQ(ierr);


                ierr = VecDestroy(&eigenvector); CHKERRQ(ierr);
            }
            ierr = MatDestroy(&temp); CHKERRQ(ierr);

        }
        
        PetscPrintf(PETSC_COMM_WORLD, "Destroying Petsc Objects  \n\n");
        ierr = PetscViewerDestroy(&viewTISE); CHKERRQ(ierr);
        ierr = EPSDestroy(&eps); CHKERRQ(ierr);
        ierr = MatDestroy(&K); CHKERRQ(ierr);
        ierr = MatDestroy(&Inv_r2); CHKERRQ(ierr);
        ierr = MatDestroy(&Inv_r); CHKERRQ(ierr);
        ierr = MatDestroy(&S); CHKERRQ(ierr);
        return ierr;
    }

    PetscErrorCode prepare_matrices(const simulation& sim)
    {   
        double time_start = MPI_Wtime();

        PetscPrintf(PETSC_COMM_WORLD, "Declaring Petsc Objects  \n\n");
        PetscErrorCode ierr;
        Mat K;
        Mat Inv_r2;
        Mat Inv_r;
        Mat S;
        Mat Der;
        
        PetscPrintf(PETSC_COMM_WORLD, "Constructing Matrices  \n\n");
        ierr = bsplines::construct_matrix(sim,S,bsplines::overlap_integrand,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_matrix(sim,K,bsplines::kinetic_integrand,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_matrix(sim,Inv_r2,bsplines::invr2_integrand,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_matrix(sim,Inv_r,bsplines::invr_integrand,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_matrix(sim,Der,bsplines::der_integrand,true,true); CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD, "Saving Matrices  \n\n");
        ierr = bsplines::save_matrix(K,"TISE_files/K.bin"); CHKERRQ(ierr);
        ierr = bsplines::save_matrix(Inv_r2,"TISE_files/Inv_r2.bin"); CHKERRQ(ierr);
        ierr = bsplines::save_matrix(Inv_r,"TISE_files/Inv_r.bin"); CHKERRQ(ierr);
        ierr = bsplines::save_matrix(S,"TISE_files/S.bin"); CHKERRQ(ierr);
        ierr = bsplines::save_matrix(Der,"TISE_files/Der.bin"); CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD, "Destroying Petsc Objects  \n\n");
        ierr = MatDestroy(&K); CHKERRQ(ierr);
        ierr = MatDestroy(&Inv_r2); CHKERRQ(ierr);
        ierr = MatDestroy(&Inv_r); CHKERRQ(ierr);
        ierr = MatDestroy(&S); CHKERRQ(ierr);
        ierr = MatDestroy(&Der); CHKERRQ(ierr);

        double time_end = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"Time to prepare matrices %.3f\n",time_end-time_start);
        return ierr;
        
    }

    PetscErrorCode solve_tise(const simulation& sim,int rank)
    {    
        double start_time = MPI_Wtime();
        PetscErrorCode ierr;

        create_directory(rank, "TISE_files");
        ierr = solve_eigensystem(sim); CHKERRQ(ierr);
        ierr = prepare_matrices(sim); CHKERRQ(ierr);



        double end_time = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"Time to solve TISE %.3f\n\n",end_time-start_time);
        return ierr;
    }

    
}


