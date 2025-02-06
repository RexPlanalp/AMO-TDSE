#include "tise.h"

#include <petscmat.h>
#include <petscviewerhdf5.h>
#include <slepceps.h>
#include "simulation.h"
#include "bsplines.h"

#include <sys/stat.h>
#include <sys/types.h>
#include <iostream>


namespace tise
{
    PetscErrorCode solve_tise(const simulation& sim,int rank)
    {   
        
        double start_time = MPI_Wtime();

        PetscPrintf(PETSC_COMM_WORLD, "Declaring Petsc Objects  \n\n");
        PetscErrorCode ierr;
        PetscViewer viewTISE;
        Mat K;
        Mat Inv_r2;
        Mat Inv_r;
        Mat S;
        Mat Der;
        Mat temp;
        EPS eps;
        int nconv;

        if (rank == 0) 
        {
            if (mkdir("TISE_files", 0777) == 0) 
            {
                PetscPrintf(PETSC_COMM_WORLD, "Directory created: %s\n\n", "TISE_files");
            } 
            else 
            {
                PetscPrintf(PETSC_COMM_WORLD, "Directory already exists: %s\n\n", "TISE_files");
            }
        }


        PetscPrintf(PETSC_COMM_WORLD, "Constructing Matrices  \n\n");
        ierr = bsplines::construct_overlap(sim,S,true,false); CHKERRQ(ierr);
        ierr = bsplines::construct_kinetic(sim,K,true,false); CHKERRQ(ierr);
        ierr = bsplines::construct_invr2(sim,Inv_r2,true,false); CHKERRQ(ierr);
        ierr = bsplines::construct_invr(sim,Inv_r,true,false); CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD, "Opening HDF5 File  \n\n");
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"TISE_files/tise_output.h5", FILE_MODE_WRITE, &viewTISE); CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD, "Setting Up Eigenvalue Problem  \n\n");
        ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
        ierr = EPSSetProblemType(eps, EPS_GNHEP); CHKERRQ(ierr);
        ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
        ierr = EPSSetType(eps,EPSKRYLOVSCHUR); CHKERRQ(ierr);
        ierr = EPSSetTolerances(eps,sim.tise_data.value("tolerance",1E-15),sim.tise_data.value("max_iter",3000)); CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD, "Solving TISE  \n\n");
        for (int l = 0; l<= sim.angular_data.at("lmax").get<int>(); ++l)
        {
            ierr = MatDuplicate(K,MAT_COPY_VALUES,&temp); CHKERRQ(ierr);
            ierr = MatAXPY(temp, l*(l+1)*0.5,Inv_r2,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
            ierr = MatAXPY(temp,-1.0,Inv_r,SAME_NONZERO_PATTERN); CHKERRQ(ierr);

            int num_of_energies = sim.angular_data.at("nmax").get<int>() - l;
            if (num_of_energies <= 0)
            {
                continue;
            }

            ierr = EPSSetOperators(eps,temp,S); CHKERRQ(ierr);
            ierr = EPSSetDimensions(eps,num_of_energies,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
            ierr = EPSSolve(eps); CHKERRQ(ierr);
            ierr = EPSGetConverged(eps,&nconv); CHKERRQ(ierr);
            PetscPrintf(PETSC_COMM_WORLD, "Eigenvalues Requested %d, Eigenvalues Converged: %d \n\n", num_of_energies,nconv); CHKERRQ(ierr);

            for (int i = 0; i < nconv; ++i)
            {
                std::complex<double> eigenvalue;
                ierr = EPSGetEigenvalue(eps,i,&eigenvalue,NULL); CHKERRQ(ierr);
                

             

                if (eigenvalue.real()>0)
                {
                    continue;
                }


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

                Vec eigenvector;
                ierr = MatCreateVecs(temp,&eigenvector, NULL); CHKERRQ(ierr);
                ierr = EPSGetEigenvector(eps,i,eigenvector,NULL); CHKERRQ(ierr);

                std::complex<double> norm;
                Vec y;
                ierr = VecDuplicate(eigenvector,&y); CHKERRQ(ierr);
                ierr = MatMult(S,eigenvector,y); CHKERRQ(ierr);
                ierr = VecDot(eigenvector,y,&norm); CHKERRQ(ierr);
                ierr = VecScale(eigenvector,1.0/std::sqrt(norm.real())); CHKERRQ(ierr);

                ierr = MatMult(S,eigenvector,y); CHKERRQ(ierr);
                ierr = VecDot(eigenvector,y,&norm); CHKERRQ(ierr);

                if (sim.debug)
                {
                    PetscPrintf(PETSC_COMM_WORLD,"Eigenvector %d -> Norm(%.4f , %.4f) -> Eigenvalue(%.4f , %.4f)  \n",i+1,norm.real(),norm.imag(),eigenvalue.real(),eigenvalue.imag()); CHKERRQ(ierr);
                }
                

                std::string eigenvector_name = std::string("psi_") + std::to_string(i+l+1) + "_" + std::to_string(l);
                ierr = PetscViewerHDF5PushGroup(viewTISE, "/eigenvectors"); CHKERRQ(ierr);
                ierr = PetscObjectSetName((PetscObject)eigenvector,eigenvector_name.c_str()); CHKERRQ(ierr);
                ierr = VecView(eigenvector,viewTISE); CHKERRQ(ierr);
                ierr = PetscViewerHDF5PopGroup(viewTISE); CHKERRQ(ierr);

                ierr = VecDestroy(&eigenvector); CHKERRQ(ierr);
                ierr = VecDestroy(&y); CHKERRQ(ierr);

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
        ierr = MatDestroy(&temp); CHKERRQ(ierr);

        double end_time = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"Time to solve TISE %.3f\n\n",end_time-start_time);
    }

    PetscErrorCode prepare_matrices(const simulation& sim,int rank)
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
        ierr = bsplines::construct_overlap(sim,S,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_kinetic(sim,K,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_invr2(sim,Inv_r2,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_invr(sim,Inv_r,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_der(sim,Der,true,true); CHKERRQ(ierr);

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
        
    }
}


