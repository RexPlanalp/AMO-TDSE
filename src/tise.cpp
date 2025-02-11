#include <sys/stat.h>
#include <iostream>


#include <petscmat.h>
#include <petscviewerhdf5.h>
#include <slepceps.h>


#include "tise.h"
#include "simulation.h"
#include "bsplines.h"



namespace tise
{
    struct tise_filepaths
    {
        static constexpr const char* tise_output = "TISE_files/tise_output.h5";
        static constexpr const char* K = "TISE_files/K.bin";
        static constexpr const char* Inv_r2 = "TISE_files/Inv_r2.bin";
        static constexpr const char* Inv_r = "TISE_files/Inv_r.bin";
        static constexpr const char* S = "TISE_files/S.bin";
        static constexpr const char* Der = "TISE_files/Der.bin";
    };

    struct tise_context 
    {
        int lmax;
        int nmax;
        int tise_mat_iter;
        double tise_tolerance;

        static tise_context set_config(const simulation& sim)
        {
            try
            {
                tise_context config;
                config.lmax = sim.angular_data.at("lmax").get<int>();
                config.nmax = sim.angular_data.at("nmax").get<int>();
                config.tise_tolerance = sim.tise_data.at("tolereance").get<double>();
                config.tise_mat_iter = sim.tise_data.at("max_iter").get<int>();
            }
            catch(const std::exception& e)
            {
                std::cerr << "Error in setting up Time Independent Schrodinger Equation Context: " << "\n\n " <<  e.what() << '\n\n';
            }
            
        }
    };



    PetscErrorCode solve_tise(const simulation& sim,int rank)
    {   
        
        double start_time = MPI_Wtime();

        tise_filepaths filepaths = tise_filepaths();
        tise_context config = tise_context::set_config(sim);
        PetscErrorCode ierr;

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
        Mat S;
        ierr = bsplines::construct_overlap(sim,S,true,false); CHKERRQ(ierr);

        Mat K;
        ierr = bsplines::construct_kinetic(sim,K,true,false); CHKERRQ(ierr);

        Mat Inv_r2;
        ierr = bsplines::construct_invr2(sim,Inv_r2,true,false); CHKERRQ(ierr);

        Mat Inv_r;
        ierr = bsplines::construct_invr(sim,Inv_r,true,false); CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD, "Opening HDF5 File  \n\n");
        PetscViewer viewTISE;
        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,filepaths.tise_output, FILE_MODE_WRITE, &viewTISE); CHKERRQ(ierr);

        PetscPrintf(PETSC_COMM_WORLD, "Setting Up Eigenvalue Problem  \n\n");
        EPS eps;
        ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
        ierr = EPSSetProblemType(eps, EPS_GNHEP); CHKERRQ(ierr);
        ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
        ierr = EPSSetType(eps,EPSKRYLOVSCHUR); CHKERRQ(ierr);
        ierr = EPSSetTolerances(eps,config.tise_tolerance,config.tise_mat_iter); CHKERRQ(ierr);


        PetscPrintf(PETSC_COMM_WORLD, "Solving TISE  \n\n");
        int nconv;
        Mat temp;
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


