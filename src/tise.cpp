#include <sys/stat.h>
#include <iostream>


#include <petscmat.h>
#include <petscviewerhdf5.h>
#include <slepceps.h>


#include "tise.h"
#include "simulation.h"
#include "bsplines.h"
#include "misc.h"

#include "petsc_wrappers/PetscMatrix.h"
#include "petsc_wrappers/PetscFileViewer.h"
#include "petsc_wrappers/PetscVector.h"
#include "petsc_wrappers/PetscEPS.h"
#include "utility.h"


namespace tise
{
    void solve_eigensystem(const simulation& sim)
    {   

    
        PetscErrorCode ierr;
        PetscPrintf(PETSC_COMM_WORLD, "Constructing Matrices  \n\n");
        
        RadialMatrix K(sim,MatrixType::PARALLEL);
        RadialMatrix Inv_r2(sim,MatrixType::PARALLEL);
        RadialMatrix Potential(sim,MatrixType::PARALLEL);
        RadialMatrix S(sim,MatrixType::PARALLEL);

        K.bindElement(bsplines::kinetic_integrand);
        Inv_r2.bindElement(bsplines::invr2_integrand);
        Potential.bindElement(bsplines::H_integrand);
        S.bindElement(bsplines::overlap_integrand);

        K.populateMatrix(sim,ECSMode::OFF);
        Inv_r2.populateMatrix(sim,ECSMode::OFF);
        Potential.populateMatrix(sim,ECSMode::OFF);
        S.populateMatrix(sim,ECSMode::OFF);

        K.assemble();
        Inv_r2.assemble();
        Potential.assemble();
        S.assemble();

    
        PetscPrintf(PETSC_COMM_WORLD, "Opening HDF5 File  \n\n");
        
        PetscHDF5Viewer viewTISE((sim.tise_output_path+"/tise_output.h5").c_str());
        

        PetscPrintf(PETSC_COMM_WORLD, "Setting Up Eigenvalue Problem  \n\n");
        PetscEPS eps;
        eps.setConvergenceParams(sim);
        
        PetscPrintf(PETSC_COMM_WORLD, "Solving TISE  \n\n");

        
       
       
        
        for (int l = 0; l<= sim.angular_params.lmax; ++l)
        {   
            
            PetscMatrix temp(K);
            ierr = MatAXPY(temp.matrix,l*(l+1)*0.5,Inv_r2.matrix,SAME_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");
            ierr = MatAXPY(temp.matrix,1.0,Potential.matrix,SAME_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");
            
            

            
            int requested_pairs = sim.angular_params.nmax - l;
            if (requested_pairs <= 0)
            {
                continue;
            }
            

            
            eps.setSolverParams(requested_pairs);
            eps.setOperators(temp,S);
            int converged_pairs = eps.solve();
            
            PetscPrintf(PETSC_COMM_WORLD, "Eigenvalues Requested %d, Eigenvalues Converged: %d \n\n", converged_pairs,requested_pairs); checkErr(ierr,"Error in PetscPrintf");

            
            for (int pair_idx = 0; pair_idx < converged_pairs; ++pair_idx)
            {
                std::complex<double> eigenvalue = eps.getEigenvalue(pair_idx);
                
                if (eigenvalue.real()>0)
                {
                    continue;
                }

                std::string eigenvector_name = std::string("psi_") + std::to_string(pair_idx+l+1) + "_" + std::to_string(l);
                std::string eigenvalue_name = std::string("E_") + std::to_string(pair_idx+l+1) + '_' + std::to_string(l);

                viewTISE.saveValue(eigenvalue, "eigenvalues", eigenvalue_name.c_str());

                Wavefunction eigenvector = eps.getEigenvector(pair_idx,S);
                eigenvector.normalize(S);

                viewTISE.saveVector(eigenvector, "eigenvectors", eigenvector_name.c_str());
                
                
                if (sim.debug)
                {   
                    std::complex<double> norm = eigenvector.computeNorm(S);
                    
                    PetscPrintf(PETSC_COMM_WORLD,"Eigenvector %d -> Norm(%.4f , %.4f) -> Eigenvalue(%.4f , %.4f)  \n",pair_idx+1,norm.real(),norm.imag(),eigenvalue.real(),eigenvalue.imag()); checkErr(ierr,"Error in PetscPrintf");
                }
            }
        }
    }

    void prepare_matrices(const simulation& sim)
    {   
        double time_start = MPI_Wtime();

        PetscPrintf(PETSC_COMM_WORLD, "Constructing Matrices  \n\n");

        
        RadialMatrix K(sim,MatrixType::PARALLEL);
        RadialMatrix Inv_r2(sim,MatrixType::PARALLEL);
        RadialMatrix Inv_r(sim,MatrixType::PARALLEL);
        RadialMatrix S(sim,MatrixType::PARALLEL);
        RadialMatrix Der(sim,MatrixType::PARALLEL);

        K.bindElement(bsplines::kinetic_integrand);
        Inv_r2.bindElement(bsplines::invr2_integrand);
        Inv_r.bindElement(bsplines::invr_integrand);
        S.bindElement(bsplines::overlap_integrand);
        Der.bindElement(bsplines::der_integrand);

        K.populateMatrix(sim,ECSMode::ON);
        Inv_r2.populateMatrix(sim,ECSMode::ON);
        Inv_r.populateMatrix(sim,ECSMode::ON);
        S.populateMatrix(sim,ECSMode::ON);
        Der.populateMatrix(sim,ECSMode::ON);

        K.assemble();
        Inv_r2.assemble();
        Inv_r.assemble();
        S.assemble();
        Der.assemble();

        PetscPrintf(PETSC_COMM_WORLD, "Saving Matrices  \n\n");
        PetscBinaryViewer viewK((sim.tise_output_path+"/K.bin").c_str());
        PetscBinaryViewer viewInv_r2((sim.tise_output_path+"/Inv_r2.bin").c_str());
        PetscBinaryViewer viewInv_r((sim.tise_output_path+"/Inv_r.bin").c_str());
        PetscBinaryViewer viewS((sim.tise_output_path+"/S.bin").c_str());
        PetscBinaryViewer viewDer((sim.tise_output_path+"/Der.bin").c_str());

        viewK.saveMatrix(K);
        viewInv_r2.saveMatrix(Inv_r2);
        viewInv_r.saveMatrix(Inv_r);
        viewS.saveMatrix(S);
        viewDer.saveMatrix(Der);

        double time_end = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"Time to prepare matrices %.3f\n",time_end-time_start);   
    }

    void solve_tise(const simulation& sim,int rank)
    {    
        double start_time = MPI_Wtime();
 

        create_directory(rank, "TISE_files");
        solve_eigensystem(sim); 
        prepare_matrices(sim); 



        double end_time = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"Time to solve TISE %.3f\n\n",end_time-start_time);
    }

    
}


