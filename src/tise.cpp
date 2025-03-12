#include <sys/stat.h>
#include <iostream>


#include <petscmat.h>
#include <petscviewerhdf5.h>
#include <slepceps.h>


#include "tise.h"
#include "simulation.h"
#include "bsplines.h"
#include "utility.h"


// NEW INCLUDES

#include "matrix.h"
#include "vector.h"
#include "viewer.h"
#include "eps.h"
#include "utility.h"



namespace tise
{

  

    

    void solve_eigensystem(const simulation& sim)
    {   
        PetscPrintf(PETSC_COMM_WORLD, "Constructing Matrices  \n\n");


        RadialMatrix S(sim,RadialMatrixType::PARALLEL);
        S.setIntegrand(bsplines::overlap_integrand);
        S.populateMatrix(sim,ECSMode::OFF);

        RadialMatrix K(sim,RadialMatrixType::PARALLEL);
        K.setIntegrand(bsplines::kinetic_integrand);
        K.populateMatrix(sim,ECSMode::OFF);

        RadialMatrix Inv_r2(sim,RadialMatrixType::PARALLEL);
        Inv_r2.setIntegrand(bsplines::invr2_integrand);
        Inv_r2.populateMatrix(sim,ECSMode::OFF);

        RadialMatrix Inv_r(sim,RadialMatrixType::PARALLEL);
        Inv_r.setIntegrand(bsplines::invr_integrand);
        Inv_r.populateMatrix(sim,ECSMode::OFF);


        PetscPrintf(PETSC_COMM_WORLD, "Opening HDF5 File  \n\n");
        PetscSaverHDF5 saveTISE((sim.tise_output_path+"/tise_output.h5").c_str());


        PetscPrintf(PETSC_COMM_WORLD, "Setting Up Eigenvalue Problem  \n\n");
        PetscEPS eps(sim);

        PetscPrintf(PETSC_COMM_WORLD, "Solving TISE  \n\n");

        int nconv;
        PetscMatrix temp;

        for (int l = 0; l<= sim.angular_params.lmax; ++l)
        {
            temp.duplicateMatrix(K);
            temp.axpy(l*(l+1)*0.5,Inv_r2,MatStructure::SAME_NONZERO_PATTERN);
            temp.axpy(-1.0,Inv_r,MatStructure::SAME_NONZERO_PATTERN);
       
            int num_of_energies = sim.angular_params.nmax - l;
            if (num_of_energies <= 0)
            {
                continue;
            }

            
            eps.setParameters(num_of_energies);
            eps.setOperators(temp,S);
            eps.solve(nconv);

            PetscPrintf(PETSC_COMM_WORLD, "Eigenvalues Requested %d, Eigenvalues Converged: %d \n\n", num_of_energies,nconv);

            for (int i = 0; i < nconv; i++)
            {
                std::complex<double> eigenvalue;
                eps.getEigenvalue(i,eigenvalue);
                
                if (eigenvalue.real()>0)
                {
                    continue;
                }
                
                std::string eigenvalue_name = std::string("E_") + std::to_string(i+l+1) + '_' + std::to_string(l);
                saveTISE.saveValue(eigenvalue,"eigenvalues",eigenvalue_name.c_str());



                std::string eigenvector_name = std::string("psi_") + std::to_string(i+l+1) + '_' + std::to_string(l);
                PetscVector eigenvector;
                eps.getNormalizedEigenvector(eigenvector,i,S);
                
                
                
                if (sim.debug)
                {   
                    std::complex<double> norm;
                    eigenvector.computeNorm(norm,S);
                    PetscPrintf(PETSC_COMM_WORLD,"Eigenvector %d -> Norm(%.4f , %.4f) -> Eigenvalue(%.4f , %.4f)  \n",i+1,norm.real(),norm.imag(),eigenvalue.real(),eigenvalue.imag()); 
                }
                saveTISE.saveVector(eigenvector,"eigenvectors",eigenvector_name.c_str());

            }
        }
        
        PetscPrintf(PETSC_COMM_WORLD, "Destroying Petsc Objects  \n\n");

    }

    void prepare_matrices(const simulation& sim)
    {   
        double time_start = MPI_Wtime();

        RadialMatrix S(sim,RadialMatrixType::PARALLEL);
        S.setIntegrand(bsplines::overlap_integrand);
        S.populateMatrix(sim,ECSMode::ON);

        RadialMatrix K(sim,RadialMatrixType::PARALLEL);
        K.setIntegrand(bsplines::kinetic_integrand);
        K.populateMatrix(sim,ECSMode::ON);

        RadialMatrix Inv_r2(sim,RadialMatrixType::PARALLEL);
        Inv_r2.setIntegrand(bsplines::invr2_integrand);
        Inv_r2.populateMatrix(sim,ECSMode::ON);

        RadialMatrix Inv_r(sim,RadialMatrixType::PARALLEL);
        Inv_r.setIntegrand(bsplines::invr_integrand);
        Inv_r.populateMatrix(sim,ECSMode::ON);

        RadialMatrix Der(sim,RadialMatrixType::PARALLEL);
        Der.setIntegrand(bsplines::der_integrand);
        Der.populateMatrix(sim,ECSMode::ON);

        PetscPrintf(PETSC_COMM_WORLD, "Saving Matrices  \n\n");

        S.saveMatrix((sim.tise_output_path+"/S.bin").c_str());
        K.saveMatrix((sim.tise_output_path+"/K.bin").c_str());
        Inv_r2.saveMatrix((sim.tise_output_path+"/Inv_r2.bin").c_str());
        Inv_r.saveMatrix((sim.tise_output_path+"/Inv_r.bin").c_str());
        Der.saveMatrix((sim.tise_output_path+"/Der.bin").c_str());
        

        

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


