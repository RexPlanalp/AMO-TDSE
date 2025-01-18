#include "tise.h"

#include <petscmat.h>
#include <petscviewerhdf5.h>
#include <slepceps.h>
#include "simulation.h"
#include "bsplines.h"


namespace tise
{
    PetscErrorCode solve_tise(simulation& sim)
    {
        PetscErrorCode ierr;
        PetscViewer viewTISE;
        Mat K;
        Mat Inv_r2;
        Mat Inv_r;
        Mat S;
        Mat temp;
        EPS eps;
        int nconv;

        ierr = bsplines::construct_overlap(sim,S,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_kinetic(sim,K,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_invr2(sim,Inv_r2,true,true); CHKERRQ(ierr);
        ierr = bsplines::construct_invr(sim,Inv_r,true,true); CHKERRQ(ierr);

        ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,"tise_output.h5", FILE_MODE_WRITE, &viewTISE); CHKERRQ(ierr);

        ierr = EPSCreate(PETSC_COMM_WORLD, &eps); CHKERRQ(ierr);
        ierr = EPSSetProblemType(eps, EPS_GNHEP); CHKERRQ(ierr);
        ierr = EPSSetWhichEigenpairs(eps, EPS_SMALLEST_REAL); CHKERRQ(ierr);
        ierr = EPSSetType(eps,EPSKRYLOVSCHUR); CHKERRQ(ierr);
        ierr = EPSSetTolerances(eps,sim.tise_data.value("tolerance",1E-8),sim.tise_data.value("max_iter",2000)); CHKERRQ(ierr);

        for (int l = 0; sim.angular_data.value("lmax",0); ++l)
        {
            ierr = MatDuplicate(K,MAT_COPY_VALUES,&temp); CHKERRQ(ierr);
            ierr = MatAXPY(temp, l*(l+1)*0.5,Inv_r2,SAME_NONZERO_PATTERN); CHKERRQ(ierr);
            ierr = MatAXPY(temp,-1.0,Inv_r,SAME_NONZERO_PATTERN); CHKERRQ(ierr);

            int num_of_energies = sim.angular_data.value("nmax",0) - l;
            if (num_of_energies <= 0)
            {
                continue;
            }

            ierr = EPSSetOperators(eps,temp,S); CHKERRQ(ierr);
            ierr = EPSSetDimensions(eps,num_of_energies,PETSC_DEFAULT,PETSC_DEFAULT); CHKERRQ(ierr);
            ierr = EPSSolve(eps); CHKERRQ(ierr);
            ierr = EPSGetConverged(eps,&nconv); CHKERRQ(ierr);
            PetscPrintf(PETSC_COMM_WORLD, "Eigenvalues Requested %d, Eigenvalues Converged: %d \n", num_of_energies,nconv); CHKERRQ(ierr);

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

                std::string eigenvector_name = std::string("psi_l_") + std::to_string(i+l+1) + "_" + std::to_string(l);
                ierr = PetscViewerHDF5PushGroup(viewTISE, "/eigenvectors"); CHKERRQ(ierr);
                ierr = PetscObjectSetName((PetscObject)eigenvector,eigenvector_name.c_str()); CHKERRQ(ierr);
                ierr = VecView(eigenvector,viewTISE); CHKERRQ(ierr);
                ierr = PetscViewerHDF5PopGroup(viewTISE); CHKERRQ(ierr);

                ierr = VecDestroy(&eigenvector); CHKERRQ(ierr);

            }
            ierr = MatDestroy(&temp); CHKERRQ(ierr);

        }

        ierr = PetscViewerDestroy(&viewTISE); CHKERRQ(ierr);
        ierr = EPSDestroy(&eps); CHKERRQ(ierr);
        ierr = MatDestroy(&K); CHKERRQ(ierr);
        ierr = MatDestroy(&Inv_r2); CHKERRQ(ierr);
        ierr = MatDestroy(&Inv_r); CHKERRQ(ierr);
        ierr = MatDestroy(&S); CHKERRQ(ierr);
        ierr = MatDestroy(&temp); CHKERRQ(ierr);
    }
}


