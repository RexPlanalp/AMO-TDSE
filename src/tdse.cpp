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
#include "petsc_wrappers/PetscKSP.h"

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

        Wavefunction starting_state(total_size, RunMode::PARALLEL);
        ierr = VecSet(starting_state.vector, 0.0); checkErr(ierr,"Error zeroing vector");

        PetscVector tise_state(n_basis,RunMode::SEQUENTIAL);
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

    PetscMatrix construct_S_atomic(const simulation& sim)
    {
        PetscErrorCode ierr;
        int n_blocks = sim.angular_params.n_blocks;

        PetscBinaryViewer SViewer((sim.tise_output_path+"/S.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix S = SViewer.loadMatrix();

    
        PetscMatrix I(n_blocks,1,RunMode::SEQUENTIAL);
        for (int i = 0; i < n_blocks; ++i)
        {
            ierr = MatSetValue(I.matrix,i,i,1.0,INSERT_VALUES); checkErr(ierr,"Error setting value");
        }
        I.assemble();



        PetscMatrix S_atomic = KroneckerProduct(I,S); 

        
        return S_atomic;
    }

    PetscMatrix construct_H_atomic(const simulation& sim)
    {
        PetscErrorCode ierr; 

        int n_blocks = sim.angular_params.n_blocks;
        int n_basis = sim.bspline_params.n_basis;
        int degree = sim.bspline_params.degree;
        int lmax = sim.angular_params.lmax;
        int total_size = n_basis * n_blocks;

        PetscBinaryViewer KViewer((sim.tise_output_path+"/K.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix K = KViewer.loadMatrix();

        PetscBinaryViewer Inv_r2Viewer((sim.tise_output_path+"/Inv_r2.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix Inv_r2 = Inv_r2Viewer.loadMatrix();

        PetscBinaryViewer PotentialViewer((sim.tise_output_path+"/Potential.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix Potential = PotentialViewer.loadMatrix();

        ierr = MatAXPY(K.matrix,1.0,Potential.matrix,SAME_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");

        PetscMatrix H_atomic(total_size,2*degree+1,RunMode::PARALLEL);
        H_atomic.assemble();

        for (int l = 0; l<=lmax; ++l)
        {
            PetscMatrix temp(K);


            ierr = MatAXPY(temp.matrix,l*(l+1)*0.5,Inv_r2.matrix,SAME_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");

            std::vector<int> indices;
            for (int i = 0; i < n_blocks; ++i)
            {   
                auto lm_pair = sim.angular_params.block_to_lm.at(i);  // Retrieve (l, m) pair
                if (lm_pair.first == l) // Compare only 'l' component
                {
                    indices.push_back(i);
                }
            }

            PetscMatrix I_partial(n_blocks,1,RunMode::SEQUENTIAL);
            for (int i = 0; i<n_blocks; ++i)
            {
                if (std::find(indices.begin(),indices.end(),i) != indices.end())
                {
                    ierr = MatSetValue(I_partial.matrix,i,i,1.0,INSERT_VALUES); checkErr(ierr,"Error setting value");
                }
            }
            I_partial.assemble();

            PetscMatrix H_partial = KroneckerProduct(I_partial,temp); checkErr(ierr,"Error in KroneckerProduct");
            ierr = MatAXPY(H_atomic.matrix,1.0,H_partial.matrix,DIFFERENT_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");
            
        }

        return H_atomic;






    }

    std::pair<PetscMatrix,PetscMatrix> construct_atomic_interaction(const simulation& sim,std::complex<double> alpha)
    {   
        PetscErrorCode ierr;
        PetscMatrix S_atomic = construct_S_atomic(sim);
        PetscMatrix H_atomic = construct_H_atomic(sim);
        
        PetscMatrix atomic_left(S_atomic);
        PetscMatrix atomic_right(S_atomic);

        ierr = MatAXPY(atomic_left.matrix,alpha,H_atomic.matrix,SAME_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");
        ierr = MatAXPY(atomic_right.matrix,-alpha,H_atomic.matrix,SAME_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");
        
        return {atomic_left,atomic_right};
    }
 
    std::pair<PetscMatrix,PetscMatrix> construct_xy_interaction(const simulation& sim)
    {
        PetscErrorCode ierr;

        PetscBinaryViewer Inv_rViewer((sim.tise_output_path+"/Inv_r.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix Inv_r = Inv_rViewer.loadMatrix();

        PetscBinaryViewer DerViewer((sim.tise_output_path+"/Der.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix Der = DerViewer.loadMatrix();

        AngularMatrix H_lm_1(sim,RunMode::SEQUENTIAL,AngularMatrixType::XY_INT_1,2);
        H_lm_1.populateMatrix(sim);

        AngularMatrix H_lm_2(sim,RunMode::SEQUENTIAL,AngularMatrixType::XY_INT_2,2);
        H_lm_2.populateMatrix(sim);


        

        PetscMatrix H_xy_1 = KroneckerProduct(H_lm_1,Inv_r);
        PetscMatrix H_xy_2 = KroneckerProduct(H_lm_2,Der);
        ierr = MatAXPY(H_xy_1.matrix,1.0,H_xy_2.matrix,SAME_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");
        

        AngularMatrix H_lm_3(sim,RunMode::SEQUENTIAL,AngularMatrixType::XY_INT_3,2);
        H_lm_3.populateMatrix(sim);

        AngularMatrix H_lm_4(sim,RunMode::SEQUENTIAL,AngularMatrixType::XY_INT_4,2);
        H_lm_4.populateMatrix(sim);

        PetscMatrix H_xy_tilde_1 = KroneckerProduct(H_lm_3,Inv_r);
        PetscMatrix H_xy_tilde_2 = KroneckerProduct(H_lm_4,Der);

    
        ierr = MatAXPY(H_xy_tilde_1.matrix,1.0,H_xy_tilde_2.matrix,SAME_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");

        return {H_xy_1,H_xy_tilde_1};

    }

    PetscMatrix construct_z_interaction(const simulation& sim)
    {
        PetscErrorCode ierr;

        AngularMatrix H_lm_1(sim,RunMode::SEQUENTIAL,AngularMatrixType::Z_INT_1,2);
        H_lm_1.populateMatrix(sim);

        AngularMatrix H_lm_2(sim,RunMode::SEQUENTIAL,AngularMatrixType::Z_INT_2,2);
        H_lm_2.populateMatrix(sim);


        PetscBinaryViewer Inv_rViewer((sim.tise_output_path+"/Inv_r.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix Inv_r = Inv_rViewer.loadMatrix();

        PetscBinaryViewer DerViewer((sim.tise_output_path+"/Der.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix Der = DerViewer.loadMatrix();

        PetscMatrix H_z_1 = KroneckerProduct(H_lm_1,Der);
        PetscMatrix H_z_2 = KroneckerProduct(H_lm_2,Inv_r);

        ierr = MatAXPY(H_z_1.matrix,1.0,H_z_2.matrix,SAME_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");
        return H_z_1;
    }

    PetscMatrix construct_x_hhg(const simulation& sim)
    {
        AngularMatrix H_lm_x(sim,RunMode::SEQUENTIAL,AngularMatrixType::X_HHG,4);
        H_lm_x.populateMatrix(sim);

        PetscBinaryViewer Inv_r2Viewer((sim.tise_output_path+"/Inv_r2.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix Inv_r2 = Inv_r2Viewer.loadMatrix();

        PetscMatrix H_x = KroneckerProduct(H_lm_x,Inv_r2);

        return H_x;
    }

    PetscMatrix construct_y_hhg(const simulation& sim)
    {
        AngularMatrix H_lm_y(sim,RunMode::SEQUENTIAL,AngularMatrixType::Y_HHG,4);
        H_lm_y.populateMatrix(sim);

        PetscBinaryViewer Inv_r2Viewer((sim.tise_output_path+"/Inv_r2.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix Inv_r2 = Inv_r2Viewer.loadMatrix();

        PetscMatrix H_y = KroneckerProduct(H_lm_y,Inv_r2);

        return H_y;
    }

    PetscMatrix construt_z_hhg(const simulation& sim)
    {
        AngularMatrix H_lm_z(sim,RunMode::SEQUENTIAL,AngularMatrixType::Z_HHG,2);
        H_lm_z.populateMatrix(sim);

        PetscBinaryViewer Inv_r2Viewer((sim.tise_output_path+"/Inv_r2.bin").c_str(), RunMode::SEQUENTIAL, OpenMode::READ);
        PetscMatrix Inv_r2 = Inv_r2Viewer.loadMatrix();

        PetscMatrix H_z = KroneckerProduct(H_lm_z,Inv_r2);

        return H_z;
    }




    PetscErrorCode solve_tdse(const simulation& sim, int rank)
    {   
        double time_start = MPI_Wtime();
        PetscErrorCode ierr; 
        
        Vec3 components = sim.laser_params.components;
        double dt = sim.grid_params.dt;
        int Nt = sim.grid_params.Nt;
        std::complex<double> alpha = PETSC_i * (dt / 2.0);
        
        
        

        Wavefunction state = load_starting_state(sim); CHKERRQ(ierr);
        create_directory(rank, sim.tdse_output_path);
        PetscHDF5Viewer viewTDSE((sim.tdse_output_path+"/tdse_output.h5").c_str(),RunMode::PARALLEL,OpenMode::WRITE);

        std::ofstream hhg_file; 

        if (sim.observable_params.hhg && rank == 0)
        {
            hhg_file.open(sim.tdse_output_path + "/hhg_data.txt");
            hhg_file << std::fixed << std::setprecision(15);
        }
        
        PetscPrintf(PETSC_COMM_WORLD, "Constructing Atomic Interaction\n\n");
        PetscMatrix atomic_left,atomic_right;
        std::tie(atomic_left, atomic_right) = construct_atomic_interaction(sim, alpha);

        PetscMatrix S_atomic = construct_S_atomic(sim);

        std::cout << sim.observable_params.hhg << std::endl;

        PetscMatrix H_x_hhg;
        if (components[0] && sim.observable_params.hhg)
        {
            PetscPrintf(PETSC_COMM_WORLD, "Constructing X HHG Interaction\n\n");
            H_x_hhg = construct_x_hhg(sim);
        }

        PetscMatrix H_y_hhg;
        if (components[1] && sim.observable_params.hhg)
        {
            PetscPrintf(PETSC_COMM_WORLD, "Constructing Y HHG Interaction\n\n");
            H_y_hhg = construct_y_hhg(sim);
        }

        PetscMatrix H_z_hhg;
        if (components[2] && sim.observable_params.hhg)
        {
            PetscPrintf(PETSC_COMM_WORLD, "Constructing Z HHG Interaction\n\n");
            H_z_hhg = construt_z_hhg(sim);
        }

        if (sim.observable_params.hhg)
        {
            std::complex<double> x_val = 0.0;
            std::complex<double> y_val = 0.0;
            std::complex<double> z_val = 0.0;

            if (components[0])
            {
                x_val = state.computeNorm(H_x_hhg);
            }
            if (components[1])
            {
                y_val = state.computeNorm(H_y_hhg);
            }
            if (components[2])
            {
                z_val = state.computeNorm(H_z_hhg);
            }

            if (rank == 0)
            {
                hhg_file << 0.0 << " " << x_val.real() << " "  << laser::A(0.0,sim,0) << " " << y_val.real() << " " << laser::A(0.0,sim,1)  << " " << z_val.real() << " " << laser::A(0.0,sim,2) << std::endl;
            }
        }

       


        PetscMatrix H_z;
        if (components[2]) 
        {   
            PetscPrintf(PETSC_COMM_WORLD, "Constructing Z Interaction\n\n");
            H_z = construct_z_interaction(sim); checkErr(ierr,"Error constructing Z interaction");
        }
        PetscMatrix H_xy, H_xy_tilde;
        if (components[0] || components[1]) 
        {
            PetscPrintf(PETSC_COMM_WORLD, "Constructing XY Interaction\n\n");
            std::tie(H_xy, H_xy_tilde) = construct_xy_interaction(sim);
        }
        
        std::complex<double> norm = state.computeNorm(S_atomic);
        PetscPrintf(PETSC_COMM_WORLD, "Norm of Initial State: (%.4f,%.4f)\n\n", norm.real(), norm.imag());


        PetscPrintf(PETSC_COMM_WORLD, "Setting up Linear Solver\n\n");
        PetscKSP ksp(RunMode::PARALLEL);
        ksp.setConvergenceParams(sim);

        PetscPrintf(PETSC_COMM_WORLD, "Preallocating Temporary Petsc Objects\n\n");
        PetscVector state_temp(state);
        
        PetscPrintf(PETSC_COMM_WORLD, "Solving TDSE\n\n");
        for (int idx = 0; idx < Nt; idx++) 
        {   
            if (sim.debug)
            {
                PetscPrintf(PETSC_COMM_WORLD, "Time Step: %d/%d\n", idx,Nt);
            }
            
            double t = idx * dt;

            // Destroy and recreate temp matrices to avoid accumulation of structural changes
           
            PetscMatrix atomic_left_temp(atomic_left);
            PetscMatrix atomic_right_temp(atomic_right);


            if (components[2]) 
            {

                double laser_val = laser::A(t+dt/2.0, sim, 2);
                ierr = MatAXPY(atomic_left_temp.matrix, alpha * laser_val, H_z.matrix, DIFFERENT_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");
                ierr = MatAXPY(atomic_right_temp.matrix, -alpha * laser_val, H_z.matrix, DIFFERENT_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");
            }
            else if (components[0] || components[1])
            {
                std::complex<double> A_tilde = laser::A(t+dt/2.0, sim, 0) + PETSC_i*laser::A(t+dt/2.0, sim, 1);
                std::complex<double> A_tilde_star = laser::A(t+dt/2.0, sim, 0) - PETSC_i*laser::A(t+dt/2.0, sim, 1);

                ierr = MatAXPY(atomic_left_temp.matrix,alpha*A_tilde_star,H_xy.matrix,DIFFERENT_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");
                ierr = MatAXPY(atomic_right_temp.matrix,-alpha*A_tilde_star,H_xy.matrix,DIFFERENT_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");

                ierr = MatAXPY(atomic_left_temp.matrix,alpha*A_tilde,H_xy_tilde.matrix,DIFFERENT_NONZERO_PATTERN);  checkErr(ierr,"Error in MatAXPY");
                ierr = MatAXPY(atomic_right_temp.matrix,-alpha*A_tilde,H_xy_tilde.matrix,DIFFERENT_NONZERO_PATTERN); checkErr(ierr,"Error in MatAXPY");

            }

            ierr = MatMult(atomic_right_temp.matrix, state.vector, state_temp.vector); checkErr(ierr,"Error in MatMult");

            ksp.setOperators(atomic_left_temp);

            ierr = KSPSolve(ksp.ksp, state_temp.vector, state.vector); checkErr(ierr,"Error in KSPSolve");

            if (sim.observable_params.hhg)
            {
                std::complex<double> x_val = 0.0;
                std::complex<double> y_val = 0.0;
                std::complex<double> z_val = 0.0;

                if (components[0])
                {
                    x_val = state.computeNorm(H_x_hhg);
                }
                if (components[1])
                {
                    y_val = state.computeNorm(H_y_hhg);
                }
                if (components[2])
                {
                    z_val = state.computeNorm(H_z_hhg);
                }

                if (rank == 0)
                {
                    hhg_file << 0.0 << " " << x_val.real() << " "  << laser::A(t,sim,0) << " " << y_val.real() << " " << laser::A(t,sim,1)  << " " << z_val.real() << " " << laser::A(t,sim,2) << std::endl;
                }
            }

            
        }

        norm = state.computeNorm(S_atomic);
        PetscPrintf(PETSC_COMM_WORLD, "Norm of Final State: (%.15f,%.15f)\n\n", double(norm.real()), double(norm.imag()));

        PetscPrintf(PETSC_COMM_WORLD, "Saving Final State\n\n");
        viewTDSE.saveVector(state,"","final_state");

        if (sim.observable_params.hhg && rank == 0)
        {
            hhg_file.close();
        }


        double time_end = MPI_Wtime();
        PetscPrintf(PETSC_COMM_WORLD,"Time to solve TDSE %.3f\n",time_end-time_start);

        return ierr;
    }
}
