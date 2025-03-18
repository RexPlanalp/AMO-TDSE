#include <iostream>
#include <sys/stat.h>

#include <petscsys.h>
#include <gsl/gsl_sf_legendre.h>
#include "utility.h"
#include "petsc_wrappers/PetscMatrix.h"
#include "petsc_wrappers/PetscVector.h"
#include "simulation.h"
#include "petsc_wrappers/PetscFileViewer.h"
#include "petsc_wrappers/PetscIS.h"

using json = nlohmann::json;

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

// Wrapper for gsl spherical harmonics
std::complex<double> compute_Ylm(int l, int m, double theta, double phi)
{


// Compute negative m using identity Y_{l,-m} = (-1)^m Y_{l,m}^*
if (m < 0) 
{
    int abs_m = -m;
    std::complex<double> Ylm = 
                                gsl_sf_legendre_sphPlm(l, abs_m, std::cos(theta)) * 
                                std::exp(std::complex<double>(0, -abs_m * phi));

    return std::pow(-1.0, abs_m) * std::conj(Ylm);
}

// For positive m evaluate as normal
return  
        gsl_sf_legendre_sphPlm(l, m, std::cos(theta)) * 
        std::exp(std::complex<double>(0, m * phi));
}

void save_lm_expansion(const std::map<std::pair<int, int>, int>& lm_to_block, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile) {
        std::cerr << "Error opening file for writing.\n";
        return;
    }
    
    for (const auto& entry : lm_to_block) {
        outFile << entry.first.first << " " << entry.first.second << " " << entry.second << "\n";
    }
    
    outFile.close();
}

void create_directory(int rank, const std::string& directory)
{
    if (rank == 0) 
    {
        if (mkdir(directory.c_str(), 0777) == -1) 
        {
            if (errno == EEXIST) 
            {
                std::cout << "Directory already exists: " << directory << "\n\n";
            }
            else 
            {
                throw std::runtime_error("Failed to create directory " + directory + 
                                       ": " + std::strerror(errno));
            }
        }
        else 
        {
            std::cout << "Directory created: " << directory << "\n\n";
        }
    }
}

void normalize_array(std::array<double,3>& vec)
{      
    double norm = std::sqrt(vec[0]*vec[0] + vec[1]*vec[1] + vec[2]*vec[2]);

    if (norm == 0) return;
    
    for (size_t idx = 0; idx< vec.size(); ++idx)
    {
        vec[idx] /= norm;
    }
}

void cross_product(const std::array<double,3>& vec1, const std::array<double,3>& vec2, std::array<double,3>& result)
{
    result[0] = vec1[1]*vec2[2] - vec1[2]*vec2[1];
    result[1] = vec1[2]*vec2[0] - vec1[0]*vec2[2];
    result[2] = vec1[0]*vec2[1] - vec1[1]*vec2[0];
}

void pes_pointwise_mult(const std::vector<double>& vec1, const std::vector<std::complex<double>>& vec2, std::vector<std::complex<double>>& result) {
    result.resize(vec1.size());  // Correct: Set the size

    for (size_t idx = 0; idx < vec1.size(); ++idx) {
        result[idx] = vec1[idx] * vec2[idx]; // Direct assignment: No push_back!
    }
}

void pes_pointwise_add(const std::vector<std::complex<double>>& vec1, const std::vector<std::complex<double>>& vec2, std::vector<std::complex<double>>& result) {
    result.resize(vec1.size());

    for (size_t idx = 0; idx < vec1.size(); ++idx) {
        result[idx] = vec1[idx] + vec2[idx]; // Direct assignment
    }
}

void pes_pointwise_magsq(const std::vector<std::complex<double>>& vec, std::vector<std::complex<double>>& result) {
    result.resize(vec.size());

    for (size_t idx = 0; idx < vec.size(); ++idx) {
        result[idx] = vec[idx] * std::conj(vec[idx]); // Direct assignment
    }
}

std::complex<double> pes_simpsons_method(const std::vector<std::complex<double>>& vec,double dr)
{
    int n = vec.size() - 1;
    std::complex<double> I {};


    if (n % 2 != 0)
    {
        I += (3 * dr / 8) * (vec[n-2]+static_cast<double>(3)*vec[n-1]+static_cast<double>(3)*vec[n]);
        n -= 2;
    }

    double p = dr / 3; 
    I += (vec[0]+vec[n]) * p;
    for (int vec_idx = 1; vec_idx < n; vec_idx += 2)
    {
        I += static_cast<double>(4) * vec[vec_idx] * p;
    }
    for (int vec_idx = 2; vec_idx < n-1; vec_idx += 2)
    {
        I += static_cast<double>(2) * vec[vec_idx] * p;
    }
    
    return I;

}

std::complex<double> H(std::complex<double> r)
{
    return -1.0/(r+1E-25);
}

std::complex<double> He(std::complex<double> r)
{
    return -1.0/(r+1E-25) - 1.0*std::exp(-2.0329*r)/(r+1E-25) - 0.3953*std::exp(-6.1805*r);
}

void scale_vector(std::vector<double>& vec, double scale)
{
    for (double& val : vec)
    {
        val *= scale;
    }
}


double alpha(int l, int m)
{
    double numerator = (l+m-1)*(l+m);
    double denominator = 4*(2*l+1)*(2*l-1);
    return std::sqrt(numerator/denominator);
}

double beta(int l, int m)
{
    double numerator = (l-m+1)*(l-m+2)*(l+1);
    double denominator = 2*(2*l+1)*(2*l+2)*(2*l+3);
    return -std::sqrt(numerator/denominator);
}

double charlie(int l, int m)
{
    double numerator = (l-m-1)*(l-m);
    double denominator = 4*(2*l+1)*(2*l-1);
    return std::sqrt(numerator/denominator);
}

double delta(int l, int m)
{
    double numerator = (l+m+1)*(l+m+2)*(l+1);
    double denominator = 2*(2*l+1)*(2*l+2)*(2*l+3);
    return -std::sqrt(numerator/denominator);
}

double echo(int l, int m)
{
    double numerator = (l+m)*(l-m);
    double denominator = (2*l-1)*(2*l+1);
    return std::sqrt(numerator/denominator);
}

double foxtrot(int l, int m)
{
    double numerator = (l+m+1)*(l-m+1);
    double denominator = (2*l+1)*(2*l+3);
    return std::sqrt(numerator/denominator);
}