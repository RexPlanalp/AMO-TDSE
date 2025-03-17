#pragma once

enum class RunMode;
class PetscVector;

#include <petscmat.h>
#include "simulation.h"
#include <functional>
#include "mpi.h"
//////////////////////////
// Petsc Matrix Wrapper //
//////////////////////////







class PetscMatrix
{
    public:

        // Default Constructor
        PetscMatrix() : matrix(nullptr), comm(MPI_COMM_NULL), local_start(0), local_end(0){}

        // Explicit Constructor
        PetscMatrix(int size, int nnz, RunMode type);

        // Copy Constructor
        PetscMatrix(const PetscMatrix& other);

        // Copy assignment operator
        PetscMatrix& operator=(const PetscMatrix& other);
        
        // Destructor
        ~PetscMatrix();

        // Assemble the matrix
        void assemble();

        // Internal matrix
        Mat matrix;
        MPI_Comm comm;
        int local_start,local_end;

};

//////////////////////////
//    Radial Subclass   //
//////////////////////////
enum class ECSMode
{
    ON,
    OFF
};


using radialElement = std::function<std::complex<double>(int,int,std::complex<double>,int,const std::vector<std::complex<double>>&)>;


class RadialMatrix : public PetscMatrix
{
    public:
    RadialMatrix(const simulation& sim, RunMode run, ECSMode ecs);
    
    // Populate the matrix
    void populateMatrix(const simulation& sim, radialElement integrand);

    const ECSMode ecs;
};  

//////////////////////////
//    Radial Subclass   //
//////////////////////////

enum class AngularMatrixType
{
    Z_INT_1,
    Z_INT_2,
    XY_INT_1,
    XY_INT_2,
    XY_INT_3,
    XY_INT_4
};

class AngularMatrix : public PetscMatrix
{
    public:
    AngularMatrix(const simulation& sim, RunMode run, AngularMatrixType type);
    
    // Populate the matrix
    void populateMatrix(const simulation& sim);

    const AngularMatrixType type;
};















