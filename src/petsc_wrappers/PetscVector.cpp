#include <petscvec.h>
#include "petsc_wrappers/PetscVector.h"
#include <complex>
#include "utility.h"
#include "mpi.h"

//////////////////////////
// Petsc Vector Wrapper //
//////////////////////////

PetscVector::PetscVector(int size, RunMode type)
{
    PetscErrorCode ierr;

    switch(type)
    {
        case RunMode::SEQUENTIAL:
            comm = PETSC_COMM_SELF;

            ierr = VecCreate(comm, &vector); checkErr(ierr, "Error Creating Vector");
            ierr = VecSetSizes(vector, PETSC_DECIDE, size); checkErr(ierr, "Error Setting Vector Size");
            ierr = VecSetType(vector, VECSEQ); checkErr(ierr, "Error Setting Vector Type");
            ierr = VecSetFromOptions(vector); checkErr(ierr, "Error Setting Vector Options");
            ierr = VecSetUp(vector); checkErr(ierr, "Error Setting Up Vector");

            local_start = 0;
            local_end = size;
            break;

        case RunMode::PARALLEL:
            comm = PETSC_COMM_WORLD;


            ierr = VecCreate(comm, &vector); checkErr(ierr, "Error Creating Vector");
            ierr = VecSetSizes(vector, PETSC_DECIDE, size); checkErr(ierr, "Error Setting Vector Size");
            ierr = VecSetType(vector,VECMPI); checkErr(ierr, "Error Setting Vector Type");
            ierr = VecSetFromOptions(vector); checkErr(ierr, "Error Setting Vector Options");
            ierr = VecSetUp(vector); checkErr(ierr, "Error Setting Up Vector");
            ierr = VecGetOwnershipRange(vector, &local_start, &local_end); checkErr(ierr, "Error Getting Ownership Range");
            break;
    }
}

PetscVector::PetscVector(const PetscVector& other)
{
    PetscErrorCode ierr;
    ierr = VecDuplicate(other.vector, &vector); checkErr(ierr, "Error duplicating vector");
    ierr = VecCopy(other.vector,vector); checkErr(ierr, "Error copying vector");
    comm = other.comm;
    local_start = other.local_start;
    local_end = other.local_end;
}

PetscVector& PetscVector::operator=(const PetscVector& other)
{
    if (this != &other)  
    {
        if (vector) 
        {
            VecDestroy(&vector);
            vector = nullptr;
        }

        PetscErrorCode ierr = VecDuplicate(other.vector, &vector);
        checkErr(ierr, "Error duplicating vector");
        ierr = VecCopy(other.vector, vector);
        checkErr(ierr, "Error copying vector");

        comm = other.comm;
        local_start = other.local_start;
        local_end = other.local_end;
    }
    return *this;
}

PetscVector::~PetscVector()
{
    if (vector) 
    {
        PetscErrorCode ierr;
        ierr = VecDestroy(&vector); checkErr(ierr, "Error destroying vector");
        vector = nullptr;
    }
}

void PetscVector::assemble()
{
    VecAssemblyBegin(vector);
    VecAssemblyEnd(vector);
}

void PetscVector::setName(const char* name)
{
    PetscErrorCode ierr;
    ierr = PetscObjectSetName((PetscObject)vector,name); checkErr(ierr,"Error setting name");
}

//////////////////////////
// Wavefunction Subclass//
//////////////////////////


std::complex<double> Wavefunction::computeNorm(const PetscMatrix& S) const
{
    PetscErrorCode ierr;
    std::complex<double> normValue = 0.0;

    // Allocate the temporary vector using S.matrix
    PetscVector temp;
    ierr = MatCreateVecs(S.matrix, &temp.vector, NULL); checkErr(ierr, "Error creating temporary vector");

    ierr = MatMult(S.matrix, vector, temp.vector); checkErr(ierr, "Error in MatMult");
    ierr = VecDot(vector, temp.vector, &normValue); checkErr(ierr, "Error in VecDot");
    
    return std::sqrt(normValue);
}

void Wavefunction::normalize(const PetscMatrix& S)
{
    PetscErrorCode ierr;
    std::complex<double> norm = computeNorm(S);
    ierr = VecScale(vector,1.0/norm); checkErr(ierr, "Error scaling eigenvector");
}





