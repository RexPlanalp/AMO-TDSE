#pragma once

#include <petscviewer.h>
#include <petscviewerhdf5.h>

#include "petsc_wrappers/PetscVector.h"
#include "mpi.h"

//////////////////////////
// Petsc Viewer Wrapper //
//////////////////////////

enum class ViewerMode
{
    BINARY,
    HDF5
};

enum class OpenMode
{
    READ,
    WRITE
};

class PetscFileViewer
{
    public:

        // Default constructor
        PetscFileViewer() : viewer(nullptr) {}

        // Destructor
        ~PetscFileViewer();

        // Internal viewer
        PetscViewer viewer;
};

//////////////////////////
// HDF5 Viewer Wrapper  //
//////////////////////////

class PetscHDF5Viewer : public PetscFileViewer
{
    public:

        // Explicit Constructor
        PetscHDF5Viewer(const char* filename, RunMode run, OpenMode mode);  
        
        void saveVector(const PetscVector& vector, const char* groupname, const char* vectorname);
        void saveValue(std::complex<double> value, const char* groupname, const char* valuename);

        // Internal communicator
        MPI_Comm comm = MPI_COMM_NULL;

};

//////////////////////////
// HDF5 Binary Wrapper  //
//////////////////////////

class PetscBinaryViewer : public PetscFileViewer
{
    public:
    
        // Explicit Constructor
        PetscBinaryViewer(const char* filename, RunMode run, OpenMode mode);
        
        void saveMatrix(const PetscMatrix& matrix);
        PetscMatrix loadMatrix();

        // Internal communicator
        MPI_Comm comm = MPI_COMM_NULL;
};







