#pragma once

#include <petscviewer.h>
#include <petscviewerhdf5.h>

#include "petsc_wrappers/PetscVector.h"

//////////////////////////
// Petsc Viewer Wrapper //
//////////////////////////

enum class ViewerMode
{
    BINARY,
    HDF5
};

class PetscFileViewer
{
    public:

        // Default constructor
        PetscFileViewer() = default;

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
        PetscHDF5Viewer(const char* filename);  
        
        void saveVector(const PetscVector& vector, const char* groupname, const char* vectorname);
        void saveValue(std::complex<double> value, const char* groupname, const char* valuename);
};

//////////////////////////
// HDF5 Binary Wrapper  //
//////////////////////////

class PetscBinaryViewer : public PetscFileViewer
{
    public:
    
        // Explicit Constructor
        PetscBinaryViewer(const char* filename);  
        
        void saveMatrix(const PetscMatrix& matrix);
};







