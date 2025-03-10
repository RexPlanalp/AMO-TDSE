#include "viewer.h"
#include "matrix.h"
#include "vector.h"
#include "utility.h"

#include <petscviewer.h>
#include <petscviewerhdf5.h>


////////////////////////////
//    Saver Baseclass     //
////////////////////////////

// Constructor: default

// Destructory: destroy the viewer
PetscSaver::~PetscSaver()
{
    PetscViewerDestroy(&getViewer());
}

// Method: get the viewer
PetscViewer& PetscSaver::getViewer()
{
    return viewer;
}

////////////////////////////
//    Binary Subclass     //
////////////////////////////


// Constructor: save to filename
PetscSaverBinary::PetscSaverBinary(const char* filename)
{
    PetscErrorCode ierr;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&getViewer()); checkError(ierr,"Error opening binary file");
} 

// Method: save the matrix
void PetscSaverBinary::saveMatrix(PetscMatrix& matrix)
{   
    PetscErrorCode ierr;
    ierr = MatView(matrix.getMatrix(),getViewer()); checkError(ierr,"Error saving matrix");
}

////////////////////////////
//     HDF5 Subclass      //
////////////////////////////

// Constructor: save to filename
PetscSaverHDF5::PetscSaverHDF5(const char* filename)
{
    PetscErrorCode ierr;
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&getViewer()); checkError(ierr,"Error opening HDF5 file");
}

// Method: save the vector
void PetscSaverHDF5::saveVector(PetscVector& vector, const char* groupname,const char* vectorname)
{
    PetscErrorCode ierr;

    ierr = PetscViewerHDF5PushGroup(getViewer(),groupname); checkError(ierr,"Error pushing group");
    ierr = PetscObjectSetName((PetscObject)vector.getVector(),vectorname); checkError(ierr,"Error setting name");
    ierr = VecView(vector.getVector(),getViewer()); checkError(ierr,"Error saving vector");
    ierr = PetscViewerHDF5PopGroup(getViewer()); checkError(ierr,"Error popping group");
}


// Method: save the value

void PetscSaverHDF5::saveValue(std::complex<double> value, const char* groupname, const char* valuename)
{
    PetscErrorCode ierr;

    ierr = PetscViewerHDF5PushGroup(getViewer(),groupname); checkError(ierr,"Error pushing group");

    PetscVector temp(1,VectorType::SEQUENTIAL);
    temp.setValue(0,value);
    temp.assemble();
    ierr = PetscObjectSetName((PetscObject)temp.getVector(),valuename); checkError(ierr,"Error setting name");
    ierr = VecView(temp.getVector(),getViewer()); checkError(ierr,"Error saving value");
    ierr = PetscViewerHDF5PopGroup(getViewer()); checkError(ierr,"Error popping group"); 
}