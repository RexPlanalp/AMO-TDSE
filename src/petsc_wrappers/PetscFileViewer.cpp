#include <petscviewer.h>
#include <petscviewerhdf5.h>
#include "petsc_wrappers/PetscFileViewer.h"
#include "utility.h"

//////////////////////////
// Petsc Viewer Wrapper //
//////////////////////////

PetscFileViewer::~PetscFileViewer()
{
    PetscViewerDestroy(&viewer);
}

//////////////////////////
// HDF5 Viewer Wrapper  //
//////////////////////////


PetscHDF5Viewer::PetscHDF5Viewer(const char* filename)
{
    PetscErrorCode ierr;
    ierr = PetscViewerHDF5Open(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE, &viewer); checkErr(ierr, "Error creating HDF5 viewer");
}

void PetscHDF5Viewer::saveVector(const PetscVector& input_vector, const char* groupname, const char* vectorname)
{
    PetscErrorCode ierr;
    ierr = PetscViewerHDF5PushGroup(viewer,groupname); checkErr(ierr, "Error pushing group");
    ierr = PetscObjectSetName((PetscObject)input_vector.vector, vectorname); checkErr(ierr, "Error setting vector name");
    ierr = VecView(input_vector.vector, viewer); checkErr(ierr, "Error viewing vector");
    ierr = PetscViewerHDF5PopGroup(viewer); checkErr(ierr, "Error popping group");
}

void PetscHDF5Viewer::saveValue(std::complex<double> value, const char* groupname, const char* valuename)
{
    PetscErrorCode ierr;
    PetscVector temp;

 
    ierr = VecCreate(PETSC_COMM_WORLD, &temp.vector); checkErr(ierr, "Error creating vector");
    ierr = VecSetSizes(temp.vector,PETSC_DECIDE,1); checkErr(ierr, "Error setting vector size");
    ierr = VecSetFromOptions(temp.vector); checkErr(ierr, "Error setting vector options");
    ierr=  VecSetValue(temp.vector,0,value,INSERT_VALUES); checkErr(ierr, "Error setting value");
    temp.assemble();
    
    this->saveVector(temp,groupname,valuename);
   
}

//////////////////////////
// Binary Viewer Wrapper//
//////////////////////////

PetscBinaryViewer::PetscBinaryViewer(const char* filename)
{
    PetscErrorCode ierr;
    ierr = PetscViewerBinaryOpen(PETSC_COMM_WORLD, filename, FILE_MODE_WRITE,&viewer); checkErr(ierr, "Error creating binary viewer");
}

void PetscBinaryViewer::saveMatrix(const PetscMatrix& input_matrix)
{
    PetscErrorCode ierr;
    ierr = MatView(input_matrix.matrix,viewer); checkErr(ierr, "Error viewing matrix");
}