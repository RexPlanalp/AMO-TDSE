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