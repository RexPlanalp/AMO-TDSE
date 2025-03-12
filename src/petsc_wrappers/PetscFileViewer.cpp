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