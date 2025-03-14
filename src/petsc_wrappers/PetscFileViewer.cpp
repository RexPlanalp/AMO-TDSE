#include <petscviewer.h>
#include <petscviewerhdf5.h>
#include "petsc_wrappers/PetscFileViewer.h"
#include "utility.h"
#include "mpi.h"
#include "simulation.h"



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


PetscHDF5Viewer::PetscHDF5Viewer(const char* filename, RunMode run, OpenMode mode)
{
    PetscErrorCode ierr;

    switch(run)
    {
        case RunMode::SEQUENTIAL:
            comm = PETSC_COMM_SELF;
            switch(mode)
            {
                case OpenMode::READ:
                    ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer); checkErr(ierr, "Error creating HDF5 viewer");
                    break;
                case OpenMode::WRITE:
                    ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_WRITE, &viewer); checkErr(ierr, "Error creating HDF5 viewer");
                    break;
            }
            break;
        case RunMode::PARALLEL:
            comm = PETSC_COMM_WORLD;
            switch(mode)
            {
                case OpenMode::READ:
                    ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_READ, &viewer); checkErr(ierr, "Error creating HDF5 viewer");
                    break;
                case OpenMode::WRITE:
                    ierr = PetscViewerHDF5Open(comm, filename, FILE_MODE_WRITE, &viewer); checkErr(ierr, "Error creating HDF5 viewer");
                    break;
            }
            break;
    }
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

 
    ierr = VecCreate(comm, &temp.vector); checkErr(ierr, "Error creating vector");
    ierr = VecSetSizes(temp.vector,PETSC_DECIDE,1); checkErr(ierr, "Error setting vector size");
    ierr = VecSetFromOptions(temp.vector); checkErr(ierr, "Error setting vector options");
    ierr=  VecSetValue(temp.vector,0,value,INSERT_VALUES); checkErr(ierr, "Error setting value");
    temp.assemble();
    
    this->saveVector(temp,groupname,valuename);
   
}

//////////////////////////
// Binary Viewer Wrapper//
//////////////////////////

PetscBinaryViewer::PetscBinaryViewer(const char* filename, RunMode run, OpenMode mode)
{
    PetscErrorCode ierr;

    switch(run)
    {
        case RunMode::SEQUENTIAL:
            comm = PETSC_COMM_SELF;
            switch(mode)
            {
                case OpenMode::READ:
                    ierr = PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &viewer); checkErr(ierr, "Error creating HDF5 viewer");
                    break;
                case OpenMode::WRITE:
                    ierr = PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer); checkErr(ierr, "Error creating HDF5 viewer");
                    break;
            }
            break;
        case RunMode::PARALLEL:
            comm = PETSC_COMM_WORLD;
            switch(mode)
            {
                case OpenMode::READ:
                    ierr = PetscViewerBinaryOpen(comm, filename, FILE_MODE_READ, &viewer); checkErr(ierr, "Error creating HDF5 viewer");
                    break;
                case OpenMode::WRITE:
                    ierr = PetscViewerBinaryOpen(comm, filename, FILE_MODE_WRITE, &viewer); checkErr(ierr, "Error creating HDF5 viewer");
                    break;
            }
            break;
    }
}

void PetscBinaryViewer::saveMatrix(const PetscMatrix& input_matrix)
{
    PetscErrorCode ierr;
    ierr = MatView(input_matrix.matrix,viewer); checkErr(ierr, "Error viewing matrix");
}

PetscMatrix PetscBinaryViewer::loadMatrix()
{
    PetscErrorCode ierr;
    PetscMatrix M;
    ierr = MatCreate(comm,&M.matrix); checkErr(ierr, "Error creating matrix");
    
    if (comm = PETSC_COMM_SELF)
    {
        ierr = MatSetType(M.matrix,MATSEQAIJ); checkErr(ierr, "Error setting matrix type");
    }
    else
    {
        ierr = MatSetType(M.matrix,MATMPIAIJ); checkErr(ierr, "Error setting matrix type");
    }

    ierr = MatLoad(M.matrix,viewer); checkErr(ierr, "Error loading matrix");
    return M; 
}