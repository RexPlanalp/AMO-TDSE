#include "viewer.h"
#include "matrix.h"

#include <petscviewer.h>


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
    PetscViewerBinaryOpen(PETSC_COMM_WORLD,filename,FILE_MODE_WRITE,&getViewer());
}

// Method: save the matrix
void PetscSaverBinary::saveMatrix(PetscMatrix& matrix)
{
    MatView(matrix.getMatrix(),getViewer());
}

