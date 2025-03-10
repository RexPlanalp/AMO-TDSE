#pragma once

#include <petscviewer.h>
#include "matrix.h"

class PetscSaver
{

    public:
        PetscSaver() = default;
        ~PetscSaver();

        PetscViewer& getViewer();
    
    private:
        PetscViewer viewer;
};

class PetscSaverBinary : public PetscSaver
{
    public:
        PetscSaverBinary(const char* filename);
        ~PetscSaverBinary();

        void saveMatrix(PetscMatrix& matrix);
};