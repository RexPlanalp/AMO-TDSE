#pragma once

#include <petscviewer.h>
#include "matrix.h"
#include "vector.h"

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

        void saveMatrix(PetscMatrix& matrix);
};

class PetscSaverHDF5 : public PetscSaver
{
    public: 
        PetscSaverHDF5(const char* filename);

        void saveVector(PetscVector& vector, const char* groupname,const char* vectorname);


        void saveValue(std::complex<double> value, const char* groupname, const char* valuename);
};