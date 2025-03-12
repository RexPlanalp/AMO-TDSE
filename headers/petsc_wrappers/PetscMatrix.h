#pragma once

#include <petscmat.h>
#include "simulation.h"
#include <functional>

//////////////////////////
// Petsc Matrix Wrapper //
//////////////////////////



enum class ECSMode
{
    ON,
    OFF
};

enum class MatrixType
{
    SEQUENTIAL,
    PARALLEL
};


class PetscMatrix
{
    public:

        // Default Constructor
        PetscMatrix() = default;

        // Copy Constructor
        PetscMatrix(const PetscMatrix& other);

        // Copy assignment operator
        PetscMatrix& PetscMatrix::operator=(const PetscMatrix& other);
        
        // Destructor
        ~PetscMatrix();

        // Assemble the matrix
        void assemble();

        // Internal matrix
        Mat matrix = nullptr;
};

//////////////////////////
//    Radial Subclass   //
//////////////////////////

using radialElement = std::function<std::complex<double>(int,int,std::complex<double>,int,const std::vector<std::complex<double>>&)>;


class RadialMatrix : public PetscMatrix
{
    public:

    // Explicit Constructor
    RadialMatrix(const simulation& sim, MatrixType type);

    // Bind matrix element
    void bindElement(radialElement input_element);

    // Populate the matrix
    void populateMatrix(const simulation& sim, ECSMode ecs);

    // locally owned rows
    int local_start, local_end;

    // Matrix element function
    radialElement element;
};  













