#pragma once

#include <petscsys.h>
#include <iostream>
#include <stdexcept>


void checkError(PetscErrorCode ierr, const char* msg = "");