#include <petscvec.h>

class PetscVector 
{
    public:
        PetscVector() = default;
        ~PetscVector();

        Vec& getVector();
    private:
        Vec vector;
};