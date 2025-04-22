# AMO-TDSE

## What is it?

**AMO-TDSE is a project written in C++ designed to solve both the Time-Indepdendent Schrödinger Equation (TISE) and the Time-Dependent Schrödinger Equation (TDSE) for atomic systems interacting with ultrashort, intensie, laser pulses**

## Why? 

**This project started as a barebones iteration on previous software I was using for my research in Ulrafast AMO Physics at JILA. Since then I've improved upon it and cleaned it up as I learned better coding practices**

## How does it work?

**The code has two main branches corresponding to the TISE and TDSE respectively, each of which are controlled by an input JSON file. The code uses Spherical Harmonic Expansions for the angular coordinates, as well as a B-Spline expnsion for the radial coordinate to construct hamiltonians as matrices. Due to the large dimensionality and sparsity of these matrices, the Portable, Extensible Toolkit for Scientific Computation (PETSc) is used both for its convenient storage formats, but also to take advantage of MPI parallelism to speed up the linear algebra**

**Specifically, Scalable Library for Eigenvalue Problem Computations (SLEPc) is used to setup and solve a generalized eigenvalue problem to find eigenstates/eigenvalues during the TISE, while the PETSc implementation of the Generalized Minimal Residual Method (GMRES) is used for solving the linear system used in Crank-Nicolson Time Propgation for the TDSE.**

## What does it produce? 

**The TISE branch of the code outputs the eigenstates and eigenvalues of the atomic hamiltonian to an HDF5 file. The number of eigenstates/eigenvalues to request is controlled via the associated entry in the input file. The code also saves various matrices to binary that are necessary to run the TDSE.**

**The TDSE branch of the code takes one of the eigenstates output by the TISE and propagates the state through time, under the influence of an laser pulse. This final state is them output to an HDF5 file. This final state can then be further processed to evaluate various phenomena of electron dynamics that occured during the laser pulse. Examples are given below:**


### High Harmonic Generation (HHG):

From classical electromagnetism,it is known that accelerating charges emit radiation. Although electrons in an atom are described by quantum wavefunctions, the underlying principle still holds. When a laser passes by an atom, the probability distribution of the electron oscillates, and this oscillation causes radiation to be emitted. This code allows for the expectation value of acceleration to be computed at each time step. A Fourier Transform of which reveals the emitted frequencies of light. 

### Photoelectron Spectra:

When the atom interacts with the laser pulse, it can absorb energy in discrete quanta (photons). With sufficient energy absorption, the electron transitions from a bound state to an ionized, continuum state.

Photoelectron spectra are analyzed in two ways:
**Angle-integrated:** Determines how much of the wavefunction ionizes at a given energy.
**Angle-resolved:** Determines how much of the wavefunction ionizes at a given energy, in a given direction.

**Important features of these phenomena are reproduced, such as the classical HHG cutoff, and Above-Threshold Ionization (ATI) peaks in the photoelectron spectra.**

**Techniques Employed:**

**Spherical Harmonic Expansion**

As a 3D object the wavefunction can be written as a function of spherical coordinates. Expanding the angular coordinates in complex spherical harmonics not only allows
for a reduction in the required dimensionality to converge, but also provides an accurate mathematical representation of "selection rules" which determines how angular
momentum can be absorbed/emitted. 


**B-Spline Basis Functions:**

While expanding on a grid for the radial coordinate presents fewer challenges that for the angular coordinates, a B-Spline basis was chosen for a few key reasons. First, B-Splines are able to capture more structure of the wavefunctions than a single grid point. This allows for a significant reduction in dimensionality at the cost of sightly decreased sparsity in the hamiltonian. 

The second key reason is that the spacing between B-Spline basis functions are easily controlled via the knot vector. This allows for the density of B-Splines in particular regions to be fine-tuned as necessary. For example, when working with SAE potentials for atoms such as Argon, the potential is so deep near r = 0 that the bound states oscillate very quickly. By putting many B-Splines near the core, the wavefunction can be more accurately represented in this region. Adjusting the density of points with a finite-difference grid is far more challenging. 

**Exterior Complex Scaling:**

One of the problems with solving the TDSE numerically is that we cannot use an infinitely large box. Since we have parts of the wavefunction becoming ionized
and flying away from the core, these parts of the wavefunction may hit that boundary. To avoid these unphysical reflections there are many techniques 
employed to absorb the wavefunction near the boundary. Some such techniques include Masking Functions, Complex Potentials, and Exterior Complex Scaling. 
Among these choices Exterior Complex Scaling is the most difficult to implement in practice, but provides the most robust and consistent absorption when
compared to the other methods. 

Exterior Complex Scaling works by rotating part of our grid into the complex plane after some cutoff radius. This rotation effectively takes outgoing momentum com




To achieve these results, the following techniques were implemented:
1. **B-Spline Basis Functions:** For adaptive grid spacing and dimensionality reduction.
2. **Exterior Complex Scaling:** To establish a perfect absorbing boundary condition at the simulation box’s edge.
3. **Crank-Nicolson Time Propagation:** For stability and high-order accuracy during time evolution.

It should be noted that this project takes advantage of the PETSc and SLEPc parallel numerical libraries, 
and as such this code has been tested and designed to be used on large computing clusters. 

## Installation & Prerequisites

### Dependencies
- **CMake:** Version 3.14 or higher
- **Spack:** Version 1.0.0.dev0 or compatible
- **PETSc:** Version 3.22.2 or higher
- **SLEPc:** Version 3.22.2 or higher
- **Nlohmann Json:** For JSON parsing

### Installation Steps

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/RexPlanalp/TDSE-Solver-for-Laser-Atomic-Interactionsgit
   cd tdse-project
