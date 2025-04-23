# AMO-TDSE

## What is it?

**AMO-TDSE is a project written in C++ designed to solve both the Time-Indepdendent Schrödinger Equation (TISE) and the Time-Dependent Schrödinger Equation (TDSE) for atomic systems interacting with ultrashort, intensie, laser pulses.**

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

## Techniques Used

### Exterior Complex Scaling

#### What is it?

**Exterior Complex Scaling is a technique for absorbing parts of the wavefunction that get too close to the boundary box of the simulation.**

#### Why?

**There are various techniques for achieving this result. One of which are Masking Functions, which at each time step multiply the wavefunction 
in a region (in this case near the boundary) by a dampening term. Another technique is introducing a Complex Potential which causes the wavefunction
to naturally decay after some cutoff similar to Exterior Scaling. While each of those other methods are used in literature, Exterior Complex Scaling is
unique in that it does not produce any artifacts that can leave an imprint on the final result, and it adapts its absorption to ensure even fast moving 
wavepackets can't reach the boundary. All of this comes at the cost of it being more difficult to implement.**

### B-Spline Basis functions

#### What is it? 

**B-Spline basis functions are a set of non-orthonormal functions that the radial component of the wavefunction are projected onto. This is in contrast to 
finite-difference approaches used in literature which effectively expand in Dirac delta functions.**

#### Why?

**B-Spline basis functions are able to capture more structure of the radial wavefunction when compared to a Dirac delta, and in practice this means less of 
them are needed to accurately represent the wavefunction, reducing dimensionality. They also have the desirable property of being extensively tunable, being able to adjust their density in certain regions by modifying the knot vector. Compared to other basis function expansions, they are also piecewise polynomial which makes computing matrix elements fast (although not as fast as a Dirac delta function basis, since those are analytic). **

### Crank-Nicolson Time Propagation

#### What is it?

**Crank-Nicolson Time Propagation is a time propagation scheme that takes the wavefunction at some earlier time, and relates it to the wavefunction at
a later point in time which are seeking.**

#### Why?

**There are other time propagation schemes such as the Forward Euler Approach. However the advanatges of Crank-Nicolson over a much simpler approach are numerous. First, Forward Euler is accurate to first order in time, while Crank-Nicolson is accurate to second order in time. In practice this means that the amount of time we evolve by (dt) at each step can be much larger for Crank-Nicolson Propagation, reducing the amount of linear solves we have to do. In addition, Forward Euler is not a unitary transofmration, so the norm of our wavefunction will not be preserved after each time step. This is important because in quantum mechanics time evolution inherently unitary. Without this we would need to renormalize the wavefunction at each step which costs resources, and we would lose information about how much of our 
wavefunction has reached the absorbing boundary.**

**Other techniues that avoid these pitfalls also exist such as the Split-Step Operator method, and I am in the process of investigating how viable this propagation scheme is for this particular system.**

## Installation & Prerequisites

### Dependencies
- **CMake:** Version 3.14 
- **PETSc:** Version 3.22.2 
- **SLEPc:** Version 3.22.2
- **GSL** Version 2.8
- **Nlohmann Json:** 

## Input File Guide

**Below is a brief description on what each part of the input file controls. If a value is to be given in atomic units or SI units it will have an (au) or (SI) next to it respectively.**

### "species" - Determines which atom to solve the TISE for. Currently Supports Hydrogen and Argon via SAE potentials. 

### "grid"

#### "grid_size" - Sets the size of the spherical box to simulation on (au)

#### "grid_spacing" - Sets the distance between Dirac Delta Functions where finite difference is necessary, only used in analysis after TISE and TDSE. (au)

#### "N" - Controls duration of the laser pulse by specifying the number of periods of the central frequency (see laser) 

#### "time_spacing" - Sets the amount of time to evolve over during each time step (au)

### "bsplines"

#### "n_basis" - Sets the number of bsplines basis functions to expand the radial wavefunction in.

#### "order" - Sets the order of the bspline basis functions

#### "R0_input" - Sets the desired distance for ECS to turn on (au)

#### "eta" - Sets how much ECS will rotate the basis into the complex plane

#### "debug" - Toggles whether the bspline basis functions are fully evaluated and dumped to a file. Simulation wide debug must also be enabled (see debug).

### "angular"

#### "lmax" - Sets the largest l value to expand up to in Spherical Harmonics

#### "mmin" - Sets the smallest m value to expand up to in Spherical Harmonics

#### "mmax" - Sets the largest m value to expand up to in Spherical Harmonics

#### "nmax" - Sets the largest energy state to solve for during the TISE

### "laser" 

#### "w" - Sets the central frequency of the laser pulse (au)

#### "I" - Sets the intensity of the laser pulse in W/cm^2 (SI)

#### "polarization" - Specifies polarization vector for laser pulse. Will be internally normalized. 

#### "poynting" - Specifies poynting vector for laser pulse. Will be internally normalized. 

#### "ell" - Sets ellipticity of laser pulse, 1 being circular and 0 being linear. 

#### "CEP" - Sets carrier envelope phase between the carrier wave and the envelope function

### "TISE" 

#### "tolerance" - Sets the tolerance for which to solve the eigenvalue problem

#### "max_iter" - Sets the maximum number of iterations to converge an eigenvalue/vector pair

### "TDSE"

#### "tolerance" - Sets the tolerance for the linear solver during time propgation

#### "state" - Sets which state from the TISE will be used as the initial state during time propagation

### "observables" 

#### "E" - Sets the range of energy values to evaluate the photoelectron spectrum over

#### "SLICE" - Sets which plane to compute and plot the angularly resolved photoelectron spectrum over

#### "HHG" - Toggles whether HHG data will be computed during time propagation

#### "CONT" - Toggles whether the final distribution of the wavefunction will be plotted with or without the bound state contributions projected out

### "debug" - Toggles whether debug info will be dumped during runtimeto help find bugs or issues


