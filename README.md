#AMO-TDSE

This repository contains classes and namespaces used to solve both the Time-Independent Schrödinger Equation (TISE) and the Time-Dependent Schrödinger Equation (TDSE) for an atomic systems.

The TISE and TDSE not only work for simulating Hydrogen, but are designed to work with Hydrogen as well as Single Active Electron (SAE) Potentials in order to accurately approximate energy eigenstates and electron dynamics of more complex atoms. 

The TISE is capable of computing the radial component of all energy eigenstates contained within the spectrum of the discretized hamiltonian. Due to the discrete nature of the hamiltonian, the continuum states (E > 0) are less useful that the bound states (E < 0) when computing this way, so those are usually suppressed in the input file to save time/space. 

The TDSE is solved in the context of an atom interacting with an ultrashort, intense laser pulse. In AMO physics, lasers of this type are frequently used to study phenomena that occur on short timescales such as electron dynamics. Two key phenomena that this code is designed to simulation are Microscopic High Harmonic Generation (HHG), and Photoelectron Spectra

**High Harmonic Generation (HHG):**

From classical electromagnetism,it is known that accelerating charges emit radiation. Although electrons in an atom are described by quantum wavefunctions, the underlying principle still holds. When a laser passes by an atom, the probability distribution of the electron oscillates, and this oscillation causes radiation to be emitted. This code allows for the expectation value of acceleration to be computed at each time step. A Fourier Transform of which reveals the emitted frequencies of light. 

**Photoelectron Spectra:**

When the atom interacts with the laser pulse, it can absorb energy in discrete quanta (photons). With sufficient energy absorption, the electron transitions from a bound state to an ionized, continuum state.

Photoelectron spectra are analyzed in two ways:
1. **Angle-integrated:** Determines how much of the wavefunction ionizes at a given energy.
2. **Angle-resolved:** Determines how much of the wavefunction ionizes at a given energy, in a given direction.


Important features of these phenomena are reproduced, such as the classical HHG cutoff, and Above-Threshold Ionization (ATI) peaks in the photoelectron spectra.

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
