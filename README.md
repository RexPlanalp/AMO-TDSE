This repository contains classes and namespaces used to solve both the time-independent and time-dependent Schrödinger Equations for an atom.

Specifically, the TDSE is solved in the context of an atom interacting with an ultrashort, intense laser pulse. In AMO physics, lasers of this type are frequently used to study phenomena that occur on ultrashort timescales. Two key phenomena that this code simulates are High Harmonic Generation (HHG) and Photoelectron Spectra. Simple descriptions of these phenomena are provided below.

**High Harmonic Generation (HHG):**

From classical electromagnetism, accelerating charges emit radiation. Although electrons in an atom are described by quantum wavefunctions, the underlying principle still holds. When a laser passes by an atom, the probability distribution of the electron oscillates (“jiggles”), and this oscillation causes radiation to be emitted. This code accounts for the dynamics of this oscillation during time propagation, allowing a Fourier Transform of the “jiggle” signal to reveal the emitted frequencies.

**Photoelectron Spectra:**

When the atom interacts with the laser pulse, it can absorb energy in discrete quanta (photons). With sufficient energy absorption, the electron transitions from a bound state to an ionized, free state.

Photoelectron spectra are analyzed in two ways:
1. **Angle-integrated:** Determines how much of the wavefunction ionizes at a given energy.
2. **Angle-resolved:** Provides additional information about the direction in which the electrons are emitted.

This code reproduces important features such as the classical HHG cutoff and above-threshold ionization (ATI) peaks in the photoelectron spectrum.

**Techniques Employed:**

To achieve these results, the following techniques were implemented:
1. **B-Spline Basis Functions:** For adaptive grid spacing and dimensionality reduction.
2. **Exterior Complex Scaling:** To establish a perfect absorbing boundary condition at the simulation box’s edge.
3. **Crank-Nicolson Time Propagation:** For stability and high-order accuracy during time evolution.

**Technologies Used:**

- **CMake:** Currently tested with Version 3.14.
- **Spack:** Currently tested with Version 1.0.0.dev0.
- **PETSc:** Currently tested with Version 3.22.2.
- **SLEPc:** Currently tested with Version 3.22.2.
- **Nlohmann Json:** For JSON parsing.
