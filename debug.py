import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import pickle

import numpy as np
import sys
import json

with open("debug/debug.json") as f:
    data = json.load(f)

if "BSPLINE" in sys.argv:
    bspline_data = np.loadtxt("build/debug/bsplines.txt")
    dbspline_data = np.loadtxt("build/debug/dbsplines.txt")

    real = bspline_data[:,0]
    imag = bspline_data[:,1]

    dreal = dbspline_data[:,0]
    dimag = dbspline_data[:,1]

    n_basis = data["bspline_data"]["n_basis"]
    Nr = data["grid_data"]["Nr"]
    grid_size = data["grid_data"]["grid_size"]
    r = np.linspace(0,grid_size,Nr)

    fig,(ax1,ax2) = plt.subplots(1,2, figsize=(10, 5))

    for i in range(n_basis):
        ax1.plot(r,real[i*Nr:(i+1)*Nr],color = "k")
        ax1.plot(r,imag[i*Nr:(i+1)*Nr],color = "brown")


    for i in range(n_basis):
        ax2.plot(dreal[i*Nr:(i+1)*Nr],color = "k")
        ax2.plot(dimag[i*Nr:(i+1)*Nr],color = "brown")

    fig.savefig("build/debug/bsplines.png")

if "LM" in sys.argv:
    lmax = data["angular_data"]["lmax"]
    lm_to_block_txt = np.loadtxt("debug/lm_to_block.txt")
    fig,ax = plt.subplots()
    space_size =lmax + 1
    space = np.zeros((space_size, 2 * lmax + 1))

    column1 = lm_to_block_txt[:,0]
    column2 = lm_to_block_txt[:,1]
    column3 = lm_to_block_txt[:,2]
    for i in range(len(column1)):
        space[lmax - int(column1[i]), int(column2[i]) + lmax] = 1

    ax.imshow(np.flipud(space), cmap='gray', interpolation='none', origin='lower')
    ax.set_xlabel('m')
    ax.set_ylabel('l')
    ax.set_xticks([i for i in range(0, 2 * lmax + 1, 10)])  # Positions for ticks
    ax.set_xticklabels([str(i - lmax) for i in range(0, 2 * lmax + 1, 10)])  # Labels from -lmax to lmax
    ax.set_title('Reachable (white) and Unreachable (black) Points in l-m Space')
    fig.savefig("debug/lm_space.png")

if "LASER" in sys.argv:
    laser_data = np.loadtxt("debug/laser.txt")
    t = laser_data[:,0]
    Ax = laser_data[:,1]
    Ay = laser_data[:,2]
    Az = laser_data[:,3]

    plt.plot(t,Ax,color = "k",label = "Ax")
    plt.plot(t,Ay,color = "brown",label = "Ay")
    plt.plot(t,Az,color = "blue",label = "Az")
    plt.legend()
    plt.savefig("debug/laser.png")

with open("partial_spectra.pkl","rb") as f:
    python_spectra = pickle.load(f)

l,m = 0,0
cpp_spectra = np.loadtxt(f"partial_{l}_{m}.txt")

python_spectrum = np.abs(python_spectra[(l,m)])**2
cpp_spectrum = cpp_spectra[:,1]**2 + cpp_spectra[:,2]**2
plt.semilogy(python_spectrum,label = "Python")
plt.semilogy(cpp_spectrum,label = "C++")
plt.legend()
plt.title(f"Partial Spectrum for l = {l} m = {m}")
plt.savefig("partial.png")
plt.clf()

data = np.loadtxt("pes.txt")
pes_cpp = data[:,1]/(2*np.pi)**3
pes_python = np.load("PES.npy")
plt.semilogy(pes_python,label = "Python")
plt.semilogy(pes_cpp,label = "C++")
plt.legend()
plt.savefig("pes.png")