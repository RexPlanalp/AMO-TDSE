import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize
from matplotlib import cm


import os
import numpy as np
import pickle
import sys
import json

with open("input.json") as f:
    data = json.load(f)

if not os.path.exists("images"):
    os.makedirs("images")

if "BLOCK" in sys.argv:
    lmax = data["angular"]["lmax"]
    lm_to_block_txt = np.loadtxt("BLOCK_files/lm_to_block.txt")
    probabilities_txt = np.loadtxt("BLOCK_files/block_norms.txt")
    fig,ax = plt.subplots(figsize=(10, 8))
    space_size =lmax + 1
    space = np.zeros((space_size, 2 * lmax + 1))

    column1 = lm_to_block_txt[:,0]
    column2 = lm_to_block_txt[:,1]
    column3 = lm_to_block_txt[:,2]
    column4 = probabilities_txt[:,0]
    column5 = probabilities_txt[:,1]
    for i in range(len(column1)):
        space[lmax - int(column1[i]), int(column2[i]) + lmax] = column4[i]

    space[space==0] = np.min(space[space!=0])
    cax = ax.imshow(space, cmap='inferno', interpolation='nearest', norm=Normalize())  
    ax.set_xlabel('m')
    ax.set_ylabel('l')
    ax.set_xticks([0, lmax, 2 * lmax])  
    ax.set_xticklabels([-lmax, 0, lmax])
    ax.set_yticks([0, lmax])  
    ax.set_yticklabels([lmax, 0])  
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False) 

    fig.colorbar(cax, ax=ax, shrink=0.5)
    ax.set_title('Heatmap of Probabilities for l and m Values')
    fig.savefig("images/pyramid.png")
    fig.clf()

if "PES" in sys.argv:
    data = np.loadtxt("PES_files/pes.txt")
    pes_cpp = data[:,1]
    e_cpp  = data[:,0]
    plt.semilogy(e_cpp,pes_cpp,label = "PES")
    plt.legend()
    plt.savefig("images/pes.png")
    # os.system("mv pes.png ~/Research/TDSE_PETSC/")
    os.system("code images/pes.png")



    pad_data = np.loadtxt("PES_files/pad.txt")
    pad_e = np.array(pad_data[:,0])
    pad_k = np.sqrt(2*pad_e)
    pad_theta = np.array(pad_data[:,1])
    pad_phi = np.array(pad_data[:,2])
    pad_p = np.array(pad_data[:,3])

    kx = pad_k*np.sin(pad_theta)*np.cos(pad_phi)
    ky = pad_k*np.sin(pad_theta)*np.sin(pad_phi)
    kz = pad_k*np.cos(pad_theta)

    max_val = np.max(pad_p)
    min_val = max_val*1e-6

    cmap = "hot_r"

    fig,ax = plt.subplots()

    norm = mcolors.LogNorm(vmin=min_val,vmax=max_val)
    #norm = mcolors.Normalize(vmin=min_val,vmax=max_val)

    sc = ax.scatter(kx,kz,c=pad_p,norm=norm,cmap=cmap)
    #sc = ax.scatter(kx,ky,c=pad_p,norm=norm,cmap=cmap)
    ax.set_aspect("equal",adjustable = "box")
    fig.colorbar(sc,ax=ax)
    fig.savefig("images/pad.png")
    #os.system("mv pad.png ~/Research/TDSE_PETSC/")
    os.system("code images/pad.png")

if "BOUND" in sys.argv:

    nmax = data["angular"]["nmax"]

    bound_pops = np.loadtxt("BOUND_files/bound_pops.txt")

    rows,cols = bound_pops.shape

    pop_matrix = np.full((nmax+1, nmax+1), np.nan)

    for i in range(rows):
        n = int(bound_pops[i,0])
        l = int(bound_pops[i,1])
        pop = bound_pops[i,2]

        print(n,l)

        pop_matrix[n,l] = pop
    for n in range(nmax+1):
        for l in range(n):  # Only allow l < n
            if np.isnan(pop_matrix[n, l]):  # Ensure only invalid positions are masked
                pop_matrix[n, l] = None  
    fig, ax = plt.subplots(figsize=(8, 6))
    cmap = cm.viridis
    cmap.set_bad(color='white')  # Make masked values white

    im = ax.imshow(pop_matrix, cmap=cmap, origin="lower", norm=LogNorm(), aspect='auto')

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label("Population")

    # Axis labels and formatting
    ax.set_xlabel("l (Orbital Quantum Number)")
    ax.set_ylabel("n (Principle Angular Momentum)")
    ax.set_title("Population Heatmap (Bound States)")
    ax.set_xticks(range(nmax+1))
    ax.set_yticks(range(nmax+1))

    plt.savefig("images/bound_pops.png")

if "HHG" in sys.argv:

    polarization = data["laser"]["polarization"]

    I_max = data["laser"]["I"]/3.51E16
    w = data["laser"]["w"]
    N = data["grid"]["N"]

    if data["species"] == "H":
        Ip = -0.5
    elif data["species"] == "Ar":
        Ip = -0.5791546178
    elif data["species"] == "He":
        Ip = -0.9443

    Up = I_max/(4*w**2)

    cut_off = 3.17*Up - Ip


    hhg_output = np.loadtxt("TDSE_files/hhg_data.txt")



    # Load data
    HHG_data = hhg_output[:, [1, 3, 5]].T
    laser_data = hhg_output[:, [2,4,6]].T
    total_time = hhg_output[:, 0]

    # Initialize dipole_acceleration as a 2D array to store each component
    dipole_acceleration = np.zeros((3, len(total_time)))

    # Calculate dipole acceleration for each component
    for i in range(3):
        dipole_acceleration[i, :] = -HHG_data[i, :] + np.gradient(laser_data[i, :], total_time)

    # Plot the dipole acceleration components (optional)
    plt.figure(figsize=(8, 6))
    for i in range(3):
        plt.plot(total_time, dipole_acceleration[i, :], label=f'Component {i+1}')
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Dipole Acceleration (a.u.)')
    plt.title('Dipole Acceleration Components')
    plt.legend()
    plt.savefig("images/dipole_accel.png")
    plt.clf()

    # Apply a window function (Blackman window) to each component
    window = np.blackman(len(total_time))
    windowed_dipole = dipole_acceleration * window  # Broadcasting applies the window to each row

    # Perform FFT on each component
    dipole_fft = np.fft.fft(windowed_dipole, axis=1)
    frequencies = np.fft.fftfreq(len(total_time), d=total_time[1] - total_time[0])

    # Compute the power spectrum (magnitude squared of FFT) for each component
    power_spectrum = np.abs(dipole_fft)**2

    # Sum the power spectra of all components to get the total power spectrum
    total_power_spectrum = np.sum(power_spectrum, axis=0)

    # Only keep the positive frequencies
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    total_power_spectrum = total_power_spectrum[positive_freq_idx]

    # Plot the total harmonic spectrum
    plt.figure(figsize=(8, 6))
    plt.semilogy(frequencies * 2*np.pi / w, total_power_spectrum, color='b')
    plt.axvline(cut_off/w, color='r', linestyle='--', label='Cut-off Energy')
    plt.xlim([0, 60])        # Adjust based on your data's frequency range
    plt.ylim([1e-4, 1e4])     # Adjust based on the power spectrum's range
    plt.xlabel('Frequency (atomic units)')
    plt.ylabel('Intensity (arb. units)')
    plt.title('Harmonic Spectrum')
    plt.grid(True, which='both', ls='--', lw=0.5)
    plt.legend()
    plt.savefig("images/harmonic_spectrum.png")
    plt.clf()

    print("Dipole acceleration components and harmonic spectrum have been successfully processed and saved.")     
        



