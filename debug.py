import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

import numpy as np
import sys
import json

# with open("build/debug/debug.json") as f:
#     data = json.load(f)

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
    lmax = data["angular"]["lmax"]
    lm_to_block_txt = np.loadtxt("build/debug/lm_to_block.txt")
    fig,ax = plt.subplots()
    space_size =lmax + 1
    space = np.zeros((space_size, 2 * lmax + 1))

    column1 = lm_to_block_txt[:,0]
    column2 = lm_to_block_txt[:,1]
    column3 = lm_to_block_txt[:,2]
    for i in range(len(column1)):
        space[lmax - column1[i], column2[i] + lmax] = 1

    ax.imshow(np.flipud(space), cmap='gray', interpolation='none', origin='lower')
    ax.set_xlabel('m')
    ax.set_ylabel('l')
    ax.set_xticks([i for i in range(0, 2 * lmax + 1, 10)])  # Positions for ticks
    ax.set_xticklabels([str(i - lmax) for i in range(0, 2 * lmax + 1, 10)])  # Labels from -lmax to lmax
    ax.set_title('Reachable (white) and Unreachable (black) Points in l-m Space')
    fig.savefig("build/debug/lm_space.png")



# laser_data = np.loadtxt("build/laser.txt")
# t = laser_data[:,0]
# Ax = laser_data[:,1]
# Ay = laser_data[:,2]
# Az = laser_data[:,3]

# plt.plot(t,Ax,color = "k",label = "Ax")
# plt.plot(t,Ay,color = "brown",label = "Ay")
# plt.plot(t,Az,color = "blue",label = "Az")
# plt.legend()
# plt.savefig("build/laser.png")

starting_val =  -0.017737546244461-0.445338526671213j
total_time = (2*np.pi/0.0850000132513)*20
phase = np.exp(-1j*(-0.5*total_time))
ending_val = starting_val*phase
print(ending_val)

# w = 0.085
# N = 20

# time_size = (2*np.pi/w)*N
# t = np.linspace(0,time_size,100000)

# envelope = np.sin(w*t/(2*N))

# t2_int = np.trapz(envelope*t**2,t)
# int = np.trapz(envelope,t)

# alpha = (0.085/w) * np.sqrt(t2_int/int)
# factor = (1+np.sqrt(1+4/(np.pi**2 * alpha**2)))/2
# w_test = w*factor

# print(w_test)