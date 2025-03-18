import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LogNorm, Normalize


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
    plt.semilogy(e_cpp,pes_cpp,label = "C++")
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
    min_val = max_val*1e-3

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

    name_map = {}

    if True:
        name_map[(1,0)] = "1s"
        name_map[(2,0)] = "2s"
        name_map[(2,1)] = "2p"
        name_map[(3,0)] = "3s"
        name_map[(3,1)] = "3p"
        name_map[(3,2)] = "3d"
        name_map[(4,0)] = "4s"
        name_map[(4,1)] = "4p"
        name_map[(4,2)] = "4d"
        name_map[(4,3)] = "4f"
        name_map[(5,0)] = "5s"
        name_map[(5,1)] = "5p"
        name_map[(5,2)] = "5d"
        name_map[(5,3)] = "5f"
        name_map[(5,4)] = "5g"
        name_map[(6,0)] = "6s"
        name_map[(6,1)] = "6p"
        name_map[(6,2)] = "6d"
        name_map[(6,3)] = "6f"
        name_map[(6,4)] = "6g"
        name_map[(6,5)] = "6h"
        name_map[(7,0)] = "7s"
        name_map[(7,1)] = "7p"
        name_map[(7,2)] = "7d"
        name_map[(7,3)] = "7f"
        name_map[(7,4)] = "7g"
        name_map[(7,5)] = "7h"
        name_map[(7,6)] = "7i"
        name_map[(8,0)] = "8s"
        name_map[(8,1)] = "8p"
        name_map[(8,2)] = "8d"
        name_map[(8,3)] = "8f"
        name_map[(8,4)] = "8g"
        name_map[(8,5)] = "8h"
        name_map[(8,6)] = "8i"
        name_map[(8,7)] = "8j"
        name_map[(9,0)] = "9s"
        name_map[(9,1)] = "9p"
        name_map[(9,2)] = "9d"
        name_map[(9,3)] = "9f"
        name_map[(9,4)] = "9g"
        name_map[(9,5)] = "9h"
        name_map[(9,6)] = "9i"
        name_map[(9,7)] = "9j"
        name_map[(9,8)] = "9k"
        name_map[(10,0)] = "10s"
        name_map[(10,1)] = "10p"
        name_map[(10,2)] = "10d"
        name_map[(10,3)] = "10f"
        name_map[(10,4)] = "10g"
        name_map[(10,5)] = "10h"
        name_map[(10,6)] = "10i"
        name_map[(10,7)] = "10j"
        name_map[(10,8)] = "10k"
        name_map[(10,9)] = "10l"

    bound_pops = np.loadtxt("BOUND_files/bound_pops.txt")

    rows,cols = bound_pops.shape

    values = []
    names = []

    for i in range(rows):
        n = bound_pops[i,0]
        l = bound_pops[i,1]
        pop = bound_pops[i,2]

        print(n,l,pop)

        if (n,l) in name_map:
            name = name_map[(n,l)]

            values.append(pop)
            names.append(name)
        
    ####
    x = np.arange(len(names))
    width = 0.2  # Adjust width to create more space between bars

    fig, ax = plt.subplots(figsize=(10, 8))

    # Create bar plot
    bars = ax.bar(x, values, width=width, color='skyblue', edgecolor='black')

    # Set y-axis to log scale
    ax.set_yscale('log')

    # Set the x-ticks with category labels and rotate them for clarity.
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=8)


    # Labels and title
    ax.set_ylabel('Value (log scale)')
    ax.set_title('Bar Plot with Log Scale and Spaced Labels')

    # Adjust layout so labels don't get cut off
    plt.tight_layout()
    plt.savefig("images/bound_pops.png")



