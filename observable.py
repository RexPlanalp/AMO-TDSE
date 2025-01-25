import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
import os

import numpy as np
import sys
import json

with open("debug/debug.json") as f:
    data = json.load(f)

if not os.path.exists("images"):
    os.makedirs("images")

if "BLOCK" in sys.argv:
    lmax = data["angular_data"]["lmax"]
    lm_to_block_txt = np.loadtxt("debug/lm_to_block.txt")
    probabilities_txt = np.loadtxt("TDSE_files/block_norms.txt")
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
    cax = ax.imshow(space, cmap='inferno', interpolation='nearest', norm=LogNorm())  
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

