import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize

import numpy as np
import sys
import json

with open("debug/debug.json") as f:
    data = json.load(f)

if "BLOCK" in sys.argv:
    lmax = data["angular_data"]["lmax"]
    lm_to_block_txt = np.loadtxt("debug/lm_to_block.txt")
    probabilities_txt = np.loadtxt("TDSE_files/block_norms.txt")
    fig,ax = plt.subplots()
    space_size =lmax + 1
    space = np.zeros((space_size, 2 * lmax + 1))

    column1 = lm_to_block_txt[:,0]
    column2 = lm_to_block_txt[:,1]
    column3 = lm_to_block_txt[:,2]
    column4 = probabilities_txt[:,0]
    column5 = probabilities_txt[:,1]
    for i in range(len(column1)):
        print("Block Norm:", i, column4[i])
        space[lmax - int(column1[i]), int(column2[i]) + lmax] = column4[i]

    space[space==0] = np.min(space[space!=0])
    cax =ax.imshow(np.flipud(space), cmap='inferno', interpolation='nearest', origin='lower',norm = LogNorm())
    ax.set_xlabel('m')
    ax.set_ylabel('l')
    ax.set_xticks([i for i in range(0, 2 * lmax + 1, 10)])  # Positions for ticks
    ax.set_xticklabels([str(i - lmax) for i in range(0, 2 * lmax + 1, 10)])  # Labels from -lmax to lmax
    ax.set_title('Reachable (white) and Unreachable (black) Points in l-m Space')

    fig.colorbar(cax, ax=ax, shrink=0.5)
    fig.savefig("block.png")
