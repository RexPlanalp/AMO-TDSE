import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import os

data = np.loadtxt("PES_files/pes.txt")
pes_cpp = data[:,1]
e_cpp  = data[:,0]
plt.semilogy(e_cpp,pes_cpp,label = "C++")
plt.legend()
plt.savefig("pes.png")
# os.system("mv pes.png ~/Research/TDSE_PETSC/")
os.system("code pes.png")



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
fig.savefig("pad.png")
#os.system("mv pad.png ~/Research/TDSE_PETSC/")
os.system("code pad.png")


