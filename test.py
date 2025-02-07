import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import pickle
import os


# coul_wave = np.loadtxt("coulomb.txt")
# x = coul_wave[:,0]
# y = coul_wave[:,1]
# plt.plot(x,y)
# plt.xlim([1990,2000])
# print(y[-1])
# plt.title("Coulomb Wave")
# plt.savefig("coulomb.png")
# plt.clf()


# with open("partial_spectra.pkl","rb") as f:
#     python_spectra = pickle.load(f)

# l,m = 0,0
# cpp_spectra = np.loadtxt(f"partial_{l}_{m}.txt")


# python_spectrum = np.abs(python_spectra[(l,m)])**2
# cpp_spectrum = cpp_spectra[:,1]**2 + cpp_spectra[:,2]**2
# plt.semilogy(python_spectrum,label = "Python")
# plt.semilogy(cpp_spectrum,label = "C++")
# plt.legend()
# plt.title(f"Partial Spectrum for l = {l} m = {m}")
# plt.savefig("partial.png")
# plt.clf()

# data = np.loadtxt("pes.txt")
# pes_cpp = data[:,1]/(2*np.pi)**3
# e_cpp  = data[:,0]
# # pes_python = np.load("PES.npy")
# # plt.semilogy(pes_python,label = "Python")
# plt.semilogy(e_cpp,pes_cpp,label = "C++")
# plt.legend()
# plt.savefig("pes.png")
# os.system("mv pes.png ~/Research/TDSE_PETSC/")


#r_range,pdf = np.load("expanded.npy")

# data = np.loadtxt("expanded_state.txt")
# r = data[:,0]
# state_real = data[:,1]
# state_imag = data[:,2]
# state_pdf = state_real**2 + state_imag**2


# plt.semilogy(r,state_pdf,label = "C++")
# #plt.semilogy(r_range,pdf,label = "Python")
# plt.legend()
# plt.savefig("expanded.png")
# os.system("mv expanded.png ~/Research/TDSE_PETSC/")


pad_data = np.loadtxt("pad.txt")
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

#norm = mcolors.LogNorm(vmin=min_val,vmax=max_val)
norm = mcolors.Normalize(vmin=min_val,vmax=max_val)

#sc = ax.scatter(kx,kz,c=pad_p,norm=norm,cmap=cmap)
sc = ax.scatter(kx,ky,c=pad_p,norm=norm,cmap=cmap)
ax.set_aspect("equal",adjustable = "box")
fig.colorbar(sc,ax=ax)
fig.savefig("pad.png")
os.system("mv pad.png ~/Research/TDSE_PETSC/")


