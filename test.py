import numpy as np
import matplotlib.pyplot as plt
import pickle


coul_wave = np.loadtxt("coulomb.txt")
x = coul_wave[:,0]
y = coul_wave[:,1]
plt.plot(x,y)
plt.xlim([1990,2000])
print(y[-1])
plt.title("Coulomb Wave")
plt.savefig("coulomb.png")
plt.clf()


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
# pes_python = np.load("PES.npy")
# plt.semilogy(pes_python,label = "Python")
# plt.semilogy(pes_cpp,label = "C++")
# plt.legend()
# plt.savefig("pes.png")