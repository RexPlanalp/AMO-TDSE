import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt

bspline_data = np.loadtxt("bsplines.txt")

real = bspline_data[:,0]
imag = bspline_data[:,1]

n_basis = 30
Nx = 3000

for i in range(n_basis):
    plt.plot(real[i*Nx:(i+1)*Nx],color = "k")
    plt.plot(imag[i*Nx:(i+1)*Nx],color = "brown")
plt.savefig("bspline.png")