import numpy as np
import matplotlib
matplotlib.use("Agg") 
import matplotlib.pyplot as plt
import sys

bspline_data = np.loadtxt("build/debug/bsplines.txt")
dbspline_data = np.loadtxt("build/debug/dbsplines.txt")

real = bspline_data[:,0]
imag = bspline_data[:,1]

dreal = dbspline_data[:,0]
dimag = dbspline_data[:,1]

n_basis = 30
Nx = 3000

for i in range(n_basis):
    plt.plot(real[i*Nx:(i+1)*Nx],color = "k")
    plt.plot(imag[i*Nx:(i+1)*Nx],color = "brown")
plt.savefig("build/debug/bspline.png")
plt.clf()
for i in range(n_basis):
    plt.plot(dreal[i*Nx:(i+1)*Nx],color = "k")
    plt.plot(dimag[i*Nx:(i+1)*Nx],color = "brown")
plt.savefig("build/debug/dbspline.png")



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