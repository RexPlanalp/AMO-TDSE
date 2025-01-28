import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("wave.txt")

x = data[:,0]
y = data[:,1]

plt.plot(x,y)
plt.xlim([0,10])
plt.savefig("wave.png")
