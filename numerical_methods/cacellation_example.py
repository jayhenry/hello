import numpy as np
import numpy. polynomial . polynomial as ply
import matplotlib .pyplot as plt

# plot y = (x-2)**9 in red
x = np.linspace (1.95 ,2.05 , 1000)
plt.plot(x,(x -2) **9,'r')

# plot p = x**9 -18x**8+144x**7 -672x**6+2016x**5 -4032x**4+5376x**3 -4608x**2+2304x-512 in blue
roots = 2 * np.ones (9)
p = ply. polyfromroots (roots)
#coefficients are in reverse order for polyval
p = p[:: -1]
plt.plot(x, np.polyval(p,x),'b')

plt.savefig('cacellation_example.png')