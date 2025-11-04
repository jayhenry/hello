import math
import matplotlib .pyplot as plt
import numpy
x=2.0
pn =0.0
error =[]
for j in range (0 ,26):
    pn = pn + (x**j)/math. factorial (j)
    error.append(math.exp (2.0) -pn)

j = numpy.arange (0 ,26)
plt.semilogy(j,error)

plt.savefig('taylor.png')
