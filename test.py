import numpy as np
import matplotlib.pyplot as plt

a = np.arange(10,1000,1)
ratio = [1-(i/10000) for i in a]
ratio = np.array(ratio)
ratio = ratio/np.mean(ratio)
print(ratio)
ratio1 = [1-(i/10000)**2 for i in a]
ratio1 = np.array(ratio1)
ratio1 = ratio1/np.mean(ratio1)
print(ratio1)
ratio2 = [1-(i/10000)**(1/2) for i in a]
ratio2 = np.array(ratio2)
ratio2 = ratio2/np.mean(ratio2)
print(ratio2)

plt.plot(a,ratio,label='1')
plt.plot(a,ratio1,label='2')
plt.plot(a,ratio2,label='3')
plt.legend()
plt.show()