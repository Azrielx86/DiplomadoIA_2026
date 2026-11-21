import matplotlib.pyplot as plt
import numpy as np

x = np.arange(0, 10)
y = x ** 2

plt.plot(x, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Simple plot')
plt.show()
