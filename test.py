import numpy as np
import matplotlib.pyplot as plt

n = np.arange(-10, 11)  # from -10 to 10

magnitude = 5
x = magnitude * np.ones_like(n)

# Plot using stem
plt.figure(figsize=(8, 4))
plt.stem(n, x)
plt.title('Discrete Signal with Constant Magnitude')
plt.xlabel('n (Discrete Time Index)')
plt.ylabel('Amplitude')
plt.grid(True)
plt.tight_layout()
plt.show()