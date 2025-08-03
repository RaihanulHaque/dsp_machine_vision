import numpy as np
import matplotlib.pyplot as plt


x = np.array([1, 2, 3, 4])
h = np.array([4, 3, 2, 1])


N = len(x)

# Circular convolution using summation formula
y = np.zeros(N)
for n in range(N):
    for m in range(N):
        y[n] += x[m] * h[(n - m) % N]

# Plotting Circular Convolution result
plt.figure()
plt.stem(range(N), y)
plt.title('Circular Convolution using Summation Formula')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)
plt.savefig('fig_3_1.png')

# Circular convolution using matrix method
H = np.zeros((N, N))
for i in range(N):
    H[:, i] = np.roll(h, i)
y_matrix = H @ x

# Plotting Matrix Method result
plt.figure()
plt.stem(range(N), y_matrix)
plt.title('Circular Convolution using Matrix Method')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)
plt.savefig('fig_3_2.png')

# Zero-padding for linear convolution
x_padded = np.pad(x, (0, N), mode='constant')
h_padded = np.pad(h, (0, N), mode='constant')

# Perform linear convolution on padded sequences
y_padded = np.convolve(x_padded, h_padded)


plt.figure()
plt.stem(range(4 * N - 1), y_padded)
plt.title('Linear Convolution using Circular Convolution with Zero Padding')
plt.xlabel('n')
plt.ylabel('y[n]')
plt.grid(True)
plt.savefig('fig_3_3.png')

plt.show()