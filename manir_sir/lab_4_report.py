import numpy as np
import matplotlib.pyplot as plt

# Parameters for synthetic audio signal
sample_rate = 44100  # Hz
duration = 1.0  # seconds
t = np.linspace(0, duration, int(sample_rate * duration))

# Generate a synthetic audio signal (sum of two sine waves)
freq1, freq2 = 440, 880  # Frequencies in Hz (e.g., A4 and A5 notes)
signal = 0.5 * np.sin(2 * np.pi * freq1 * t) + 0.3 * np.sin(2 * np.pi * freq2 * t)

# Compute DFT using FFT
dft = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(dft), 1 / sample_rate)

# Compute IDFT to reconstruct the signal
reconstructed_signal = np.fft.ifft(dft).real

# Plotting
plt.figure(figsize=(15, 10))

# Plot original signal
plt.subplot(3, 1, 1)
plt.plot(t[:1000], signal[:1000])  # Plot first 1000 samples for clarity
plt.title('Original Audio Signal (Time Domain)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

# Plot frequency spectrum (magnitude)
plt.subplot(3, 1, 2)
plt.plot(frequencies[:len(frequencies)//2], np.abs(dft)[:len(frequencies)//2])
plt.title('Frequency Spectrum (DFT Magnitude)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.xlim(0, 2000)  # Limit to 0-2000 Hz for better visualization

# Plot reconstructed signal
plt.subplot(3, 1, 3)
plt.plot(t[:1000], reconstructed_signal[:1000])
plt.title('Reconstructed Audio Signal (IDFT)')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')

plt.tight_layout()

# Verify reconstruction accuracy
mse = np.mean((signal - reconstructed_signal) ** 2)
print(f'Mean Squared Error between original and reconstructed signal: {mse:.2e}')

# Display plots (commented out for non-interactive environments)
plt.show()