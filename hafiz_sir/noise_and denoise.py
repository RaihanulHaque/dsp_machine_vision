import cv2
import numpy as np

def gaussian_noise(img, sigma=25):
    noise = np.random.normal(0, sigma, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

def uniform_noise(img, high=50):
    noise = np.random.uniform(0, high, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)

def salt_pepper_noise(img, prob=0.01):
    noisy = img.copy()
    coords = np.random.rand(*img.shape) < prob
    noisy[coords] = 255  # This is for salt
    coords = np.random.rand(*img.shape) < prob
    noisy[coords] = 0    # This is for pepper
    return noisy

def gaussian_blur(img):
    return cv2.GaussianBlur(img, (5,5), 0)

def median_blur(img):
    return cv2.medianBlur(img, 5)

def fast_denoise(img):
    return cv2.fastNlMeansDenoising(img, None, 10, 7, 21)

def bilateral_filter(img):
    return cv2.bilateralFilter(img, 9, 75, 75)

img = cv2.imread('/home/lelouch/Code/dsp_machine_vision/image.png', 0)

noisy_g = gaussian_noise(img)
noisy_u = uniform_noise(img)
noisy_sp = salt_pepper_noise(img)


cv2.imshow('Original', img)
cv2.imshow('Gaussian Noise', noisy_g)
cv2.imshow('Uniform Noise', noisy_u)
cv2.imshow('Salt & Pepper Noise', noisy_sp)
cv2.imshow('Gaussian Filter on Gaussian Noise', gaussian_blur(noisy_g))
cv2.imshow('Median Filter on Salt & Pepper Noise', median_blur(noisy_sp))
cv2.imshow('Mean Filter on Uniform Noise', fast_denoise(noisy_u))
cv2.imshow('Bilateral Filter on Gaussian Noise', bilateral_filter(noisy_g))
cv2.waitKey(0)
cv2.destroyAllWindows()
