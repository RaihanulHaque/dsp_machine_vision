import cv2
import numpy as np

image_path = '/home/lelouch/Code/dsp_machine_vision/image.png'
img = cv2.imread(image_path)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 1. Canny
canny_edges = cv2.Canny(blurred, 90, 200)

# 2. Sobel 
# Sobel in X direction
sobel_x = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=5)
# Sobel in Y direction
sobel_y = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=5)
# Combine Sobel X and Y
sobel_combined = cv2.magnitude(sobel_x, sobel_y)
# Normalize for display
sobel_combined = cv2.normalize(sobel_combined, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

# 3. Laplacian
laplacian = cv2.Laplacian(blurred, cv2.CV_64F)
# Convert to uint8 for display
laplacian = cv2.convertScaleAbs(laplacian)

cv2.imshow('Original Image', img)
cv2.imshow('Canny Edge Detection', canny_edges)
cv2.imshow('Sobel Edge Detection', sobel_combined)
cv2.imshow('Laplacian Edge Detection', laplacian)
cv2.waitKey(0)
cv2.destroyAllWindows()