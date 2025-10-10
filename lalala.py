import cv2
import matplotlib.pyplot as plt


image_path = "/Users/rahi/Code/dsp_machine_vision/hafiz_sir/9255.png"
img = cv2.imread(image_path)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Calculate histogram
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Apply thresholding (simple binary)
_, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY )

cv2.imshow("Original Image", img)
cv2.imshow("Grayscale Image", gray)
cv2.imshow("Thresholded Image", thresh)

plt.figure()
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.plot(hist)
plt.xlim([0, 256])
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
