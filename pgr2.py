import cv2
import numpy as np

# Load image
image = cv2.imread(r"C:\Users\prasa.SIVA_REDDY\OneDrive\Pictures\image.jpeg")  # Replace with actual path

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Show grayscale
cv2.imshow('Grayscale', gray)
cv2.waitKey(0)

# ---------------------- Sobel Edge Detection ----------------------

# Sobel X and Y
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

# Convert to uint8
sobelx = cv2.convertScaleAbs(sobelx)
sobely = cv2.convertScaleAbs(sobely)
sobel_combined = cv2.bitwise_or(sobelx, sobely)

cv2.imshow('Sobel X', sobelx)
cv2.imshow('Sobel Y', sobely)
cv2.imshow('Sobel Combined', sobel_combined)
cv2.waitKey(0)

# ---------------------- Canny Edge Detection ----------------------

canny_edges = cv2.Canny(gray, 100, 200)
cv2.imshow('Canny Edge Detection', canny_edges)
cv2.waitKey(0)

# ---------------------- Thresholding on Grayscale ----------------------

_, binary_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
_, inv_thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

cv2.imshow('Binary Threshold', binary_thresh)
cv2.imshow('Inverse Threshold', inv_thresh)
cv2.waitKey(0)

# ---------------------- Thresholding on Color ----------------------

# Split the channels
b, g, r = cv2.split(image)

# Apply threshold on each channel
_, b_thresh = cv2.threshold(b, 127, 255, cv2.THRESH_BINARY)
_, g_thresh = cv2.threshold(g, 127, 255, cv2.THRESH_BINARY)
_, r_thresh = cv2.threshold(r, 127, 255, cv2.THRESH_BINARY)

# Merge thresholded channels
color_thresh = cv2.merge((b_thresh, g_thresh, r_thresh))

cv2.imshow('Color Thresholded Image', color_thresh)
cv2.waitKey(0)

# Cleanup
cv2.destroyAllWindows()
