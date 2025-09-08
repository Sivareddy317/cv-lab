import cv2
image = cv2.imread(r"C:\Users\prasa.SIVA_REDDY\OneDrive\Pictures\image.jpeg")
cv2.imshow('Original Image', image)
cv2.waitKey(0)
resized = cv2.resize(image, (300, 300))  
cv2.imshow('Resized Image', resized)
cv2.waitKey(0)
cropped = image[100:400, 200:500]  
cv2.imshow('Cropped Image', cropped)
cv2.waitKey(0)
rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
cv2.imshow('Rotated Image', rotated)
cv2.waitKey(0)
cv2.destroyAllWindows()





