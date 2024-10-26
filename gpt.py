import cv2
import numpy as np
import pytesseract

# Update this path to where Tesseract is installed
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
# Load the image
# Load the image
image = cv2.imread('poker.png')

# Preprocess: Grayscale, Gaussian blur, and thresholding
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
_, thresh = cv2.threshold(blur, 120, 255, cv2.THRESH_BINARY_INV)

# Canny Edge Detection
edges = cv2.Canny(thresh, 50, 150)

# Find contours
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Loop through contours and draw bounding boxes
for contour in contours:
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    if len(approx) == 4:  # Ensure it's a rectangle (potential card)
        x, y, w, h = cv2.boundingRect(contour)
        
        # Extract region of interest (ROI) for each card
        card_roi = image[y:y+h, x:x+w]
        
        # Apply OCR or template matching for rank/suit detection (as previously explained)
        card_text = pytesseract.image_to_string(card_roi, config='--psm 6')

        # Draw rectangle around detected card
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print("Detected card text:", card_text)

# Show the detected card image
cv2.imshow('Detected Cards', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
