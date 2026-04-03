import cv2
import numpy as np
import tensorflow as tf

# 1. Load the model
try:
    model = tf.keras.models.load_model('digit_model.h5')
except:
    print("Error: 'digit_model.h5' not found. Please train your model first.")
    exit()

def get_prediction(roi):
    """Refined preprocessing to match MNIST format"""
    # Convert to grayscale and blur to reduce noise
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive threshold handles uneven lighting/shadows
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)

    # Find the largest object (the digit) and center it
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Get the bounding box of the largest contour
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        
        # Crop to the digit and add padding to make it look like MNIST
        digit_crop = thresh[y:y+h, x:x+w]
        padded_digit = cv2.copyMakeBorder(digit_crop, 20, 20, 20, 20, 
                                          cv2.BORDER_CONSTANT, value=0)
        
        # Resize to 28x28
        final_img = cv2.resize(padded_digit, (28, 28))
    else:
        final_img = cv2.resize(thresh, (28, 28))

    # Normalize and reshape for the model
    normalized = final_img / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)
    
    prediction = model.predict(reshaped, verbose=0)
    return np.argmax(prediction), np.max(prediction), thresh

# 2. Main Webcam Loop
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Define a focused ROI (Region of Interest)
    # Draw a blue box: everything inside this box is processed
    x1, y1, x2, y2 = 200, 150, 450, 400
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    roi = frame[y1:y2, x1:x2]

    # Process and Predict
    digit, confidence, debug_img = get_prediction(roi)

    # UI Feedback: Show green if confident, red if not
    color = (0, 255, 0) if confidence > 0.7 else (0, 0, 255)
    label = f"Digit: {digit} ({confidence*100:.1f}%)" if confidence > 0.7 else "Scanning..."
    
    cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Show the "AI View" (what the computer actually sees)
    cv2.imshow("Binary View (AI Eyes)", debug_img)
    cv2.imshow("Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
