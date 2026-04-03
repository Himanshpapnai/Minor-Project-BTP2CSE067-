import tensorflow as tf
import cv2
import numpy as np

def preprocess_roi(roi):
    # 1. Convert to Grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # 2. Gaussian Blur to remove camera sensor noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # 3. Adaptive Thresholding (Crucial for handling shadows)
    # Block size 11 and constant 2 are standard starting points
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # 4. Clean up small dots (Morphology)
    kernel = np.ones((3,3), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # 5. Resize and Normalize
    resized = cv2.resize(thresh, (28, 28))
    normalized = resized / 255.0
    return normalized.reshape(1, 28, 28, 1), thresh
cap = cv2.VideoCapture(0)
model = tf.keras.models.load_model('enhanced_digit_model.h5')

while True:
    ret, frame = cap.read()
    if not ret: break
    
    # Define ROI
    x, y, w, h = 150, 150, 250, 250
    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    roi = frame[y:y+h, x:x+w]
    
    # Preprocess and Predict
    input_data, debug_view = preprocess_roi(roi)
    prediction = model.predict(input_data, verbose=0)
    
    digit = np.argmax(prediction)
    confidence = np.max(prediction)
    
    # Logic: Only show if confidence is high
    if confidence > 0.85:
        text = f"Digit: {digit} ({confidence*100:.1f}%)"
        cv2.putText(frame, text, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Align Digit...", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    cv2.imshow("Main Feed", frame)
    cv2.imshow("AI Vision (Binary)", debug_view) # Shows you exactly what the AI sees

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
