import cv2
import numpy as np
import tensorflow as tf

# Load model
try:
    model = tf.keras.models.load_model('digit_model.h5')
except:
    print("Error: Train the model first using the script above!")
    exit()

def preprocess_digit(roi):
    """Transforms webcam ROI to MNIST format."""
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # Use Gaussian Blur to smooth noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Adaptive threshold handles shadows/lighting better than simple thresholding
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours to crop and center the digit
    cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        digit = thresh[y:y+h, x:x+w]
        # Pad to maintain aspect ratio and look like MNIST
        digit = cv2.copyMakeBorder(digit, 4, 4, 4, 4, cv2.BORDER_CONSTANT, value=0)
        digit = cv2.resize(digit, (28, 28))
    else:
        digit = cv2.resize(thresh, (28, 28))

    return digit.astype('float32') / 255.0, thresh

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Define scanning area (ROI)
    h, w, _ = frame.shape
    x1, y1, x2, y2 = w//2-100, h//2-100, w//2+100, h//2+100
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
    roi = frame[y1:y2, x1:x2]
    processed, debug_img = preprocess_digit(roi)
    
    # Predict
    pred = model.predict(processed.reshape(1, 28, 28, 1), verbose=0)
    digit, conf = np.argmax(pred), np.max(pred)

    # UI Feedback
    color = (0, 255, 0) if conf > 0.8 else (0, 0, 255)
    label = f"Digit: {digit} ({conf*100:.1f}%)" if conf > 0.8 else "Scanning..."
    cv2.putText(frame, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    cv2.imshow("Main Feed", frame)
    cv2.imshow("AI Vision (Binary)", debug_img)

    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()
