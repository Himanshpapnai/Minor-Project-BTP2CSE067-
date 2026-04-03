import cv2
import numpy as np
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('digit_model.h5')

# Connect to the webcam (0 is usually the default internal camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    # Define a window/box where the user should show the digit
    # (x1, y1) to (x2, y2)
    cv2.rectangle(frame, (100, 100), (350, 350), (0, 255, 0), 2)
    
    # Crop the image to the box (ROI)
    roi = frame[100:350, 100:350]
    
    # Preprocess ROI for the model
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28))
    # MNIST is white digits on black background; invert if your paper is white
    inverted = cv2.bitwise_not(resized) 
    normalized = inverted / 255.0
    reshaped = normalized.reshape(1, 28, 28, 1)

    # Predict
    prediction = model.predict(reshaped)
    digit = np.argmax(prediction)
    confidence = np.max(prediction)

    # Display result on screen
    cv2.putText(frame, f"Digit: {digit} ({confidence*100:.1f}%)", 
                (100, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Handwritten Digit Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
