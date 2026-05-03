# import necessary packages
import os
import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
from tf_keras.models import load_model

# initialize mediapipe
try:
    mp_solutions = mp.solutions
except AttributeError:
    from mediapipe.python import solutions as mp_solutions

mpHands = mp_solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.9)
mpDraw = mp_solutions.drawing_utils

# Load the gesture recognizer model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model = load_model(os.path.join(BASE_DIR, 'mp_hand_gesture'))

# Load class names
classNames = ["Okay","Peace","Thumbs up","Thumbs down","Call me","Stop","I Love You","Hello","No","Smile"]

# ==========================================
# DROIDCAM SETUP
# ==========================================
# OPTION 1: WiFi IP Address (Most Reliable)
# Look at the DroidCam app on your phone and type the exact WiFi IP here.
# Don't forget to keep the '/video' at the very end!
cap = cv2.VideoCapture('http://10.183.137.158:4747/video')

# OPTION 2: USB Client (If you use the DroidCam desktop app via USB)
# If the WiFi IP above doesn't work, comment it out and uncomment the line below:
# cap = cv2.VideoCapture(1) 

if not cap.isOpened():
    print("Error: Could not connect to DroidCam. Check your IP address or camera index!")
    exit()

while True:
    # Read each frame from the webcam
    ret, frame = cap.read()
    
    # Safety check in case the video feed drops
    if not ret:
        print("Lost connection to DroidCam.")
        break

    x, y, c = frame.shape

    # Flip the frame vertically (horizontally, actually: 1 means horizontal flip)
    frame = cv2.flip(frame, 1)
    framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmark prediction
    result = hands.process(framergb)

    className = ''

    # post process the result
    if result.multi_hand_landmarks:
        landmarks = []
        for handslms in result.multi_hand_landmarks:
            for lm in handslms.landmark:
                lmx = int(lm.x * x)
                lmy = int(lm.y * y)
                landmarks.append([lmx, lmy])
            
            # Drawing landmarks on frames
            mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

            # Predict gesture
            prediction = model.predict([landmarks], verbose=0) # Added verbose=0 to stop terminal spam
            classID = np.argmax(prediction)
            
            if prediction[0][classID] * 100 >= 70.00:
                className = classNames[classID]

    # show the prediction on the frame 
    cv2.putText(frame, className, (20, 60), cv2.FONT_HERSHEY_COMPLEX, 
                   2, (0,194,247), 3, cv2.LINE_AA)

    # Show the final output
    cv2.imshow("Output", frame) 

    if cv2.waitKey(1) == ord('q'):
        break

# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()