import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3

# Speech setup
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# Load model
model = load_model("mask_detector.model")

# Load face detector
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

last_label = ""

while True:
    ret, frame = cap.read()

    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        face = cv2.resize(face, (224,224))
        face = face / 255.0
        face = np.reshape(face, (1,224,224,3))

        pred = model.predict(face)

        if pred[0][0] > 0.5:
            label = "Mask"
            color = (0,255,0)

            if last_label != "Mask":
                speak("Thank you for wearing mask")
                last_label = "Mask"

        else:
            label = "No Mask"
            color = (0,0,255)

            if last_label != "No Mask":
                speak("Please wear mask")
                last_label = "No Mask"

        # Draw on screen
        cv2.putText(frame, label, (x,y-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color,2)

    cv2.imshow("Mask Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()