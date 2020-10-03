import numpy as np
import cv2
import matplotlib.pyplot as plt

# Read in the cascade classifiers for face
face_cascade = cv2.CascadeClassifier('haarcascade_face.xml')

# Reading in the image and creating copies
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    for (x, y, w, h) in faces:
        cv2.rectangle(gray, (x, y), (x+w, y+h), (255, 0, 0), 3)
        print(x, y, w, h)

    cv2.imshow('image', gray)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
