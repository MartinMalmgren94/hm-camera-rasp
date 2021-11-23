import cv2
import numpy as np
import time

# Getting the camera we are going to use on the device. (on pi = -1, on computer = 0)
cap = cv2.VideoCapture(0)
# Setting xml classifer for frontal face
face_cascade = cv2.CascadeClassifier("./haarcascade/haarcascade_frontalface_default.xml")
# Setting start time of script
start_time = time.time()

while (True):
    ret, frame = cap.read()
    # Setting the capure to gray to make it easier
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detecting faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    # Looking for faces in the cap.
    for (x, y, w, h) in faces:
        elapsed_time = time.time() - start_time 
        # Checking every 2 secound
        if(elapsed_time > 2):
            # Marking where the face is on the cap.
            color = (255, 0, 0)
            stroke = 2
            end_cord_x = x + w
            end_cord_y = y + h
            cv2.rectangle(frame, (x, y), (end_cord_x, end_cord_y), color, stroke)
            start_time = time.time()

    # Naming the window that will display the video and displaying it.
    cv2.imshow("cap", frame)
    # To interrupt the program easy.
    if not ret:
        break
    k = cv2.waitKey(1)

    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
# Releasing the camera.
cap.release()
# Closing all the windows.
cv2.destroyAllWindows()