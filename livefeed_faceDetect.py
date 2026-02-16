import cv2 as cv
import os

cap = cv.VideoCapture(0)
haar_cascade_path = os.path.join(os.path.dirname(__file__), 'haar_face.xml')
haar_cascade = cv.CascadeClassifier(haar_cascade_path)

while(True):
    ret, frame = cap.read()
    if not ret:
        break
    grey = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    faces_rect = haar_cascade.detectMultiScale(grey, scaleFactor=1.3, minNeighbors=9)
    number_of_faces = len(faces_rect)
    for (x,y,w,h) in faces_rect:
        cv.rectangle(frame, (x,y), (x+w,y+h), (0,255,0), thickness=2)
    
    text = f"Number of faces: {number_of_faces}"
    cv.putText(frame, text, (10,30), cv.FONT_HERSHEY_COMPLEX, 0.8, (0,0,255), 2)
    cv.imshow('Facial Detection Live', frame)
    if cv.waitKey(1) & 0xFF == ord('e'):
        break

cap.release()
cv.destroyAllWindows()

