from matplotlib import pyplot as plt
from facemodel import face_recognition
import cv2

face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')

video_capture = cv2.VideoCapture('Bill Gates gets super lucky playing cards! - Netflix.mp4')

while True:

    ret, frame = video_capture.read()
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Face extract
    faces = face_cascade.detectMultiScale(rgb, 1.3, 5)

    # Recognition and draw bbox with label
    for (x, y, w, h) in faces:

        rect_face = cv2.rectangle(frame, (x, y), (x+w, y+h), (46, 204, 113), 2)
        face = rgb[y:y+h, x:x+w]
        # face = face.astype('float32') / face.max()

        predicted_name, class_probability = face_recognition(face)
        print("Result: ", predicted_name, class_probability)
        if class_probability >= 50:
            rect_face = cv2.rectangle(frame, (x, y-15), (x+w, y+10), (46, 204, 113), -1)
            cv2.putText(rect_face, predicted_name, (x+1, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 236, 240, 241), 2)
        else:
            rect_face = cv2.rectangle(frame, (x, y-15), (x+w, y+10), (46, 204, 113), -1)
            cv2.putText(rect_face, "Unknown", (x+1, y+5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, ( 236, 240, 241), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()