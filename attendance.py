import cv2
import face_recognition
import pickle
import pandas as pd
from datetime import datetime
import numpy as np

# Load known face encodings
with open("encodings.pkl", "rb") as f:
    data = pickle.load(f)

# Set to keep track of names already marked as present
marked_names = set()

def mark_attendance(name):
    try:
        df = pd.read_csv("Attendance.csv")
    except FileNotFoundError:
        df = pd.DataFrame(columns=["Name", "Time"])

    if name not in df['Name'].values:
        now = datetime.now()
        dtString = now.strftime('%Y-%m-%d %H:%M:%S')
        new_entry = pd.DataFrame({"Name": [name], "Time": [dtString]})
        df = pd.concat([df, new_entry], ignore_index=True)
        df.to_csv("Attendance.csv", index=False)
    else:
        print(f"{name} is already marked as present today.")

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    img_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
    img_small = cv2.cvtColor(img_small, cv2.COLOR_BGR2RGB)

    faces_cur_frame = face_recognition.face_locations(img_small)
    encodes_cur_frame = face_recognition.face_encodings(img_small, faces_cur_frame)

    for encode_face, face_loc in zip(encodes_cur_frame, faces_cur_frame):
        matches = face_recognition.compare_faces(data["encodings"], encode_face)
        face_dis = face_recognition.face_distance(data["encodings"], encode_face)
        match_index = np.argmin(face_dis)

        if matches[match_index]:
            name = data["names"][match_index].upper()
            if name not in marked_names:
                mark_attendance(name)
                marked_names.add(name)
                y1, x2, y2, x1 = face_loc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
    
    cv2.imshow('Webcam', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
