

import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
import winsound

st.set_page_config(page_title="Drowsiness Detection")
st.title("😴 CNN-based Drowsiness Detection")


model = tf.keras.models.load_model(
    r"C:\Users\Leelaprasad\PycharmProjects\PythonProject5\.venv\Lib\eye_state_model.keras"
)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)


if "run" not in st.session_state:
    st.session_state.run = False

start = st.button("Start Camera")
stop = st.button("Stop Camera")

frame_window = st.image([])  # placeholder for webcam frames

closed_frames = 0
DROWSY_THRESHOLD = 5
alarm_on = False

if start:
    st.session_state.run = True
if stop:
    st.session_state.run = False


cap = cv2.VideoCapture(0)

while st.session_state.run: #Runs only when camera is ON
    ret, frame = cap.read() #ret=true/false
    if not ret:
        st.error("Webcam not working")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    eye_closed = False


    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray)

        if len(eyes) > 0:
            # Select the largest detected eye
            ex, ey, ew, eh = max(eyes, key=lambda e: e[2]*e[3]) #area = width × height(e[2]=height and e[3]=width)


            eye_img = roi_color[ey:ey+eh, ex:ex+ew]
            eye_img = cv2.resize(eye_img, (64,64))
            eye_img = eye_img / 255.0
            eye_img = np.expand_dims(eye_img, axis=0)
#after each step
            # After crop	(h, w, 3)
            # After resize	(64, 64, 3)
            # After normalize	(64, 64, 3)
            # After expand_dims	(1, 64, 64, 3)
            prediction = model.predict(eye_img, verbose=0) #verbose==No progress bar or logs are printed
            eye_closed = prediction[0][0] < 0.5

            label = "Closed" if eye_closed else "Open"
            color = (0, 0, 255) if eye_closed else (0, 255, 0)

            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), color, 2)
            cv2.putText(roi_color, label, (ex, ey-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    #Drowsiness Logic
    if eye_closed:
        closed_frames += 1
    else:
        closed_frames = 0
        alarm_on = False

    if closed_frames > DROWSY_THRESHOLD:
        if not alarm_on:
            winsound.PlaySound(
                r"C:\Users\Leelaprasad\PycharmProjects\PythonProject5\alarm.wav",
                winsound.SND_FILENAME | winsound.SND_ASYNC
            )
            alarm_on = True
    else:
        winsound.PlaySound(None, winsound.SND_PURGE)
        alarm_on = False

    # Display Frame
    frame_rgb = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    frame_window.image(frame_rgb)

cap.release()
