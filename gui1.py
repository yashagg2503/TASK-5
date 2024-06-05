import cv2
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

# Load pre-trained model
model = load_model(r'C:\Users\25dea\OneDrive\Desktop\TASK_5\emotion_detection_model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Function to predict emotion
def predict_emotion(face):
    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
    face = cv2.resize(face, (48, 48))
    face = face.astype('float') / 255.0
    face = img_to_array(face)
    face = np.expand_dims(face, axis=0)
    predictions = model.predict(face)[0]
    return emotion_labels[np.argmax(predictions)]

# Create GUI window
class EmotionDetectionApp:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        self.vid = cv2.VideoCapture(video_source)
        self.canvas = Canvas(window, width=self.vid.get(cv2.CAP_PROP_FRAME_WIDTH), height=self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.canvas.pack()

        self.btn_snapshot = Button(window, text="Snapshot", command=self.snapshot)
        self.btn_snapshot.pack(anchor=CENTER, expand=True)

        self.delay = 15
        self.update()
        self.window.mainloop()

    def snapshot(self):
        ret, frame = self.vid.read()
        if ret:
            cv2.imwrite("frame-" + str(int(self.vid.get(cv2.CAP_PROP_POS_FRAMES))) + ".jpg", cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    def update(self):
        ret, frame = self.vid.read()
        if ret:
            faces = self.detect_faces(frame)
            for (x, y, w, h) in faces:
                face = frame[y:y+h, x:x+w]
                emotion = predict_emotion(face)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            self.photo = ImageTk.PhotoImage(image=Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            self.canvas.create_image(0, 0, image=self.photo, anchor=NW)
        self.window.after(self.delay, self.update)

    def detect_faces(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces

    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()

# Run the application
if __name__ == "__main__":
    root = Tk()
    app = EmotionDetectionApp(root, "Emotion Detection")