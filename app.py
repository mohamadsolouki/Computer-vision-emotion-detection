import os
import numpy as np
import cv2
from flask import Flask, render_template, request, Response, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = load_model('xception_model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the image size expected by the model
img_size = (71, 71)

# Load OpenCV face detector model
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        file = request.files['image']  # Get the file from the request
        filename = file.filename  # Save the filename
        img_path = 'uploads/' + filename
        file.save(img_path)  # Save the file to the filesystem

        # Preprocess the image
        img = image.load_img(img_path, target_size=img_size)
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array /= 255.0

        # Make predictions
        predictions = model.predict(img_array)
        predicted_probs = predictions[0]
        predicted_emotion = emotion_labels[np.argmax(predicted_probs)]
        
        # Create a dictionary of emotion labels and their corresponding probabilities
        emotion_probs = {label: prob for label, prob in zip(emotion_labels, predicted_probs)}

        # Detect face and draw rectangle and text on the image
        orig_img = cv2.imread(img_path)
        gray = cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            cv2.rectangle(orig_img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(orig_img, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the image with rectangles
        processed_img_path = 'uploads/processed_' + filename
        cv2.imwrite(processed_img_path, orig_img)

        return render_template('result.html', emotion=predicted_emotion, emotion_probs=emotion_probs, img_path=processed_img_path)
    
    return "No image uploaded"

@app.route('/uploads/<filename>')
def serve_image(filename):
    return send_file(os.path.join('uploads', filename), mimetype='image/jpeg')

@app.route('/webcam')
def webcam():
    return render_template('webcam.html')

def generate_frames():
    camera = cv2.VideoCapture(0)
    while True:
        success, frame = camera.read()
        if not success:
            break
        else:
            # Detect face and draw rectangle and text on the frame
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
            
            for (x, y, w, h) in faces:
                face_frame = frame[y:y+h, x:x+w]
                face_frame = cv2.resize(face_frame, img_size)
                face_frame = face_frame.astype('float32') / 255.0
                face_frame = np.expand_dims(face_frame, axis=0)
                
                # Make predictions
                predictions = model.predict(face_frame)
                predicted_probs = predictions[0]
                predicted_emotion = emotion_labels[np.argmax(predicted_probs)]
                
                # Draw rectangle and text on the frame
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, predicted_emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)