import os
import numpy as np
import cv2
from flask import Flask, render_template, request, Response, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from io import BytesIO

app = Flask(__name__)

# Load the trained model
model = load_model('xception_model.h5')

# Define the emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define the image size expected by the model
img_size = (71, 71)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' in request.files:
        img = request.files['image']
        img_path = 'uploads/' + img.filename
        img.save(img_path)

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

        return render_template('result.html', emotion=predicted_emotion, emotion_probs=emotion_probs, img_path=img_path)
    
    return "No image uploaded"

@app.route('/uploads/<path:filename>')
def serve_image(filename):
    return send_file('uploads/' + filename, mimetype='image/jpeg')

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
            # Resize the frame to a larger size
            frame = cv2.resize(frame, (400, 400))
            
            # Preprocess the frame
            frame_array = cv2.resize(frame, img_size)
            frame_array = np.expand_dims(frame_array, axis=0)
            frame_array = frame_array.astype('float32') / 255.0
            
            # Make predictions
            predictions = model.predict(frame_array)
            predicted_probs = predictions[0]
            
            # Draw the predicted probabilities for each emotion on the frame
            for i, (label, prob) in enumerate(zip(emotion_labels, predicted_probs)):
                text = f"{label}: {prob:.2f}"
                cv2.putText(frame, text, (10, 30 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)