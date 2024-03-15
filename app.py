from PIL import Image
from flask import Flask, request, jsonify, render_template
import tensorflow as tf
import numpy as np
import cv2
import os
from preprocess import preprocess_image

app = Flask(__name__)
model = tf.keras.models.load_model('best_xception_model.keras')


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict_emotion():
    file = request.files['file']
    if not file:
        return jsonify({'error': 'No file provided'}), 400

    # Convert file stream to a numpy array
    file_stream = file.stream.read()
    npimg = np.frombuffer(file_stream, np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    # Convert to grayscale as the model expects grayscale images
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    processed_image = preprocess_image(image)

    # Predict the emotion
    prediction = model.predict(processed_image)
    predicted_class = np.argmax(prediction, axis=1)
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion = emotions[predicted_class[0]]

    return jsonify({'emotion': emotion})


@app.route('/metrics', methods=['GET'])
def get_metrics():
    # Load metrics from a file or database
    # Replace the below line with actual loading mechanism
    metrics = {}  # Example: {"accuracy": 0.95, "f1_score": 0.94, ...}
    return jsonify(metrics)


if __name__ == '__main__':
    app.run(debug=True, port=5000)
