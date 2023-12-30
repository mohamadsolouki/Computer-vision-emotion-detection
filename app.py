from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import cv2
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
model = tf.keras.models.load_model('emotion_detection_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    filename = secure_filename(file.filename)
    file_path = os.path.join('uploads', filename)
    file.save(file_path)

    # Preprocessing the image
    image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (48, 48))
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # Predicting the emotion
    prediction = model.predict(image)
    predicted_class = np.argmax(prediction, axis=1)

    # Mapping the prediction to the actual emotion
    emotions = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
    emotion = emotions[predicted_class[0]]

    return jsonify({'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)
