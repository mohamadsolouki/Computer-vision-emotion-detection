# Emotion Detection using Xception and Flask

This project aims to detect emotions from facial expressions using deep learning techniques. It utilizes the Xception model and attention layer to classify emotions into seven categories: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral. The model is trained on a dataset of facial expressions (FER-2013) and can be used to predict emotions from uploaded images or real-time webcam feed through a Flask web application.

## Project Structure

The project has the following structure:

- `data/`: This folder contains a text file with the link to the dataset used for training and testing the model. The actual dataset is not included in the repository due to its large size.
- `templates/`: This folder contains the HTML templates for the Flask web application:
  - `index.html`: The main page of the web application.
  - `result.html`: The page that displays the predicted emotion and probabilities for an uploaded image.
  - `webcam.html`: The page that captures real-time webcam feed and displays the predicted emotions.
- `uploads/`: This folder stores the uploaded images for analysis in the web application.
- `logs/`: This folder contains TensorFlow logs generated during model training. It is ignored in the repository but will be created when running the application.
- `app.py`: The main Flask application file that handles routes, image uploads, and real-time emotion detection.
- `model.py` and `model.ipynb`: These files contain the code for creating and training the emotion detection model.
- `requirements.txt`: The file listing the required Python dependencies for the project.
- `.gitignore`: The file specifying files and directories to be ignored by Git.
- `LICENSE`: The license file for the project.

Note: When running the model, two files named `best_xception_model.keras` and `xception_model.h5` will be created. These files are not uploaded to the repository due to their large size.

## Dependencies

The project requires the following dependencies:

- Python 3.x
- Keras
- Pillow
- NumPy
- OpenCV (cv2)
- Flask
- Streamlit
- Scikit-learn
- Matplotlib
- Seaborn

You can install the required dependencies using the following command:

```
pip install -r requirements.txt
```

## Dataset

The dataset used for training and testing the emotion detection model is not included in the repository due to its large size. However, you can find the link to the dataset in the `data/` folder. Download the dataset and place it in the appropriate directory before running the model.

## Model Training

To train the emotion detection model, you can run the `model.py` file or execute the cells in the `model.ipynb` notebook. The model architecture and training process are defined in these files. The trained model will be saved as `xception_model.h5`.

## Running the Application

To run the Flask web application, follow these steps:

1. Install the required dependencies using `pip install -r requirements.txt`.
2. Run the `app.py` file using the command `python app.py`.
3. Access the web application in your browser at `http://localhost:5000`.

You can upload an image for emotion detection or use the webcam feed for real-time emotion detection.

## License

This project is licensed under the [MIT License](LICENSE).

---