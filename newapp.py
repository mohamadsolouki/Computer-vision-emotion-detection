import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.xception import preprocess_input

# Load the trained model
model = load_model('best_xception_model.keras')

# Define the class labels
class_labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# Streamlit app
def main():
    st.title("Emotion Detection App")

    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Perform emotion detection on the uploaded image
        image = cv2.resize(image, (75, 75))
        image = preprocess_input(image)
        predictions = model.predict(np.expand_dims(image, axis=0))
        predicted_class = np.argmax(predictions[0])
        predicted_label = class_labels[predicted_class]

        st.write("Predicted Emotion:", predicted_label)
        st.write("Probabilities:")
        for label, prob in zip(class_labels, predictions[0]):
            st.write(f"{label}: {prob:.4f}")

    # Real-time webcam analysis
    if st.button("Start Webcam"):
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Perform emotion detection on each frame
            frame = cv2.resize(frame, (75, 75))
            frame = preprocess_input(frame)
            predictions = model.predict(np.expand_dims(frame, axis=0))
            predicted_class = np.argmax(predictions[0])
            predicted_label = class_labels[predicted_class]

            # Display the predicted emotion and probabilities on the frame
            cv2.putText(frame, "Emotion: " + predicted_label, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            for i, (label, prob) in enumerate(zip(class_labels, predictions[0])):
                cv2.putText(frame, f"{label}: {prob:.4f}", (10, 60 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Display the frame in the Streamlit app
            st.image(frame, channels="BGR")

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    main()