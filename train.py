import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model
from preprocess import preprocess_image

# Parameters
data_dir = 'path_to_your_dataset'  # Replace with the path to your dataset
batch_size = 32
num_classes = 7  # Update based on your dataset
input_shape = (48, 48, 1)
epochs = 50

def load_data(directory):
    """
    Load and preprocess data from the given directory.

    Args:
    directory (str): Path to the data directory.

    Returns:
    X (numpy.ndarray): Array of images.
    y (numpy.ndarray): Array of labels.
    """
    X = []
    y = []
    labels = os.listdir(directory)
    label_map = {label: idx for idx, label in enumerate(labels)}

    for label in labels:
        images = os.listdir(os.path.join(directory, label))
        for img_name in images:
            img_path = os.path.join(directory, label, img_name)
            img = preprocess_image(img_path)
            X.append(img)
            y.append(label_map[label])

    return np.array(X), np.array(y)

def main():
    # Load and preprocess the data
    X, y = load_data(data_dir)

    # Convert labels to one-hot encoding
    y = tf.keras.utils.to_categorical(y, num_classes)

    # Create the model
    model = create_model(input_shape, num_classes)

    # Data augmentation
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest')

    # Train the model
    model.fit(datagen.flow(X, y, batch_size=batch_size), epochs=epochs)

    # Save the model
    model.save('emotion_detection_model.h5')

if __name__ == '__main__':
    main()
