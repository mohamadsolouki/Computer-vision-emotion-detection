import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization

def create_model(input_shape=(48, 48, 1), num_classes=7):
    """
    Create a CNN model for emotion detection.

    Args:
    input_shape (tuple): The shape of the input images.
    num_classes (int): The number of emotion classes.

    Returns:
    tensorflow.keras.Model: The constructed CNN model.
    """
    model = Sequential([
        # Convolutional layer with Batch Normalization
        Conv2D(64, kernel_size=(3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        # Adding more convolutional layers with increasing filters
        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),

        Conv2D(256, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.5),

        # Flattening and Fully Connected Layers
        Flatten(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')  # Output layer for classification
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model
