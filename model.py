from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator


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


# Constants
TRAIN_DIR = 'data/train'  # Path to training data
TEST_DIR = 'data/test'  # Path to testing data
IMG_SIZE = (48, 48)  # Image size, matching the model input
BATCH_SIZE = 32  # Batch size for training
EPOCHS = 50  # Number of epochs for training


def main():
    # Create the model
    model = create_model(input_shape=(*IMG_SIZE, 1), num_classes=7)
    model.summary()

    # Create data generators
    train_datagen = ImageDataGenerator(rescale=1.0 / 255.0, horizontal_flip=True)
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

    # Create data loaders from generators
    train_loader = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, color_mode='grayscale',
                                                     batch_size=BATCH_SIZE, class_mode='categorical')
    test_loader = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, color_mode='grayscale',
                                                   batch_size=BATCH_SIZE, class_mode='categorical')

    # Train the model
    model.fit(train_loader, epochs=EPOCHS, validation_data=test_loader)

    # Save the model
    model.save('emotion_detection_model.h5')


if __name__ == '__main__':
    main()
