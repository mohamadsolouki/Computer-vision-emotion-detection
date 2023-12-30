import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from model import create_model



# Constants
TRAIN_DIR = 'data/train'  # Path to training data
TEST_DIR = 'data/test'    # Path to testing data
IMG_SIZE = (48, 48)       # Image size, matching the model input
BATCH_SIZE = 32           # Batch size for training
EPOCHS = 50               # Number of epochs for training

def main():
    # Data Generators for training and validation
    train_datagen = ImageDataGenerator(rescale=1./255, rotation_range=10, width_shift_range=0.1,
                                       height_shift_range=0.1, shear_range=0.1, zoom_range=0.1,
                                       horizontal_flip=True, fill_mode='nearest')
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(TRAIN_DIR, target_size=IMG_SIZE, color_mode='grayscale',
                                                        batch_size=BATCH_SIZE, class_mode='categorical')
    test_generator = test_datagen.flow_from_directory(TEST_DIR, target_size=IMG_SIZE, color_mode='grayscale',
                                                      batch_size=BATCH_SIZE, class_mode='categorical')

    # Create the model
    model = create_model(input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1))

    # Train the model
    model.fit(train_generator, validation_data=test_generator, epochs=EPOCHS)

    # Save the trained model
    model.save('emotion_detection_model.h5')

if __name__ == '__main__':
    main()
