import cv2
import numpy as np


def preprocess_image(image):
    """
    Preprocess an image for the emotion detection model.

    Args:
    image (numpy.ndarray): Image array.

    Returns:
    numpy.ndarray: Preprocessed image ready for model input.
    """
    # Ensure the image is in grayscale
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Resize image to 48x48
    image = cv2.resize(image, (48, 48), interpolation=cv2.INTER_AREA)

    # Normalize the image
    image = image / 255.0

    # Reshape image to add batch dimension for model input
    image = np.reshape(image, (1, *image.shape, 1))

    return image

