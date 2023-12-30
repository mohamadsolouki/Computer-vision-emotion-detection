import cv2
import numpy as np

def load_image(image_path):
    """
    Load an image from a given path and convert it to grayscale.

    Args:
    image_path (str): Path to the image file.

    Returns:
    numpy.ndarray: Grayscale image.
    """
    # Read the image in grayscale mode
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Image not found at {image_path}")
    return image

def resize_image(image, size=(48, 48)):
    """
    Resize an image to a given size.

    Args:
    image (numpy.ndarray): Input image.
    size (tuple): Desired size as (width, height).

    Returns:
    numpy.ndarray: Resized image.
    """
    resized_image = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
    return resized_image

def normalize_image(image):
    """
    Normalize the pixel values of an image.

    Args:
    image (numpy.ndarray): Input image.

    Returns:
    numpy.ndarray: Normalized image.
    """
    # Normalize pixel values to the range [0,1]
    normalized_image = image / 255.0
    return normalized_image

def preprocess_image(image_path):
    """
    Preprocess an image: load, convert to grayscale, resize, and normalize.

    Args:
    image_path (str): Path to the image file.

    Returns:
    numpy.ndarray: Preprocessed image ready for model input.
    """
    image = load_image(image_path)
    image = resize_image(image)
    image = normalize_image(image)
    # Reshape image to add batch dimension
    image = np.reshape(image, (1, *image.shape, 1))
    return image
