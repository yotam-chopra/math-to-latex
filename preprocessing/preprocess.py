import cv2

import numpy as np


class ImagePreprocessor:
    def __init__(
        self,
        image_size=(224, 224)
    ):
        self.image_size = image_size

    def preprocess(self, image_path):
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        blurred = cv2.GaussianBlur(
            image,
            (5, 5),
            0
        )

        thresholded = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11,
            2
        )

        resized = cv2.resize(
            thresholded,
            self.image_size
        )

        normalized = (
            resized.astype(np.float32)
            / 255.0
        )

        return normalized


if __name__ == "__main__":
    preprocessor = ImagePreprocessor()

    image = preprocessor.preprocess(
        "../data/rendered/equation_0.png"
    )

    print("Processed image shape:")

    print(image.shape)

    print("Min pixel value:")
    print(image.min())

    print("Max pixel value:")
    print(image.max())