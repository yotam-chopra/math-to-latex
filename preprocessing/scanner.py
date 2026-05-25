import cv2

import numpy as np

import random


class EquationAugmenter:
    def __init__(self):
        pass

    def rotate(self, image):
        angle = random.uniform(-10, 10)

        height, width = image.shape[:2]

        matrix = cv2.getRotationMatrix2D(
            (width // 2, height // 2),
            angle,
            1.0
        )

        rotated = cv2.warpAffine(
            image,
            matrix,
            (width, height),
            borderMode=cv2.BORDER_REPLICATE
        )

        return rotated

    def add_noise(self, image):
        noise = np.random.normal(
            0,
            15,
            image.shape
        ).astype(np.uint8)

        noisy = cv2.add(image, noise)

        return noisy

    def blur(self, image):
        blurred = cv2.GaussianBlur(
            image,
            (3, 3),
            0
        )

        return blurred

    def adjust_brightness(self, image):
        alpha = random.uniform(0.8, 1.2)

        beta = random.randint(-20, 20)

        adjusted = cv2.convertScaleAbs(
            image,
            alpha=alpha,
            beta=beta
        )

        return adjusted

    def augment(self, image_path):
        image = cv2.imread(
            image_path,
            cv2.IMREAD_GRAYSCALE
        )

        image = self.rotate(image)

        image = self.add_noise(image)

        image = self.blur(image)

        image = self.adjust_brightness(image)

        return image


if __name__ == "__main__":
    augmenter = EquationAugmenter()

    augmented = augmenter.augment(
        "../data/rendered/equation_0.png"
    )

    cv2.imshow(
        "Augmented Equation",
        augmented
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()