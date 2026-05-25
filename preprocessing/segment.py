import cv2

import numpy as np


class EquationSegmenter:
    def __init__(self):
        pass

    def segment(self, image_path):
        image = cv2.imread(image_path)

        gray = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2GRAY
        )

        blurred = cv2.GaussianBlur(
            gray,
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

        contours, _ = cv2.findContours(
            thresholded,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return image

        largest_contour = max(
            contours,
            key=cv2.contourArea
        )

        x, y, w, h = cv2.boundingRect(
            largest_contour
        )

        padding = 20

        x = max(0, x - padding)

        y = max(0, y - padding)

        w = min(
            image.shape[1] - x,
            w + 2 * padding
        )

        h = min(
            image.shape[0] - y,
            h + 2 * padding
        )

        cropped = image[
            y:y+h,
            x:x+w
        ]

        return cropped


if __name__ == "__main__":
    segmenter = EquationSegmenter()

    segmented = segmenter.segment(
        "../data/rendered/equation_0.png"
    )

    cv2.imshow(
        "Segmented Equation",
        segmented
    )

    cv2.waitKey(0)

    cv2.destroyAllWindows()