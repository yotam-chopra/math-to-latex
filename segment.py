import cv2
import numpy as np

def segment_lines(processed_image):
    inverted = cv2.bitwise_not(processed_image)
    row_sums = np.sum(inverted, axis = 1)

    threshold = 1000

    is_text = row_sums > threshold

    lines = []
    in_line = False
    start = 0

    for i, value in enumerate(is_text):
        if value and not in_line:
            start = i
            in_line = True
        elif not value and in_line:
            end = i
            lines.append((start, end))
            in_line = False

    if end - start > 5 and np.mean(row_sums[start:end]) > 2000:
        lines. append((start, end))

    merged = []

    for start, end in lines:
        if not merged:
            merged.append([start, end])
        else:
            prev_start, prev_end = merged[-1]

            if start - prev_end < 10:
                merged[-1][1] = end
            else:
                merged.append([start, end])

    merged = [(s, e) for s, e in merged]

    return merged
