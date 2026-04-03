from scanner import scan_document
from preprocess import  preprocess_image
from segment import segment_lines
import cv2


def main():
    image_path = input("Enter image path: ")

    warped = scan_document(image_path)
    processed = preprocess_image(warped)
    lines = segment_lines(processed)

    for i, (start, end) in enumerate(lines):
        line_img = warped[start:end, :]
        cv2.imwrite(f"lines/line_{i}.png", line_img)

    line_preview = warped.copy()

    for start, end in lines:
        cv2.rectangle(line_preview, (0, start), (warped.shape[1], end), (0, 0, 255), 2)



    cv2.imshow("warped", warped)
    cv2.imshow("processed", processed)
    cv2.imshow("lines", line_preview)

    cv2.imwrite("images/warped.png", warped)
    cv2.imwrite("images/processed.png", processed)

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()