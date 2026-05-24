import cv2
import numpy as np

def distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5

def load_image(image_path):
    img_cv = cv2.imread(image_path)

    if img_cv is None:
        raise ValueError("Could not load image.")

    return img_cv

def detect_edges(img_cv):
    gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 20, 130)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    thick_edges = cv2.dilate(edges, kernel, iterations = 1)

    return thick_edges

def find_page_contour(thick_edges):
    contours, _ = cv2.findContours(thick_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key = cv2.contourArea, reverse = True)

    if not contours:
        raise ValueError("No contours found.")

    return contours[0]

def get_four_corners(page_contour):
    peri = cv2.arcLength(page_contour, True)
    approx = cv2.approxPolyDP(page_contour, 0.02 * peri, True)

    if len(approx) != 4:
        raise ValueError("Could not detect 4 page corners")

    return approx

def order_points(approx):
    pts = approx.reshape(4, 2)

    pts_sorted = sorted(pts, key = lambda p: p[1])

    top_two = pts_sorted[:2]
    bottom_two = pts_sorted[2:]

    top_two = sorted(top_two, key=lambda p: p[0])
    bottom_two = sorted(bottom_two, key=lambda p: p[0])

    top_left = top_two[0]
    top_right = top_two[1]
    bottom_left = bottom_two[0]
    bottom_right = bottom_two[1]

    return top_left, top_right, bottom_right, bottom_left


def warp_image(img_cv, top_left, top_right, bottom_right, bottom_left):
    src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

    width_top = distance(top_left, top_right)
    width_bottom = distance(bottom_left, bottom_right)
    max_width = int(max(width_top, width_bottom))

    height_left = distance(top_left, bottom_left)
    height_right = distance(top_right, bottom_right)
    max_height = int(max(height_left, height_right))

    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]
    ], dtype="float32")

    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img_cv, M, (max_width, max_height))

    return warped


def scan_document(image_path):
    img_cv = load_image(image_path)
    thick_edges = detect_edges(img_cv)
    page_contour = find_page_contour(thick_edges)
    approx = get_four_corners(page_contour)
    top_left, top_right, bottom_right, bottom_left = order_points(approx)

    warped = warp_image(img_cv, top_left, top_right, bottom_right, bottom_left)
    return warped