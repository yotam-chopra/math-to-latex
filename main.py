import cv2
import numpy as np

img_cv = cv2.imread("images/testprime.png")

print(img_cv is None)

gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)

blurred = cv2.GaussianBlur(gray, (5, 5), 0)

edges = cv2.Canny(blurred, 20, 130)

kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(1,2))

thick_edges = cv2.dilate(edges, kernel, iterations = 1)

contours, _ = cv2.findContours(thick_edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

contours = sorted(contours, key = cv2.contourArea, reverse = True)

page_contour = contours[0]

peri = cv2.arcLength(page_contour, True)

approx = cv2.approxPolyDP(page_contour, 0.02 * peri, True)

pts = approx.reshape(4,2)

pts_sorted = sorted(pts, key = lambda p: p[1])

top_two = pts_sorted[:2]
bottom_two = pts_sorted[2:]

top_two = sorted(top_two, key = lambda p: p[0])
bottom_two = sorted(bottom_two, key = lambda p: p[0])

top_left = top_two[0]
top_right = top_two[1]
bottom_left = bottom_two[0]
bottom_right = bottom_two[1]

def distance(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2) ** 0.5

width_top = distance(top_left, top_right)
width_bottom = distance(bottom_left, bottom_right)
max_width = int(max(width_top, width_bottom))

height_left = distance(top_left, bottom_left)
height_right = distance(top_right, bottom_right)
max_height = int(max(height_left, height_right))


dst = np.array([
    [0, 0],
    [max_width - 1, 0],
    [max_width - 1, max_height -1],
    [0, max_height - 1]
], dtype="float32")


src = np.array([top_left, top_right, bottom_right, bottom_left], dtype="float32")

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img_cv, M, (max_width, max_height))

cv2.imwrite("images/warped.png", warped)

cv2.imshow("warped", warped)
cv2.waitKey(0)
cv2.destroyAllWindows()