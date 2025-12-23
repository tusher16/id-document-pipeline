import cv2
import numpy as np

def deskew_hough(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLines(edges, 1, np.pi / 180, 200)
    if lines is None:
        return bgr

    angles = []
    for rho, theta in lines[:, 0]:
        angle = (theta - np.pi/2) * (180/np.pi)
        angles.append(angle)

    if not angles:
        return bgr

    rot = float(np.median(angles))
    h, w = bgr.shape[:2]
    M = cv2.getRotationMatrix2D((w/2, h/2), rot, 1.0)
    return cv2.warpAffine(bgr, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
