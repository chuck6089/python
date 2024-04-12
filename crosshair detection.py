# -*- coding: utf-8 -*-
"""
Created on Wed Nov  1 14:52:38 2023

@author: xzd6089
"""

import cv2
import numpy as np

# Load the image
image = cv2.imread('crosshairs_image.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian blur and Canny edge detection
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)

# Perform Hough Line Transformation to detect lines
lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

# Initialize a list to store the endpoints of the detected lines
line_endpoints = []

if lines is not None:
    for line in lines:
        rho, theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * rho
        y0 = b * rho
        x1 = int(x0 + 1000 * (-b))
        y1 = int(y0 + 1000 * (a))
        x2 = int(x0 - 1000 * (-b))
        y2 = int(y0 - 1000 * (a))
        line_endpoints.append((x1, y1, x2, y2))

# Filter the lines based on length and angle
filtered_lines = []
for x1, y1, x2, y2 in line_endpoints:
    line_length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    angle = np.arctan2(y2 - y1, x2 - x1)
    angle_deg = np.degrees(angle)
    if 50 < line_length < 200 and abs(angle_deg) < 10:
        filtered_lines.append((x1, y1, x2, y2))

# Find and draw the intersection points for each pair of lines
for i in range(len(filtered_lines)):
    for j in range(i + 1, len(filtered_lines)):
        x1, y1, x2, y2 = filtered_lines[i]
        x3, y3, x4, y4 = filtered_lines[j]
        den = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if den != 0:
            px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / den
            py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / den
            intersection_point = (int(px), int(py))
            cv2.circle(image, intersection_point, 5, (0, 0, 255), -1)  # Highlight the intersection point

# Display the result
cv2.imshow('Multiple Crosshair Detection', image)
cv2.waitKey(0)
cv2.destroyAllWindows()