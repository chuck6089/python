# -*- coding: utf-8 -*-
"""
Created on Thu Nov 16 16:23:47 2023

@author: xzd6089
"""

import cv2
import numpy as np

def find_chessboard_corners(image_path, checkerboard_size):
    # Load the image
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the corners in the image
    ret, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)

    # If corners are found, draw them on the image
    if ret:
        # Draw the corners on the image
        cv2.drawChessboardCorners(image, checkerboard_size, corners, ret)

        # Display the image with corners
        cv2.imshow('Corners', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Corners not found in the image.")

if __name__ == "__main__":
    # Specify the path to your image
    image_path = r"C:\Users\xzd6089\Dropbox (Meta)\Python data processing\test\slbnegchecker.png"

    # Define the dimensions of the chessboard (number of corners in each row and column)
    checkerboard_size = (17, 17)

    # Call the function to find and display corners
    find_chessboard_corners(image_path, checkerboard_size)