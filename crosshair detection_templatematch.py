# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 10:42:27 2023

@author: xzd6089
"""

import cv2
import numpy as np

# Load the search image and the template image (crosshair)

def get_crosshairs(search_image, template):

    #search_image = cv2.imread(search_image_path)
    #template = cv2.imread(template_path)
    
    # Convert the images to grayscale
    search_gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    
    # Define a threshold to consider a match
    threshold = 0.5
    
    crosshair_coordinates = []
    
    for i in range(20):  # max 20 crosshairs
        # Perform template matching
        result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        
        # Find the maximum correlation score and its location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        
        # If the maximum correlation score is above the threshold, consider it a match
        if max_val >= threshold:
            # Get the coordinates of the top-left corner of the match
            x, y = max_loc 
            crosshair_coordinates.append((x + template.shape[1]//2,y + template.shape[0]//2))
            cv2.rectangle(search_image, (x,y), (x + template.shape[1], y + template.shape[0]), (0, 255, 0), 2)
            cv2.circle(search_image, (x + template.shape[1]//2, y + template.shape[0]//2), 5, (0, 0, 255), -1)
            # Draw a black rectangle to cover the matched area
            cv2.rectangle(search_gray, (x, y), (x + template.shape[1], y + template.shape[0]), (0, 0, 0), -1)
        else:
            # No more matches found
            break
    
    # Display the result
    cv2.imshow("Crosshair Detection", search_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    crosshair_coordinates = np.array(crosshair_coordinates)
    return crosshair_coordinates

def get_FOV_rotation(coords_a, coords_b):   #Find rotation angle of coords_b relative to coords_a

    centroid_a = np.mean(coords_a, axis=0)
    centroid_b = np.mean(coords_b, axis=0)

    # Translate the coordinates to the centroid
    translated_coords_a = coords_a - centroid_a
    translated_coords_b = coords_b - centroid_b

    # Calculate the cross-correlation matrix
    cross_corr_matrix = np.dot(translated_coords_a.T, translated_coords_b)

    # Perform Singular Value Decomposition (SVD) to find the rotation matrix
    U, _, VT = np.linalg.svd(cross_corr_matrix)

    # Compute the rotation matrix
    rotation_matrix = np.dot(VT.T, U.T)

    # Calculate the rotation angle
    rotation_angle = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])

    # Convert angle from radians to degrees
    rotation_angle_degrees = np.degrees(rotation_angle)
    
    return rotation_angle_degrees

#def get_FOV(image):
    

if __name__ == "__main__":
    search_image = cv2.imread('crosshairs_image.png')
    template = cv2.imread('crosshair_template.png')
    rotate_image =  cv2.imread('rotate_image.png')  
    coords_a = get_crosshairs(search_image,template)
    coords_b = get_crosshairs(rotate_image,template)
    
    rotate_angle = get_FOV_rotation(coords_a, coords_b)
    print(coords_a)
    print("Rotation angle (in degrees):", rotate_angle)

