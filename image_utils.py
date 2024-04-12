# -*- coding: utf-8 -*-

__authors__ = "Colin Chow (colinmec), Alden, Zhida Xu"
__version__ = ""

import csv
import pdb #for debug
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import array, fromstring, min as np_min, ndarray, pad
from PIL.Image import open as img_open
from scipy.ndimage import rotate
from scipy.optimize import curve_fit


def rotate_image(img, angle, pivot=None):
    # Rotate image about a pivot point
    # First, pad the image such that it is centered at the pivot point
    # After that, rotate then crop

    if img is None:
        return None

    if pivot is None:
        return rotate(img, angle, reshape=False)

    pad_x = array([img.shape[1] - pivot[0], pivot[0]])
    pad_x = pad_x - np_min(pad_x) + 1
    pad_y = array([img.shape[0] - pivot[1], pivot[1]])
    pad_y = pad_y - np_min(pad_y) + 1
    img_p = pad(img, [pad_y, pad_x], "constant")
    img_r = rotate(img_p, angle, reshape=False)

    return img_r[pad_y[0] : -pad_y[1], pad_x[0] : -pad_x[1]]
    # End of rotate_image


def decode_omet_png(img_file: str) -> (ndarray, ndarray, ndarray, dict):
    # For extracting matrix and axis encoded as PNG following OMET's scheme

    img = img_open(img_file)
    img.load()
    img_mtrx = array(img)
    img_descrb = img.info["Description"].split(";")
    info_dict = {}
    for item in img_descrb:
        [field, value] = item.split(":")
        info_dict[field] = value

    scale = fromstring(info_dict["A"], sep=" ")[0]
    x_axis = fromstring(info_dict["x"], sep=" ")
    y_axis = fromstring(info_dict["y"], sep=" ")

    del info_dict["A"], info_dict["x"], info_dict["y"]

    return img_mtrx * scale, x_axis, y_axis, info_dict
    # End of decode_omet_png


def get_crosshairs(input_image, template, showcross = False):  # template match by Zhida

    # search_image = cv2.imread(search_image_path)
    # template = cv2.imread(template_path)

    # Convert the images to grayscale
    
    search_image = (input_image/256).astype('uint8')
    
    
    if len(search_image.shape) == 3:    #Convert to grey image if RGB 
        search_gray = cv2.cvtColor(search_image, cv2.COLOR_BGR2GRAY)
    else:
        search_gray = search_image.copy()
        #search_gray = cv2.convertScaleAbs(search_image)
    
    if len(template.shape) == 3:    #Convert to grey image if RGB 
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        template_gray = template.copy()

    # Define a threshold to consider a match
    threshold = 0.7

    crosshair_coordinates = []

    #pdb.set_trace()
    for i in range(20):  # max 20 crosshairs
        # Perform template matching
        result = cv2.matchTemplate(search_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # Find the maximum correlation score and its location
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # If the maximum correlation score is above the threshold, consider it a match
        if max_val >= threshold:
            # Get the coordinates of the top-left corner of the match
            x, y = max_loc
            crosshair_coordinates.append(
                (x + template.shape[1] // 2, y + template.shape[0] // 2)
            )
            #pdb.set_trace()
            cv2.rectangle(
                search_image,
                (x, y),
                (x + template.shape[1], y + template.shape[0]),
                (255, 0, 0),
                2,
            )
            cv2.circle(
                search_image,
                (x + template.shape[1] // 2, y + template.shape[0] // 2),
                5,
                (255, 0, 255),
                -1,
            )
            # Draw a black rectangle to cover the matched area
            cv2.rectangle(
                search_gray,
                (x, y),
                (x + template.shape[1], y + template.shape[0]),
                (0, 0, 0),
                -1,
            )
        else:
            # No more matches found
            break

    # Display the result
    if showcross == True: 
        cv2.imshow("Crosshair Detection", search_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    crosshair_coordinates = np.array(crosshair_coordinates)
    return crosshair_coordinates


def get_FOV_rotation(
    coords_a, coords_b
):  # Find rotation angle of coords_b relative to coords_a, by Zhida

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


def seddist(r, k1, k2, k3):
    return 1 + k1 * np.power(r, 2) + k2 * np.power(r, 4) + k3 * np.power(r, 6)


def fisheyetheta(theta, k1, k2, k3, k4):
    return (
        1
        + k1 * np.power(theta, 2)
        + k2 * np.power(theta, 4)
        + k3 * np.power(theta, 6)
        + k4 * np.power(theta, 8)
    )


def readxls(filename, col_name_1, col_name_2):
    df = pd.read_excel(filename)
    r = df[col_name_1]
    y = df[col_name_2]
    return r, y


def distortion_coefficients(
    distortionfile = "", r_col_name="Angle(rad)", y_col_name="Ref_height_ratio(Ftan)", Ftheta= False
):
    r, y = readxls(distortionfile, r_col_name, y_col_name)

    if Ftheta:
        # thetaX = [x * 180/3.1415926 for x in r]  #degrees
        thetaX = r
        thetaY = y
        popt, pcov = curve_fit(fisheyetheta, thetaX, thetaY)
        k1, k2, k3, k4 = popt
        plt.plot(thetaX, thetaY, "b+", label="data")
        plt.plot(thetaX, fisheyetheta(thetaX, *popt), "r-", label="fit")
        print("F-Theta")
        print("K1:", k1)
        print("K2:", k2)
        print("K3", k3)
        print("K4", k4)
        print(popt)
    else:
        popt, pcov = curve_fit(seddist, r, y)
        k1, k2, k3 = popt
        plt.plot(r, y, "b+", label="data")
        plt.plot(r, seddist(r, *popt), "r-", label="fit")
        print("F-Tan")
        print("K1:", k1)
        print("K2:", k2)
        print("K3:", k3)
        print(popt)
    print(distortionfile)
    return popt


def undistort(
    image, dist_coef, Ftheta= False, pixelsize=22, EFL=23.8
):  # pixelsize unit it um, EFL unit in mm, Use fisheye(Ftheta) distortion model if Ftheta is true
    h, w = image.shape[0], image.shape[1]
    K_mtx = np.array(
        [
            [EFL * 1000 / pixelsize, 0, w / 2],
            [0, EFL * 1000 / pixelsize, h / 2],
            [0, 0, 1],
        ]
    )
    if not Ftheta:
        dist = np.array([dist_coef[0], dist_coef[1], 0, 0, dist_coef[2]])
        undistorted = cv2.undistort(image, K_mtx, dist, None, K_mtx)
    else:
        dist = np.array([dist_coef[0], dist_coef[1], dist_coef[2], dist_coef[3]])
        undistorted = cv2.fisheye.undistortImage(image, K_mtx, dist, K_mtx)
    return undistorted, K_mtx

#def contrast

def imagestitch(img1,img2,coords1,coords2):  #stich two images by two coordinates, the order of coords1 need be same as coords 2
    #pdb.set_trace()
    H, _ = cv2.findHomography(coords2, coords1, cv2.RANSAC)
    img2_warped = cv2.warpPerspective(img2, H, (img1.shape[1] + img2.shape[1], img1.shape[0]))
    stitched_image = np.zeros_like(img2_warped)
    stitched_image[:img1.shape[0], :img1.shape[1]] = img1
    stitched_image = cv2.addWeighted(stitched_image, 1, img2_warped, 1, 0)
    #cv2.imshow("Stitched Image", stitched_image)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    
    return stitched_image

def stitchimage_crosshair(img1, img2, template):
    #pdb.set_trace()
    coords1 = get_crosshairs(img1,template)
    coords2 = get_crosshairs(img2,template)
    
    #ranking coords1 to coords2
    if len(coords1) != len(coords2):
        print('Unequal number of crosshairs between 2 images')
        return
    else:
        #pdb.set_trace()
        coords1 = np.array(sorted(coords1, key=lambda x: x[0]))
        coords2 = np.array(sorted(coords2, key=lambda x: x[0]))
        stitched_image = imagestitch(img1,img2,coords1,coords2)
        return stitched_image