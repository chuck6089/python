import numpy as np
import json
from skimage.color import lab2xyz, xyz2lab, xyz2rgb
from scipy.linalg import lstsq
from color_tools.utilities.wuc_unifmormity_metrics import *
from metrics.color_tools import (
    image2XYZ,
    Lab2dLdCdE,
    pow2lightness,
    SCIELAB_filter,
    set_Lab_white,
    XYZ2SpatialCIELAB,
)
from typing import Tuple, Dict, List
from metrics.utilities import (
    conv2,
    fov_zones,
    normalize_zone,
    resize_and_filter_img,
)
from metrics.uniformity_metrics import (
	compute_baseline_shape,
	fov_zones,
	gen_params,
	baselineRolloffXYZ,
    wg_synthetic_correction,
    whiteXYZ_IQ
)
from metrics import sort2grid, MTF_cal
from CTT_Analysis_20240117backup import CTTanalysis

import pandas as pd
import os, sys, pdb, util, config, cv2, time, copy, colour, math, glob
from scipy import ndimage
from skimage import filters
from skimage import transform
from collections import deque, defaultdict, Counter
from matplotlib.patches import Polygon as matplotlib_Polygon
from matplotlib.patches import Rectangle
from scipy.ndimage import binary_erosion, binary_dilation
from matplotlib.collections import PatchCollection
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy.ndimage import rotate
from datetime import datetime
from sklearn.linear_model import LinearRegression
from shapely.geometry import Polygon

import colour
import math
from scipy.optimize import curve_fit
import cv2


import warnings

def get_brightness(Y_dict:dict, fov_zones:dict) ->dict:

    '''
    Calculate average brightness for each image in Y_dict in the given fov_zone masks.
    Also calculate average, min, and max brightness across all images for each fov_zone mask.
    input: 
        Y_dict: dictionary of Y tristimulus image data from XYZ data.
                Dictionary can have any number of images. The dictionary keys can have any name
                {"pupil 1":np.ndarray, "pupil 2":np.ndarray, "pupil 3":np.ndarray...}
        fov_zones: dict of FOV zone masks {"zoneA":np.ndarray,"zoneB":np.ndarray,"zoneC":np.ndarray}
    output:
        Dictionary containing mean brightness for each image and each zone. Dictionary also
        contains mean, min, max for each zone across the images
        brightness_dict: dict of mean brightness in each FOV zone 
                            {
                            'zoneA pupil 1 luminance mean': float, 
                            'zoneA pupil 2 luminance mean': float, 
                            'zoneA pupil 3 luminance mean': float, 
                            'zoneA all pupil luminance mean': float, 
                            'zoneA all pupil luminance max': float, 
                            'zoneA all pupil luminance min': float,
                            'zoneB pupil 1 luminance mean': float, 
                            ...
                            }
    The function loads the fov_zone mask data, calculates the average luminance for each image in Y_dict, and calculates
    the average, min, and max across all images.
    The process repeats for each fov_zone mask and the results are returned as a dictionary.                                                             
    '''
    # Check if the variable is a dictionary with at least "region 0"
    if isinstance(Y_dict, dict):
        pass
    else:
        print('Y_dict is not a dictionary')
        return False
    #Validate Y_dict
    for key in Y_dict.keys():
        #Check if the key corresponds to a list or numpy array
        if not isinstance(Y_dict[key],np.ndarray):
            print(f"{key} is not a numpy array")
        #Check if input arrays have empty data
        if len(Y_dict[key]) == 0:
            print(f"Y_dict[key] is empty")
            return False
    #Check if zoneA in fov_zones
    if not "zoneA" in fov_zones:
        print("fov_zones is not a dictionary containing at least 'zoneA'")
        return False    
    #Validate fov_zones
    for zone in fov_zones.keys():
        #Check if fov_zone mask is a boolean numpy array
        if isinstance(fov_zones[zone],np.ndarray) and np.isin(fov_zones[zone], [True,False]).all():
            pass
        else:
            print(f"{zone} mask is not a boolean numpy array")
            return False
        #Check if fov_zone mask is same size as the Y_data arrays
        for Y_key in Y_dict.keys():
            #Check if the input array is of the correct dimensions
            if np.shape(Y_dict[Y_key]) != np.shape(fov_zones[zone]):
                print(f"{zone} mask is not the same shape as {Y_key}")

    #Create empty dictionary to store the brightness values for each FOV zone
    brightness_dict = {}
    
    #Iterate over each FOV zone in the fov_zone dictionary
    for fov_zone in fov_zones.keys():
        brightness_dict[fov_zone] = {}
        zone_averages = []
        #Iterate over each Y_image in Y_dict:
        for Y_key in Y_dict.keys():
            #Create new dictionary to store the brightness values for current FOV zone
            brightness_dict[fov_zone][Y_key] = {}
            
            #Use util.get_FOV_zone_data function to apply fov_zone mask to Y_image input data
            masked_brightness = util.get_FOV_zone_data(Y_dict[Y_key], fov_zones[fov_zone])
            
            #Calculate the mean brightness value for the masked data, ignoring nan values
            mean_masked_brightness = np.nanmean(masked_brightness)

            #Append mean brightness value to zone_averages
            zone_averages.append(mean_masked_brightness)
            
            #Store the mean brightness value for the current FOV zone
            brightness_dict[fov_zone][Y_key]["luminance mean"] = mean_masked_brightness
    
        #Add mean, min, and max of the zone_averages to the dictionary for the current zone.
        brightness_dict[fov_zone]["all pupil luminance mean"] = np.mean(zone_averages)
        brightness_dict[fov_zone]["all pupil luminance max"] = np.amax(zone_averages)
        brightness_dict[fov_zone]["all pupil luminance min"] = np.amin(zone_averages)

    #Flatten dictionary
    brightness_dict = util.flatten_dict(brightness_dict)

    #Return the brightness dictionary 
    return brightness_dict

def get_chromaticity(XYZ_dict:dict, fov_zones:dict) -> dict:
    '''
    Calculate the chromaticity in u'v' cie xy space for the given fov_zone masks.
    The u'v' and cie xy values are calculated using the average XYZ in the fov_zone.
    input: 
        XYZ_dict: dictionary of XYZ data numpy arrays  {"X":np.ndarray,"Y":np.ndarray,"Z":np.ndarray}
        fov_zones: dict of FOV zone masks {"zoneA":np.ndarray,"zoneB":np.ndarray,"zoneC":np.ndarray}
    output:
        chromaticity_dict: dict of numpy arrays for each zone containing the corresponding u' and v' values 
                           {"zoneA u'v'":np.array[float,float], "zoneA cxcy":np.array[float,float]
                            "zoneB u'v'":np.array[float,float], "zoneB cxcy":np.array[float,float]
                            ...
                            }
    The function loads the XYZ image data, applies the fov_zone mask, and calculates the mean XYZ values for the zone.
    The XYZ values are then converted to u'v'. The values for each zone are returned as arrays in a dictionary.                                                              
    '''

    #Check if the variable is a dictionary with keys "X", "Y", and "Z"
    if isinstance(XYZ_dict, dict) and all(key in XYZ_dict for key in ["X", "Y", "Z"]):
        pass
    else:
        print("The XYZ_dict not a dictionary with keys 'X', 'Y', and 'Z'.")
        return False
    #Validate XYZ_dict
    for key in XYZ_dict.keys():
        #Check if the input arrays are of the correct type
        if not isinstance(XYZ_dict[key], np.ndarray):
            print(f"{key} must be a numpy array")
            return False
        #Check if input arrays have empty data
        if XYZ_dict[key].shape[0] == 0:
            print(f"{key} is empty")
            return False
    #Validate fov_zones
    for zone in fov_zones.keys():
        #Check if fov_zone mask is a boolean numpy array
        if isinstance(fov_zones[zone],np.ndarray) and np.isin(fov_zones[zone], [True,False]).all():
            pass
        else:
            print(f"{zone} mask is not a boolean numpy array")
            return False
        #Check if fov_zone mask is same size as the Y_data arrays
        for key in XYZ_dict.keys():
            #Check if the input array is of the correct dimensions
            if np.shape(XYZ_dict[key]) != np.shape(fov_zones[zone]):
                print(f"{zone} mask is not the same shape as image")
                return False

    #Create empty dictionary to store the chromaticity values for each FOV zone
    chromaticity_dict = {}
    
    #Iterate over each FOV zone in the fov_zone dictionary
    for fov_zone in fov_zones.keys():
        #Create new dictionary to store the brightness values for current FOV zone
        chromaticity_dict[fov_zone] = {}

        #Get each XYZ image data from dictionary
        X_image = XYZ_dict["X"]
        Y_image = XYZ_dict["Y"]
        Z_image = XYZ_dict["Z"]

        #Get XYZ image data inside the current FOV zone
        masked_X_image = util.get_FOV_zone_data(X_image,fov_zones[fov_zone])
        masked_Y_image = util.get_FOV_zone_data(Y_image,fov_zones[fov_zone])
        masked_Z_image = util.get_FOV_zone_data(Z_image,fov_zones[fov_zone])

        #Get the mean XYZ values for the current FOV Zone
        X_mean = np.nanmean(masked_X_image)
        Y_mean = np.nanmean(masked_Y_image)
        Z_mean = np.nanmean(masked_Z_image)

        #Get the u' and v' for the current FOV Zone
        denominator = X_mean + 15*Y_mean + 3*Z_mean
        u_prime = 4*X_mean/denominator 
        v_prime = 9*Y_mean/denominator

        #Get the cie x and y for the current FOV Zone
        denominator = X_mean + Y_mean + Z_mean
        cx = X_mean/denominator
        cy = Y_mean/denominator

        #Store the u'v' and cxcy for the current FOV Zone
        chromaticity_dict[fov_zone]["u'v'"] = np.array([u_prime,v_prime])
        chromaticity_dict[fov_zone]["cxcy"] = np.array([cx,cy])

    chromaticity_dict = util.flatten_dict(chromaticity_dict)

    #Return the chromaticity dictionary 
    return chromaticity_dict

def get_white_point_error(up_vp_array, ref_white = "D65") -> float:
    '''
        Calculate the white point error for given u'v' values with respect to a given ref_white standard illuminant.
        input: 
            up_vp: numpy array containing numpy arrays with 2 float values. Must contain
                   at least one set of values. 
                    np.array([u',v'],[u',v'],[u',v'],...)
            ref_white: string for the standard illuminant. Must be "D65", "D55", "D50", "D75", or "D90"
        output:
            white_point_error: a 1-d array of delta u'v' values. One value for each input [u',v'] array

        The function calculates the white point error for the given u'v' values with respect to a standard illuminant
        using the standard Euclidean distance formula.                                                           
    '''

    ref_white_dict = {
                      "D65":[0.1978,.4683],
                      "D55":[0.2041,.4158],
                      "D50":[0.2092,.4881],
                      "D75":[0.29902,.31485],
                      "D93":[0.28315,.29711]
                     }

    #Validate input up_vp array
    if not isinstance(up_vp_array, np.ndarray):
        print("up_vp_array must be a numpy array containing numpy arrays with 2 float values")
        return False
    #Check if all elements in the array are numpy arrays with two float values
    for array in up_vp_array:
        if not (isinstance(array, np.ndarray) and array.size == 2 and np.issubdtype(array.dtype, np.floating)):
            print("up_vp_array must be a numpy array containing numpy arrays with 2 float values")
            return False
    #Validate input ref_white
    if ref_white not in ref_white_dict.keys():
        print('ref_white must be a value of "D65", "D55", "D50", "D75", or "D90"')

    #Create 1-d numpy array to store the delta u'v' values
    white_point_error_array = np.empty((len(up_vp_array[:,0])))

    #Calcualte the delta u'v' for each u' & v' data set in up_vp_array
    ref_u_prime = ref_white_dict[ref_white][0]
    ref_v_prime = ref_white_dict[ref_white][1]
    for i in range(len(up_vp_array[:,0])):
        u_prime = up_vp_array[i,0]
        v_prime = up_vp_array[i,1]
        white_point_error = (((u_prime - ref_u_prime)**2) + ((v_prime - ref_v_prime)**2))**0.5
        white_point_error_array[i] = white_point_error

    return white_point_error_array

def get_checkerboard_contrast(posIm: np.ndarray, 
                              negIm: np.ndarray, 
                              color: str,
                              analysis_config: dict, 
                              station_config: dict,
                              fov_zones: dict) -> dict:
    """
    This function calculates the contrast of the checkerboard pattern in two images (positive and negative).
    Inputs:
        posIm: numpy array containing the positive image
        negIm: numpy array containing the negative image
        analysis_config: analysis config dictionary
        station_config: station config dictionary
        fov_zones: dict of FOV zone masks {"zoneA":np.ndarray,"zoneB":np.ndarray,"zoneC":np.ndarray}
        
    Output:
        A dictionary containing the contrast data and summary metrics for each FOV zone mask passsed to the function.
        {
        'zoneA contrast mean':float, 'zoneA contrast max':float, 'zoneA contrast min':float, 'zoneA # < thresh':int, 'zoneA detected_checker_ratio':float, 
        'zoneB contrast mean':float, 'zoneB contrast max':float, 'zoneB contrast min':float, 'zoneB # < thresh':int, 'zoneB detected_checker_ratio':float, 
        ...
        }
    The function first loads the configuration file and extracts the relevant parameters. 
    It then calculates the average luminance of ROIs in the checkerboard pattern in both images. 
    It calculates the contrast between the positive and negative images for each FOV zone mask. 
    It finally returns a dictionary of contrast data and summary metrics for each zone.
    """

    #Set show_result = False to skip plotting 
    #Set show_result = True to plot checker ROIs and print data for debugging purposes
    show_result = False

    # Check if analysis_config is a dictionary
    if not isinstance(analysis_config, dict):
        print("analysis_config is not a dictionary")
        return False
    # Check analysis_config keys and their types
    analysis_keys = {
        'checkerboard_columns': int,
        'checkerboard_rows': int,
        'checker_width': int,
        'checker_height': int,
        'checkerboard_ROI_area_ratio': float,
        'save_image': str,
        'contrast_threshold': (int, float)
    }
    for key, expected_type in analysis_keys.items():
        if key not in analysis_config:
            print(f"Key '{key}' not found in analysis_config")
            return False
        if not isinstance(analysis_config[key], expected_type):
            print(f"Expected type {expected_type} for key '{key}', but got {type(analysis_config[key])}")
            return False
        if key == 'checkerboard_ROI_area_ratio' and not (0 <= analysis_config[key] < 1):
            print(f"Expected 'checkerboard_ROI_area_ratio' to be a float less than 1, but got {analysis_config[key]}")
            return False
        if key == 'save_image' and analysis_config[key] not in ["True", "False"]:
            print(f"Expected 'save_image' to be 'True' or 'False', but got {analysis_config[key]}")
            return False
    # Check if station_config is a dictionary
    if not isinstance(station_config, dict):
        print("station_config is not a dictionary")
        return False
    # Check station_config keys and their types
    if 'checkerboard_contrast_cal' not in station_config:
        print("'checkerboard_contrast_cal' not found in station_config")
        return False
    if not isinstance(station_config['checkerboard_contrast_cal'], dict):
        print(f"Expected type dict for 'checkerboard_contrast_cal', but got {type(station_config['checkerboard_contrast_cal'])}")
        return False
    #Check if the input arrays are of the correct type
    if not isinstance(posIm, np.ndarray):
        print("posIm must be a numpy array")
        return False
    if not isinstance(negIm, np.ndarray):
        print("negIm must be a numpy array")
        return False
    #Check if input arrays have equal dimensions
    if posIm.shape != negIm.shape:
        print("posIm and negIm must have equal dimensions")
        return False
    #Check if input arrays have empty data
    if posIm.shape[0] == 0 or negIm.shape[0] == 0:
        print("Data is empty")
        return False
    #Check if color is a string
    if not isinstance(color, str):
        print("color must be a string of 'W','R','G', or 'B'")
        return False
    # Check if the posIm or negIm is an RGB or grayscale image
    posIm = util.check_if_8bit_image(posIm)
    negIm = util.check_if_8bit_image(negIm)
    #Check if either array contains any zeros
    if np.any(posIm == 0) or np.any(negIm == 0):
        # Adding a very small number to both arrays
        posIm += 1e-10
        negIm += 1e-10
    #Check if color in RGBW
    if color not in ['W','R','G','B']:
        print("color must be a string of 'W','R','G', or 'B'")
        return False
    #Validate fov_zones
    for zone in fov_zones.keys():
        #Check if fov_zone mask is a boolean numpy array
        if isinstance(fov_zones[zone],np.ndarray) and np.isin(fov_zones[zone], [True,False]).all():
            pass
        else:
            print(f"{zone} mask is not a boolean numpy array")
            return False
        #Check if fov_zone mask is same size as the Y_data arrays
        for image in [posIm,negIm]:
            #Check if the input array is of the correct dimensions
            if np.shape(image) != np.shape(fov_zones[zone]):
                print(f"{zone} mask is not the same shape as image")
                return False

    fov_zones = copy.deepcopy(fov_zones)
    posIm = copy.deepcopy(posIm)
    negIm = copy.deepcopy(negIm)

    if show_result == True:
        print(analysis_config)
        print(f"posIm max {np.amax(posIm)}")
        print(f"negIm max {np.amax(negIm)}")
        plt.imshow(posIm, cmap='gray', origin='upper')
        plt.show()

    #Rotate array and fov_zones
    posIm = util.rotate_array(posIm,analysis_config["rotation_angle"])
    negIm = util.rotate_array(negIm,analysis_config["rotation_angle"])
    for fov_zone in fov_zones.keys(): 
        fov_zones[fov_zone] = util.rotate_array(fov_zones[fov_zone],analysis_config["rotation_angle"])

    if show_result == True:
        plt.imshow(posIm, cmap='gray', origin='upper')
        plt.show()

    #Read the analysis_config file to get checkerboard_columns, checkerboard_rows, checker_width, and checker_height
    checkerboard_columns = analysis_config["checkerboard_columns"]
    checkerboard_rows = analysis_config["checkerboard_rows"]
    checker_width = analysis_config["checker_width"]
    checker_height = analysis_config["checker_height"]  
        
    checker_points = util.get_checkerboard_points(posIm, checkerboard_columns*checkerboard_rows, checker_width,analysis_config) 

    #Calculate the ROI dimensions using checkerboard_ROI_area_ratio
    roi_width = int(checker_width * (analysis_config["checkerboard_ROI_area_ratio"]**0.5))
    roi_height = int(checker_height * (analysis_config["checkerboard_ROI_area_ratio"]**0.5))
        
    if show_result == True or analysis_config["save_image"] == "True":
        #Display the image using plt.imshow()
        plt.imshow(posIm, cmap='gray', origin='upper')
        #Plot filtered points on top of the image
        plt.plot(checker_points[:, 0], checker_points[:, 1], 'ro', markersize=.25)
        
    rois = []
        
    #Calculate offset of the center of the ROI from the detected checker point
    x_roi_offset = checker_width // 2
    y_roi_offset = checker_height // 2
    #Draw rectangular ROIs around each point
    for x, y in checker_points:
        #Calculate the corners of the ROIs
        #Check if x - roi_width is negative and x + roi_width outside range
        corners = [
            (x - x_roi_offset, y - y_roi_offset),  #Top-left
            (x + x_roi_offset, y - y_roi_offset),  #Top-right
            (x - x_roi_offset, y + y_roi_offset),  #Bottom-left
            (x + x_roi_offset, y + y_roi_offset)   #Bottom-right
        ]
        
        #Create roi with dimensions (x,y,roi_width,roi_height) and append to roi list
        for cx, cy in corners:
            #check if xy coordinates are out of range of image. If they are go to next point
            if cx > len(posIm[0,:]) or cx < 0: continue
            if cy > len(posIm[:,0]) or cy < 0: continue

            new_roi = (cx - roi_width / 2, cy - roi_height / 2, roi_width, roi_height)
            
            #Check for overlap with already-drawn ROIs
            if all(not util.is_overlapping(new_roi, existing_roi) for existing_roi in rois):

                #Draw the ROI rectangle
                if show_result == True or analysis_config["save_image"] == "True": 
                    rect = Rectangle((new_roi[0], new_roi[1]), new_roi[2], new_roi[3], linewidth=.25, edgecolor='g', facecolor='none')
                    plt.gca().add_patch(rect)
                        
                #Append the ROI info to rois
                rois.append(new_roi)
                
    if len(rois) == 0:
        print("No rois detected. Check data")
        return False

    #################################################################################################################################################################################################################
    #Get the posIm roi array with average luminance
    posIm_roi_avg_lum = util.get_checkerboard_avg_lum(posIm,rois)
    if show_result == True: print(f"average of posIm checkers {np.average(posIm_roi_avg_lum[:, 4])}")
    if show_result == True: print(f"max of posIm checkers {np.amax(posIm_roi_avg_lum[:, 4])}")
        
    #Get the posIm white checker rois
    posIm_roi_avg_lum = util.get_white_checkers(posIm_roi_avg_lum,checker_width*1.2)
    if show_result == True: print(f"average of posIm white checkers {np.average(posIm_roi_avg_lum[:, 4])}")
    if show_result == True: print(f"max of posIm white checkers {np.amax(posIm_roi_avg_lum[:, 4])}")
 
    #Draw the rois on the posIm plot if show_result or save_image is True
    if show_result == True or analysis_config["save_image"] == "True": 
        plt.clf()
        plt.imshow(posIm, cmap='gray', origin='upper')
        #Draw filtered white checker ROIs on top of the image
        for x_start, y_start, width, height, _ in posIm_roi_avg_lum:
            rect = Rectangle((x_start, y_start), width, height, linewidth=1, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)
        if show_result == True: plt.show()

    #Get the rois with avg luminance in the negIm using the white checker rois for posIm
    posIm_roi_avg_lum_inverse = util.get_checkerboard_avg_lum(negIm,posIm_roi_avg_lum[:,:4])
    if show_result == True: print(f"average of negIm checkers {np.average(posIm_roi_avg_lum_inverse[:, 4])}")
    if show_result == True: print(f"max of negIm checkers {np.amax(posIm_roi_avg_lum_inverse[:, 4])}")  
    #################################################################################################################################################################################################################

    #Get the negIm roi array with average luminance
    negIm_roi_avg_lum = util.get_checkerboard_avg_lum(negIm,rois)
    if show_result == True: print(f"average of negIm checkers {np.average(negIm_roi_avg_lum[:, 4])}")
    if show_result == True: print(f"max of negIm checkers {np.amax(negIm_roi_avg_lum[:, 4])}")
        
    #Get the negIm white checker rois
    negIm_roi_avg_lum = util.get_white_checkers(negIm_roi_avg_lum,checker_width*1.2)
    if show_result == True: print(f"average of negIm white checkers {np.average(negIm_roi_avg_lum[:, 4])}")
    if show_result == True: print(f"max of negIm white checkers {np.amax(negIm_roi_avg_lum[:, 4])}")
 
    #Display the image using plt.imshow()
    if show_result == True or analysis_config["save_image"] == "True": 
        plt.clf()
        plt.imshow(negIm, cmap='gray', origin='upper')
        #Draw filtered white checker ROIs on top of the image
        for x_start, y_start, width, height, _ in negIm_roi_avg_lum:
            rect = Rectangle((x_start, y_start), width, height, linewidth=1, edgecolor='g', facecolor='none')
            plt.gca().add_patch(rect)
        if show_result == True: plt.show()
        plt.clf()
    # #Save an image of the detected white checkers if save_image == True
    # if analysis_config["save_image"] == "True":
    #     path = "images"
    #     #Check if the directory exists, if not create it
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     else:
    #         #Get the date and time
    #         current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            
    #         #Define the path 
    #         save_path = os.path.join(path, f'checkerboard contrast {current_time}.jpg')
            
    #         #Save the image
    #         plt.savefig(save_path,dpi=400)
    #         print(f"Image saved at {save_path}")
    # if show_result == True: plt.show()

    #Get the rois with avg luminance in the negIm using the white checker rois for negIm
    negIm_roi_avg_lum_inverse = util.get_checkerboard_avg_lum(posIm,negIm_roi_avg_lum[:,:4])
    if show_result == True: print(f"average of negIm checkers {np.average(negIm_roi_avg_lum_inverse[:, 4])}")
    if show_result == True: print(f"max of negIm checkers {np.amax(negIm_roi_avg_lum_inverse[:, 4])}")  
    #################################################################################################################################################################################################################

    #Create dict for summary metrics
    summary_metrics = {}

    #Iterate through all fov zones, calculate summary metrics and add to data dict
    for zone in fov_zones.keys():
        mask = fov_zones[zone]

        #Concatenate white checker rois from posIm and white checker rois from negIm
        white_roi_avg_lum = np.concatenate((posIm_roi_avg_lum,negIm_roi_avg_lum), axis=0)
        #white_roi_avg_lum = negIm_roi_avg_lum
        #Concatenate black checker rois from negIm and black checker rois from posIm
        black_roi_avg_lum = np.concatenate((posIm_roi_avg_lum_inverse,negIm_roi_avg_lum_inverse),axis = 0)
        #black_roi_avg_lum = negIm_roi_avg_lum_inverse
        #Order of white and black checker rois with respect to each other in the two arrays should be maintained


        #Get masked data for current FOV zone. Add roi_width/2 & roi_height/2 from coordinates, since the ROI coordinates 
        #correspond to top left corner of ROI.
        white_roi_avg_lum[:,0] = white_roi_avg_lum[:,0] + roi_width/2
        white_roi_avg_lum[:,1] = white_roi_avg_lum[:,1] + roi_height/2
        black_roi_avg_lum[:,0] = black_roi_avg_lum[:,0] + roi_width/2
        black_roi_avg_lum[:,1] = black_roi_avg_lum[:,1] + roi_height/2
        masked_white_roi_avg_lum = util.get_FOV_zone_data(white_roi_avg_lum,fov_zones[zone])
        masked_black_roi_avg_lum = util.get_FOV_zone_data(black_roi_avg_lum,fov_zones[zone])

        if len(masked_white_roi_avg_lum) == 0 or len(masked_black_roi_avg_lum) == 0:
            summary_metrics[zone] = {}
            summary_metrics[zone]["contrast mean"] = np.nan
            summary_metrics[zone]["contrast max"] = np.nan
            summary_metrics[zone]["contrast min"] = np.nan
            summary_metrics[zone]["# < thresh"] = np.nan
            continue

        #Subtract back roi_width/2 and roi_height/2 to the masked ROI coordinates and original ROI coordinates
        masked_white_roi_avg_lum[:,0] = masked_white_roi_avg_lum[:,0] - roi_width/2
        masked_white_roi_avg_lum[:,1] = masked_white_roi_avg_lum[:,1] - roi_height/2
        masked_black_roi_avg_lum[:,0] = masked_black_roi_avg_lum[:,0] - roi_width/2
        masked_black_roi_avg_lum[:,1] = masked_black_roi_avg_lum[:,1] - roi_height/2
        white_roi_avg_lum[:,0] = white_roi_avg_lum[:,0] - roi_width/2
        white_roi_avg_lum[:,1] = white_roi_avg_lum[:,1] - roi_height/2
        black_roi_avg_lum[:,0] = black_roi_avg_lum[:,0] - roi_width/2
        black_roi_avg_lum[:,1] = black_roi_avg_lum[:,1] - roi_height/2

        #Calculate the checkerboard contrast array
        contrast_array = masked_white_roi_avg_lum
        contrast_array[:,4] = masked_white_roi_avg_lum[:,4]/masked_black_roi_avg_lum[:,4]

        #Compensate contrast_array result for camera checkerboard contrast calibration parameter
        camera_contrast = station_config["checkerboard_contrast_cal"][color]
        contrast_array[:,4] = 1/((1/(contrast_array[:,4])) - (1/(camera_contrast)))
        
        #Add contrast summary metrics to the current zone dict
        summary_metrics[zone] = {}
        summary_metrics[zone]["contrast mean"] = np.average(contrast_array[:,4])
        summary_metrics[zone]["contrast max"] = np.amax(contrast_array[:,4])
        summary_metrics[zone]["contrast min"] = np.amin(contrast_array[:,4])
        summary_metrics[zone]["# < thresh"] = np.sum(contrast_array[:,4] < analysis_config["contrast_threshold"])
        
        if show_result == True or analysis_config["save_image"] == "True":
            plt.cla() 
            plt.imshow(posIm, cmap='gray', origin='upper')
            #Draw filtered white checker ROIs within the False region on top of the image
            for x_start, y_start, width, height, _ in contrast_array:
                rect = Rectangle((x_start,y_start), width, height, linewidth=1, edgecolor='g', facecolor='none')
                plt.gca().add_patch(rect)

            #Get the date and time
            current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            #Define the path 
            path = "images"
            save_path = os.path.join(path, f'checkerboard contrast {zone} {current_time}.jpg')
            #Save the image
            plt.savefig(save_path,dpi=400)
            print(f"Image saved at {save_path}")
        
            if show_result == True: plt.show()

    #Add detected checker ratio to the summary_metrics dict
    summary_metrics["detected checker ratio"] = len(masked_white_roi_avg_lum)/(checkerboard_columns*checkerboard_rows)

    if show_result == True: 
        print(summary_metrics)
        
        colors = ["red","green","blue","orange"]
        i_color = 0
        for zone in fov_zones:
            #Dilate the region of interest to create an outline
            zone_outline = binary_dilation(fov_zones[zone]) ^ fov_zones[zone]

            #Plot the luminance array
            plt.imshow(posIm, cmap='gray', origin='upper')

            #Overlay the outline on the luminance array
            plt.contour(zone_outline, colors=colors[i_color], linewidths=2)
            i_color = i_color + 1
            if i_color > len(colors): i_color = 0

        #Show the plot
        plt.show() 
    
    #Flatten dictionary
    summary_metrics = util.flatten_dict(summary_metrics)

    return summary_metrics

def get_luminance_rolloff_2(Y_dict:dict, fov_zones) -> dict:
    """
    This function calculates the luminance rolloff for the Artemis definition:
    https://docs.google.com/document/d/1-elYMHJtEu5ccoo3tAg0e1tle-5l4-aMboWSyLzQAt0/edit

    This function calculates the luminance rolloff of eyebox region 0 measurements. 
    If data is passed for region 1, luminance rolloff is also calculated for region 1.
    Inputs:
        Y_dict: dictionary of lists containing Y data numpy arrays .
                Dictionary must contain at least contain "region 0". "region 0" and "region 1" can contain a single or multiple arrays for Y data
                        {
                        "region 0":[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]
                        "region 1":[np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray,np.ndarray]
                        }
            -Can be Y data from White, Red, Green, or Blue
        fov_zones: dict of FOV zone masks 
                   Dictionary must contain at least "zoneA"
                        {"zoneA":np.ndarray,"zoneB":np.ndarray,"zoneC":np.ndarray}
        
    Output:
        rolloff_dict: A dictionary of float values for the brightness rolloff for each region for each FOV zone
                      Dictionary will contain at least "region 0" and "zoneA" data. However "region 0" "zoneA" 
                      rolloff will always equal to 1 using the Artemis defintion.
        {
        'region 0 zone A Yrolloff':float,
        'region 0 zone B Yrolloff':float,
        'region 1 zone A Yrolloff':float,
        'region 1 zone B Yrolloff':float,
        }
    The function first loads the station and analyzer configuration files and extracts the relevant parameters. 
    The luminance rolloff is then calculated for region 0 using the Artemis definition:  
        1) The average luminance for zoneA in each region 0 EB positions is calculated.
        2) The average luminance for each zone in each region 0 EB positions is calculated.
        3) The luminance rolloff is for each zone is calculated by dividing the average for the zone by the average of zoneA.
    The luminance rolloff is then calculated for region 1 using the Artemis definition: 
        1) The average luminance for each zone in each region 0 EB positions is calculated.
        2) The average luminance for each zone in all EB positions is calculated.
        3) The luminance roll for each zone is calculated by dividing the average for all EB positions by average of region 0.
    """
    # Check if the variable is a dictionary with at least "region 0"
    if isinstance(Y_dict, dict) and "region 0" in Y_dict:
        pass
    else:
        print('Y_dict is not a dictionary containing at least "region 0"')
        return False
    #Validate Y_dict
    for key in Y_dict.keys():
        #Check if the key corresponds to a list or numpy array
        if not isinstance(Y_dict[key],(list, np.ndarray)):
            print(f"{Y_dict[key]} is neither a list nor numpy array")
        #Check if each array in the list is correct
        for array_index in range(len(Y_dict[key])):
            Y_image = Y_dict[key][array_index]
            #Check if the input array is of the correct type
            if not isinstance(Y_image, np.ndarray):
                print(f"Y_image must be a numpy array")
                return False
            #Check if input arrays have empty data
            if Y_image.shape[0] == 0:
                print(f"Y_image is empty")
                return False
    if not "zoneA" in fov_zones:
        print("fov_zones is not a dictionary containing at least 'zoneA'")
        return False
    #Validate fov_zones
    for zone in fov_zones.keys():
        #Check if fov_zone mask is a boolean numpy array
        if isinstance(fov_zones[zone],np.ndarray) and np.isin(fov_zones[zone], [True,False]).all():
            pass
        else:
            print(f"{zone} mask is not a boolean numpy array")
            return False
        #Check if fov_zone mask is same size as the Y_data arrays
        for Y_key in Y_dict.keys():
            for array_index in range(len(Y_dict[Y_key])):
                Y_image = Y_dict[Y_key][array_index]
                #Check if the input array is of the correct dimensions
                if np.shape(Y_image) != np.shape(fov_zones[zone]):
                    print(f"{zone} mask is not the same shape as Y_image")


    rolloff_dict = {}

    #Initialize an empty list to store the average luminance values for each zone in region 0
    region0_zoneA_averages = []
    
    #Loop through each array in Y_dict["region 0"]
    for array_index in range(len(Y_dict["region 0"])):
        #Extract the data for zoneA from the current array using the util.get_FOV_zone_data function
        Y_zoneA = util.get_FOV_zone_data(Y_dict["region 0"][array_index],fov_zones["zoneA"])
        
        #Calculate the average luminance value for the extracted data
        Y_zoneA_average = np.nanmean(Y_zoneA)
        
        #Append the average luminance value to the region0_zoneA_averages list
        region0_zoneA_averages.append(Y_zoneA_average)
    
    #Calculate the overall average luminance value for zoneA in region 0
    region0_zoneA_average = np.average(region0_zoneA_averages)
    
    #Create a dictionary to store the rolloff values for each zone in region 0
    rolloff_dict["region 0"] = {}
    
    #Loop through each zone in fov_zones
    for zone in fov_zones.keys():
        #Create a new dictionary to store the rolloff values for the current zone
        rolloff_dict["region 0"][zone] = {}
        
        #Initialize an empty list to store the average luminance values for the current zone
        region0_currentZone_averages = []
        
        #Loop through each array in Y_dict["region 0"]
        for array_index in range(len(Y_dict["region 0"])):
            #Extract the data for the current zone from the current array using the util.get_FOV_zone_data function
            Y_currentZone = util.get_FOV_zone_data(Y_dict["region 0"][array_index],fov_zones[zone])
            
            #Calculate the average luminance value for the extracted data
            Y_currentZone_average = np.nanmean(Y_currentZone)
            
            #Append the average luminance value to the region0_currentZone_averages list
            region0_currentZone_averages.append(Y_currentZone_average)
        
        #Calculate the overall average luminance value for the current zone
        region0_currentZone_average = np.average(region0_currentZone_averages)
        
        #Calculate the rolloff value for the current zone by dividing its average luminance value by the average luminance value for zoneA
        region0_currentZone_rolloff = (region0_currentZone_average/region0_zoneA_average)*100
        
        #Store the rolloff value in the rolloff_dict["region 0"][zone] dictionary
        rolloff_dict["region 0"][zone]["Yrolloff"] = region0_currentZone_rolloff
    
    #Check if "region 1" is present in Y_dict
    if "region 1" in Y_dict:
        #Create a dictionary to store the rolloff values for each zone in region 1
        rolloff_dict["region 1"] = {}
        
        #Loop through each zone in fov_zones
        for zone in fov_zones.keys():
            #Create a new dictionary to store the rolloff values for the current zone
            rolloff_dict["region 1"][zone] = {}
            
            #Initialize two empty lists to store the average luminance values for the current zone in regions 0 and 1
            region0_currentZone_averages = []
            region1_currentZone_averages = []

            #Iterate through each element in the "region 0" array of Y_dict
            for array_index in range(len(Y_dict["region 0"])):
                #Get the data for the current zone in the FOV using the util.get_FOV_zone_data function
                Y_currentZone = util.get_FOV_zone_data(Y_dict["region 0"][array_index],fov_zones[zone])
                
                #Calculate the average value of the current zone data
                Y_currentZone_average = np.nanmean(Y_currentZone)
                
                #Append the average value to the region0_currentZone_averages list
                region0_currentZone_averages.append(Y_currentZone_average)
            #Calculate the average of all the averages in the region0_currentZone_averages list
            region0_currentZone_average = np.average(region0_currentZone_averages)
            #Iterate through each element in the "region 1" array of Y_dict

            for array_index in range(len(Y_dict["region 1"])):
                #Get the data for the current zone in the FOV using the util.get_FOV_zone_data function
                Y_currentZone = util.get_FOV_zone_data(Y_dict["region 1"][array_index],fov_zones[zone])
                
                #Calculate the average value of the current zone data
                Y_currentZone_average = np.nanmean(Y_currentZone)
                
                #Append the average value to the region1_currentZone_averages list
                region1_currentZone_averages.append(Y_currentZone_average)

            #Calculate the average of all the averages in the region1_currentZone_averages list
            region1_currentZone_average = np.average(region1_currentZone_averages)
            #Calculate for region 1
            region1_currentZone_rolloff = (region1_currentZone_average/region0_currentZone_average)*100
            #Add the region 1 rolloff value at the current zone to the dictionary
            rolloff_dict["region 1"][zone]["Yrolloff"] = region1_currentZone_rolloff

    #Flatten dictionary
    rolloff_dict = util.flatten_dict(rolloff_dict)

    #Return the results dictionary and rolloff table
    return rolloff_dict #rolloffTable

def get_luminance_rolloff_1(XYZ_dict:dict, 
                          analysis_config:dict, 
                          station_config:dict,
                          fov_zones:dict
                          ) -> dict:
    """
    This function calculates the luminance rolloff for definition 1.
    The function calculates the luminance rolloff of a single image measurement (data as XYZ arrays).
    Inputs:
        XYZ_dict: dictionary of XYZ data numpy arrays  {"X":np.ndarray,"Y":np.ndarray,"Z":np.ndarray}
            -Can be XYZ from White, Red, Green, or Blue
        analysis_config: analysis config dictionary
        station_config: station config dictionary
        
    Output:
        rolloff_dict: A dictionary of float values for the brightness rolloff for each FOV zone
        {
        'Yrolloff_ZoneA': float, 
        'Yrolloff_ZoneB': float, 
        ...
        }
       
    The function first loads the station and analyzer configuration files and extracts the relevant parameters. 
    It then calls functions from uniformity_metrics to calculate the luminance rolloff and return results as dict.
    """
    # Check analysis_config keys and their types
    analysis_keys = {
        'freqLB': list,
        'freqMB': list,
        'freqHB': list
    }
    # Check if analysis_config is a dictionary
    if not isinstance(analysis_config, dict):
        print("analysis_config is not a dictionary")
        return False
    for key, expected_type in analysis_keys.items():
        if key not in analysis_config:
            print(f"Key '{key}' not found in analysis_config")
            return False
        if not isinstance(analysis_config[key], expected_type) or len(analysis_config[key]) != 2:
            print(f"Expected a list of two elements for key '{key}',\
                  but got {type(analysis_config[key])} with length {len(analysis_config[key])}")
            return False
    # Check if station_config is a dictionary
    if not isinstance(station_config, dict):
        print("station_config is not a dictionary")
        return False
    # Check station_config keys and their types
    station_keys = {
        'vFoV': (int, float),
        'hFoV': (int, float),
        'zoneA_params': list,
        'zoneB_params': list,
        'zoneC_params': list
    }
    for key, expected_type in station_keys.items():
        if key not in station_config:
            print(f"Key '{key}' not found in station_config")
            return False
        if not isinstance(station_config[key], expected_type):
            print(f"Expected type {expected_type} for key '{key}', but got {type(station_config[key])}")
            return False
        if key in ['zoneA_params', 'zoneB_params', 'zoneC_params'] and len(station_config[key]) != 4:
            print(f"Expected a list of four elements for key '{key}',\
                  but got {type(station_config[key])} with length {len(station_config[key])}")
            return False
    #Check if XYZ_dict is a dictionary with keys "X", "Y", and "Z"
    if isinstance(XYZ_dict, dict) and all(key in XYZ_dict for key in ["X", "Y", "Z"]):
        pass
    else:
        print("The XYZ_dict not a dictionary with keys 'X', 'Y', and 'Z'.")
        return False
    #Validate XYZ_dict
    for key in XYZ_dict.keys():
        #Check if the input arrays are of the correct type
        if not isinstance(XYZ_dict[key], np.ndarray):
            print(f"{key} must be a numpy array")
            return False
        #Check if input arrays have empty data
        if XYZ_dict[key].shape[0] == 0:
            print(f"{key} is empty")
            return False
        
    # Extract X, Y, and Z values from the input dictionary
    X = XYZ_dict['X']
    Y = XYZ_dict['Y']
    Z = XYZ_dict['Z']

    # Generate parameters for image processing
    params = util.gen_params(
                        vFoV = station_config["vFoV"],
                        hFoV = station_config["hFoV"],
                        zoneA_params = station_config["zoneA_params"],
                        zoneB_params = station_config["zoneB_params"],
                        zoneC_params = station_config["zoneC_params"],
                        filter_params = {"filter_params": {  
                                        "freqLB": 1 / np.array(analysis_config["freqLB"]),
                                        "freqMB": 1 / np.array(analysis_config["freqMB"]),
                                        "freqHB": 1 / np.array(analysis_config["freqHB"])
                                    }}
                        )

    #Stack X, Y, and Z values into a 3D array along the third axis
    imageXYZ = np.stack([X, Y, Z], axis=2)

    #Create an empty dictionary to store image processing results
    imageResults = {}

    #Add a small value to each pixel in the image to ensure all values are positive
    imageXYZ = imageXYZ + np.max(imageXYZ) * 1e-5

    #Define pixels per degree (ppd) and field of view (FoV) angles
    ppd = 1/ math.degrees(math.atan(station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
    hFoV = station_config["hFoV"]
    vFoV = station_config["vFoV"]

    #Resize and filter the image
    imageXYZ_resized = np.zeros([int(vFoV * ppd), int(hFoV * ppd), 3])
    for i in range(3):
        imageXYZ_resized[:, :, i] = resize_and_filter_img(
            imageXYZ[:, :, i], 1, int(hFoV * ppd), int(vFoV * ppd), False, False
        )
    imageXYZ = imageXYZ_resized

    #Resize zones to the dimensions of the resized image
    rows = len(imageXYZ[:,0,0])
    columns = len(imageXYZ[0,:,0])
    for zone in fov_zones.keys():
        zone_mask = fov_zones[zone]
        # Resize the array using interpolation
        resized_array = ndimage.zoom(zone_mask.astype(float), (rows/zone_mask.shape[0], columns/zone_mask.shape[1]), order=1)
        # Threshold the array back to boolean
        resized_array = resized_array > 0.5
        fov_zones[zone] = resized_array

    #Combine the zones into a single 3D array
    fov_zone_list = []  
    fov_zone_names = []  
    for key, value in fov_zones.items():  
        fov_zone_names.append(key) 
        fov_zone_list.append(value)  
    zoneMatrix = np.dstack(fov_zone_list)

    # Normalize by the lumninance, whose index=1 for array of (X,Y,Z)
    #Normalize the image by dividing by the mean luminance in zone A
    zoneA = fov_zones["zoneA"]
    imageXYZ /= np.mean(np.ma.MaskedArray(imageXYZ[:, :, 1], mask=zoneA).compressed())

    #Convert the image from XYZ color space to Lab color space
    imageLab = xyz2lab(imageXYZ)

    #Create a boolean mask for the full FOV for computer_baseline_shape
    ffov = np.array(np.zeros(np.shape(zoneA)), dtype=bool)

    #Compute the baseline shape of the image
    LabBaseline, baselineCoeffs, baselineMaxInds = compute_baseline_shape(imageLab, params, ffov)

    #Set the baseline values for channels 1 and 2 to zero
    for i in [1, 2]:
        LabBaseline[:, :, i] = np.zeros(LabBaseline[:, :, i].shape)

    #Convert the baseline image from Lab color space to XYZ color space
    imageXYZBaseline = lab2xyz(np.copy(LabBaseline))

    #Compute the rolloff table for each zone
    rolloffTable = baselineRolloffXYZ(
        imageXYZ, imageXYZBaseline[:, :, 1], zoneMatrix, fov_zone_names, normZone=zoneA
    )

    #Convert rolloffTable to dict
    rolloff_dict = rolloffTable.iloc[0].to_dict()

    #Flatten dictionary
    rolloff_dict = util.flatten_dict(rolloff_dict)

    #Return the results dictionary and rolloff table
    return rolloff_dict #rolloffTable

def get_color_uniformity_2(image_dict:dict, 
                          analysis_config:dict, 
                          station_config:dict,
                          fov_zones:dict) -> pd.DataFrame:
    """
    Color uniformity Son Nguyen's implementation.   
    This function calculates the color uniformity of a white image measurement.
    The function can process a single measurement dataset, or multiple measurements passed to image_dict.
    A dataframe of color uniformity metrics is returned with a row for each input image.
    Inputs:
        image_dict: dictionary containing 3 dimensional arrays containing XYZ data.
                    The image names are declared in the image_dict key.
                    Each key will contain a single 3 dimensional XYZ data array.
                    The XYZ data array is arranged as: array[:,:,0] = X, array[:,:,1] = Y, array[:,:,2] = Z
                    Example:
                        {
                        "image1":np.ndarray,
                        "image2":np.ndarray ...
                        }
        analysis_config: dict
        station_config: dict
        fov_zones: dict of FOV zone masks 
                   Dictionary must contain all zones 
                        {"zoneA":np.ndarray,"zoneB":np.ndarray,"zoneC":np.ndarray, "full FOV":np.ndarray}

    Output:
        results_df: A dataframe containing summary metrics for color uniformity for each image
        image      | metric |   zoneA_RMS  |   zoneA_max  |   zoneA_P95 ... 
        str        | str    |   float      |   float      |   float     ...

    The function first loads the station and analyzer configuration files and extracts the relevant parameters. 
    compute_metrics() from wuc_uniformity_metrics.py is used to get uniformity metrics.
    The function loops through every measurement dataset passed to image_dict and returns each result in a dataframe.
    """

    # Check if analysis_config is a dictionary
    if not isinstance(analysis_config, dict):
        print("analysis_config is not a dictionary")
        return False
    # Check analysis_config keys and their types
    analysis_keys = {
        'freqLB': list,
        'freqMB': list,
        'freqHB': list,
        'rotation_angle': (float, int)
    }
    for key, expected_type in analysis_keys.items():
        if key not in analysis_config:
            print(f"Key '{key}' not found in analysis_config")
            return False
        if isinstance(expected_type, tuple):
            if not any(isinstance(analysis_config[key], t) for t in expected_type):
                print(f"Expected a numerical value (float or int) for key '{key}', but got {type(analysis_config[key])}")
                return False
        else:
            if not isinstance(analysis_config[key], expected_type) or len(analysis_config[key]) != 2:
                print(f"Expected a list of two elements for key '{key}', but got {type(analysis_config[key])} with length {len(analysis_config[key])}")
                return False
            
    # Check if station_config is a  dictionary
    if not isinstance(station_config, dict):
        print("station_config is not a dictionary")
        return False
    # Check station_config keys and their types
    station_keys = {
        #'ppd': float
    }
    for key, expected_type in station_keys.items():
        if key not in station_config:
            print(f"Key '{key}' not found in station_config")
            return False
        if not isinstance(station_config[key], expected_type):
            print(f"Expected type {expected_type} for key '{key}', but got {type(station_config[key])}")
            return False
    #Validate image_dict
    for key in image_dict.keys():
        #Check if the current key corresponds to a 3-dimensional numpy array
        if isinstance(image_dict[key], np.ndarray):
            pass
        else:
            print(f"{key} must be a 3-dimensional numpy array")
            return False
        #Check if input arrays have empty data
        if np.shape(image_dict[key]) == 0:
            print(f"{key} is empty")
            return False
    for zone in ["zoneA","zoneB","zoneC","full FOV"]:
        if not zone in fov_zones:
            print(f'fov_zones is not a dictionary containing at least "{zone}"')
            return False
    #Validate fov_zones
    for zone in fov_zones.keys():
        #Check if fov_zone mask is a boolean numpy array
        if isinstance(fov_zones[zone],np.ndarray) and np.isin(fov_zones[zone], [True,False]).all():
            pass
        else:
            print(f"{zone} mask is not a boolean numpy array")
            return False
        #Check if fov_zone mask is same size as the image_dict arrays
        for key in image_dict.keys():
            #Check if the input array is of the correct dimensions
            array_shape = np.shape(image_dict[key][:,:,0])
            if array_shape != np.shape(fov_zones[zone]):
                print(f"{zone} mask is not the same shape as image")
                return False

    image_dict = copy.deepcopy(image_dict)
    fov_zones = copy.deepcopy(fov_zones)

    #Rotate fov_zones by "rotation_angle" 
    for fov_zone in fov_zones.keys(): 
        fov_zones[fov_zone] = util.rotate_array(fov_zones[fov_zone],analysis_config["rotation_angle"])

    #Crop masks to the area defined by the "full FOV" mask
    full_fov_mask = fov_zones["full FOV"]
    # Find rows and columns where not all values are True
    rows = np.any(~full_fov_mask, axis=1)
    cols = np.any(~full_fov_mask, axis=0)
    # Find the boundaries of the rectangle
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    for key in fov_zones.keys():
        fov_zones[key] = fov_zones[key][rmin:rmax+1, cmin:cmax+1]

    result_df = {}
    image_number = 0
    for image_key in image_dict.keys():

        imageXYZ = image_dict[image_key]

        #Rotate imageXYZ by rotation_angle
        imageXYZ = util.rotate_array(imageXYZ,analysis_config["rotation_angle"])

        #Crop imageXYZ to the area defined by the "full FOV" mask
        imageXYZ = imageXYZ[rmin:rmax+1, cmin:cmax+1, :]

        #Define pixels per degree (ppd) and field of view (FoV) angles
        ppd = 1/ math.degrees(math.atan(station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
        hFoV = len(imageXYZ[0,:,0])*(1/ppd)
        vFov = len(imageXYZ[:,0,0])*(1/ppd)

        # Generate parameters for image processing
        params = util.gen_params(
                            vFoV = vFov,
                            hFoV = hFoV,
                            filter_params = {"filter_params": {  
                                            "freqLB": np.array(analysis_config["freqLB"]),
                                            "freqMB": np.array(analysis_config["freqMB"]),
                                            "freqHB": np.array(analysis_config["freqHB"])
                                        }}
                            )
        
        #Store the fov_zones in the params dict
        params["zoneA"] = fov_zones["zoneA"]
        params["zoneB"] = fov_zones["zoneB"]
        params["zoneC"] = fov_zones["zoneC"] 
        params["chrom_metric_ppd"] = ppd

        current_image_dict  = compute_metric(imageXYZ, params, 'right', True)
        # Convert the dictionary to a dataframe and add the image_key as a column
        current_image_df = pd.DataFrame.from_records([current_image_dict]).T
        current_image_df.reset_index(inplace=True)
        current_image_df.columns = ['metric', 'value']
        # Normalize the 'value' column into separate columns
        value_df = pd.json_normalize(current_image_df['value'])
        # Concatenate the original dataframe with the new 'value' dataframe
        current_image_df = pd.concat([current_image_df.drop('value', axis=1), value_df], axis=1)
        current_image_df.insert(0, 'image', image_key)
        if image_number == 0:
            result_df = current_image_df
        else:
            result_df = pd.concat([result_df,current_image_df],ignore_index=True)
        image_number = image_number + 1

    return result_df


def get_color_uniformity_1(image_dict:dict, 
                          analysis_config:dict, 
                          station_config:dict,
                          fov_zones:dict) -> pd.DataFrame:
    """
    Color uniformity Simon Swifter's implementation.
    This function calculates the color uniformity of a white image measurement, or a combined
    red+green+blue three image measurement (data as XYZ arrays for each color).  The function can process 
    a single measurement dataset, or multiple measurements passed to image_dict.
    Inputs:
        image_dict: dictionary of lists containing 3 dimensional arrays containing XYZ data.
                   The image names are declared in the image_dict key.
                   For white image case, the list contains a single 3 dimensional XYZ data array.
                   For the r+g+b case, the lists contains three XYZ data arrays in the order r,g,b.
                   The XYZ data array is arranged as: array[:,:,0] = X, array[:,:,1] = Y, array[:,:,2] = Z
                   white image case example:
                        {
                         "image1":[np.ndarray],
                         "image2":[np.ndarray] ...
                         }
                    r+g+b image case example:
                        {
                         "image1":[np.ndarray,np.ndarray,np.ndarray],
                         "image2":[np.ndarray,np.ndarray,np.ndarray] ...
                         }  
        analysis_config: dict
        station_config: dict
        fov_zones: dict of FOV zone masks 
                   Dictionary must contain all zones 
                        {"zoneA":np.ndarray,"zoneB":np.ndarray,"zoneC":np.ndarray, "full FOV":np.ndarray}
        
    Output:
        results_df: A dataframe containing summary metrics for color uniformity for each image. One row for 
                      each image.
        EyeboxLoc  |   dL_Global_ZoneA_RMS  |   dL_Global_ZoneA_max  |   dL_Global_ZoneA_min ... 
        str        |   float                |   float                |   float               ...

    The function first loads the station and analyzer configuration files and extracts the relevant parameters. 
    whiteXYZ_IQ() from uniformity_metrics.py is used to get uniformity metrics for the white image case.
    wg_synethetic_correction() from uniformity_metrics.py is used for the red+green+blue three image case.
    The function loops through every measurement dataset passed to image_dict and returns each result in a dataframe.
    """
    # Check if analysis_config is a dictionary
    if not isinstance(analysis_config, dict):
        print("analysis_config is not a dictionary")
        return False
    # Check analysis_config keys and their types
    analysis_keys = {
        'freqLB': list,
        'freqMB': list,
        'freqHB': list,
        'rotation_angle': (float, int)
    }
    for key, expected_type in analysis_keys.items():
        if key not in analysis_config:
            print(f"Key '{key}' not found in analysis_config")
            return False
        if isinstance(expected_type, tuple):
            if not any(isinstance(analysis_config[key], t) for t in expected_type):
                print(f"Expected a numerical value (float or int) for key '{key}', but got {type(analysis_config[key])}")
                return False
        else:
            if not isinstance(analysis_config[key], expected_type) or len(analysis_config[key]) != 2:
                print(f"Expected a list of two elements for key '{key}', but got {type(analysis_config[key])} with length {len(analysis_config[key])}")
                return False
            
    # Check if station_config is a  dictionary
    if not isinstance(station_config, dict):
        print("station_config is not a dictionary")
        return False
    # Check station_config keys and their types
    station_keys = {
        #'ppd': float
    }
    for key, expected_type in station_keys.items():
        if key not in station_config:
            print(f"Key '{key}' not found in station_config")
            return False
        if not isinstance(station_config[key], expected_type):
            print(f"Expected type {expected_type} for key '{key}', but got {type(station_config[key])}")
            return False
    #Validate image_dict
    for key in image_dict.keys():
        #Check if the current key corresponds to a list of size 1 or 3
        if isinstance(image_dict[key], list) and (len(image_dict[key]) == 1 or len(image_dict[key]) == 3):
            pass
        else:
            print(f"{key} must be a list of size 1 or 3")
            return False
        #Validate the XYZ arrays in the list
        for XYZ_array in image_dict[key]:
            if not isinstance(XYZ_array, np.ndarray):
                print(f"XYZ_array must be a 3 dimensional numpy array")
                return False
            #Check if input arrays have empty data
            if np.shape(XYZ_array) == 0:
                print(f"{key} is empty")
                return False
            #Check if input arrays are all the same dimensions
            array_shape = np.shape(image_dict[key][0])
            if np.shape(XYZ_array) != array_shape:
                print("All XYZ_arrays must be the same dimensions")
                return False
    for zone in ["zoneA","zoneB","zoneC","full FOV"]:
        if not zone in fov_zones:
            print(f'fov_zones is not a dictionary containing at least "{zone}"')
            return False
    #Validate fov_zones
    for zone in fov_zones.keys():
        #Check if fov_zone mask is a boolean numpy array
        if isinstance(fov_zones[zone],np.ndarray) and np.isin(fov_zones[zone], [True,False]).all():
            pass
        else:
            print(f"{zone} mask is not a boolean numpy array")
            return False
        #Check if fov_zone mask is same size as the image_dict arrays
        for key in image_dict.keys():
            #Check if the input array is of the correct dimensions
            array_shape = np.shape(image_dict[key][0][:,:,0])
            if array_shape != np.shape(fov_zones[zone]):
                print(f"{zone} mask is not the same shape as image")
                return False

    image_dict = copy.deepcopy(image_dict)
    fov_zones = copy.deepcopy(fov_zones)

    #Rotate fov_zones by "rotation_angle" 
    for fov_zone in fov_zones.keys(): 
        fov_zones[fov_zone] = util.rotate_array(fov_zones[fov_zone],analysis_config["rotation_angle"])

    #Crop masks to the area defined by the "full FOV" mask
    full_fov_mask = fov_zones["full FOV"]
    # Find rows and columns where not all values are True
    rows = np.any(~full_fov_mask, axis=1)
    cols = np.any(~full_fov_mask, axis=0)
    # Find the boundaries of the rectangle
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    for key in fov_zones.keys():
        fov_zones[key] = fov_zones[key][rmin:rmax+1, cmin:cmax+1]

    result_df = {}
    image_number = 0
    for image_key in image_dict.keys():
        
        #Check if the analysis is for the white image case
        if len(image_dict[image_key]) == 1:

            imageXYZ = image_dict[image_key][0]

            #Rotate imageXYZ by rotation_angle
            imageXYZ = util.rotate_array(imageXYZ,analysis_config["rotation_angle"])

            #Crop imageXYZ to the area defined by the "full FOV" mask
            imageXYZ = imageXYZ[rmin:rmax+1, cmin:cmax+1, :]

            #Define pixels per degree (ppd) and field of view (FoV) angles
            ppd = 1/ math.degrees(math.atan(station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            hFoV = len(imageXYZ[0,:,0])*(1/ppd)
            vFov = len(imageXYZ[:,0,0])*(1/ppd)

            # Generate parameters for image processing
            params = util.gen_params(
                                vFoV = vFov,
                                hFoV = hFoV,
                                filter_params = {"filter_params": {                                                
                                                "freqLB": np.array(analysis_config["freqLB"]),
                                                "freqMB": np.array(analysis_config["freqMB"]),
                                                "freqHB": np.array(analysis_config["freqHB"])
                                            }}
                                )
            
            eyebox_location_names = [image_key]

            #Store the fov_zones in the params dict
            params["zoneA"] = fov_zones["zoneA"]
            params["zoneB"] = fov_zones["zoneB"]
            params["zoneC"] = fov_zones["zoneC"] 
            params["chrom_metric_ppd"] = ppd

            current_image_df, results_dict, new_image_dict = whiteXYZ_IQ(
                                                                    xyz_white_data = {f"{image_key}":imageXYZ},
                                                                    eyebox_location_names = eyebox_location_names,
                                                                    params = params,
                                                                    compute_results_dict = True,
                                                                    compute_image_dict = True,
                                                                    multiprocess = False,
                                                                )
            
            # current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            # np.savetxt(f'images\{image_key} DE_00_Global {current_time}.csv', new_image_dict[image_key]["dE_00_Global"], delimiter=',')    
            # np.savetxt(f'images\{image_key} DE_00_Local {current_time}.csv', new_image_dict[image_key]["dE_00_Local"], delimiter=',')   
            # np.savetxt(f'images\{image_key} DE_00_freqLB {current_time}.csv', new_image_dict[image_key]["dE_00_freqLB"], delimiter=',')  
            # np.savetxt(f'images\{image_key} DE_00_freqMB {current_time}.csv', new_image_dict[image_key]["dE_00_freqMB"], delimiter=',')  
            # np.savetxt(f'images\{image_key} DE_00_freqHB {current_time}.csv', new_image_dict[image_key]["dE_00_freqHB"], delimiter=',')  

        #Check if the analysis is for the R+G+B image case
        if len(image_dict[image_key]) == 3:

            #Rotate image_data by rotation_angle
            image_dict[image_key][0] = util.rotate_array(image_dict[image_key][0],analysis_config["rotation_angle"])
            image_dict[image_key][1] = util.rotate_array(image_dict[image_key][1],analysis_config["rotation_angle"])
            image_dict[image_key][2] = util.rotate_array(image_dict[image_key][2],analysis_config["rotation_angle"])

            #Crop image data to the area defined by the "full FOV" mask
            image_dict[image_key][0] = image_dict[image_key][0][rmin:rmax+1, cmin:cmax+1, :]
            image_dict[image_key][1] = image_dict[image_key][1][rmin:rmax+1, cmin:cmax+1, :]
            image_dict[image_key][2] = image_dict[image_key][2][rmin:rmax+1, cmin:cmax+1, :]

            RGB_dict = {}
            RGB_dict[f"{image_key}_rXYZ"] = image_dict[image_key][0]
            RGB_dict[f"{image_key}_gXYZ"] = image_dict[image_key][1]
            RGB_dict[f"{image_key}_bXYZ"] = image_dict[image_key][2]

            #Define pixels per degree (ppd) and field of view (FoV) angles
            ppd = 1/ math.degrees(math.atan(station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            hFoV = len(RGB_dict[f"{image_key}_rXYZ"][0,:,0])*(1/ppd)
            vFov = len(RGB_dict[f"{image_key}_rXYZ"][:,0,0])*(1/ppd)

            #Generate parameters for image processing
            params = util.gen_params(
                                vFoV = vFov,
                                hFoV = hFoV,
                                filter_params = {"filter_params": {         
                                                "freqLB": np.array(analysis_config["freqLB"]),
                                                "freqMB": np.array(analysis_config["freqMB"]),
                                                "freqHB": np.array(analysis_config["freqHB"])
                                            }}
                                )

            eyebox_location_names = [image_key]

            #Store the fov_zones in the params dict
            params["zoneA"] = fov_zones["zoneA"]
            params["zoneB"] = fov_zones["zoneB"]
            params["zoneC"] = fov_zones["zoneC"] 
            params["chrom_metric_ppd"] = ppd

            current_image_df, results_dict, new_image_dict = wg_synthetic_correction(
                                                                    run_or_raw_data = RGB_dict,
                                                                    eyebox_location_names = eyebox_location_names,
                                                                    params = params,
                                                                    compute_results_dict = True,
                                                                    compute_image_dict = True,
                                                                    multiprocess = False,
                                                                ) 

        if image_number == 0:
            result_df = current_image_df
        else:
            result_df = pd.concat([result_df,current_image_df],ignore_index=True)

        image_number = image_number + 1 

 
    # Return the results dictionary 
    return result_df

def get_ghost(image_dict:dict,
              eye_side:str,
              analysis_config:dict) -> dict:
    
    """
    This function calculates the coordinates of a source pattern and ghost pattern from the input image.
    The strength of the ghost is also caclulated.
    Inputs:
        image_dict: a dictionary of image data. Image data can be in the format of grayscale (0-255 integer)
                    or luminance (float values). The image data must be a 2-dimensional array
                    The keys of the dictionary can have any name.
                    {"image_name_1":np.ndarray,
                     "image_name_2":np.ndarray}
        eye_side: a string specifying which eye the measurement was taken. "left" or "right". The image
                  will be flipped is "right" is specified.
        analysis_config: dictionary with analysis config parameters
        
    Output:
        A dictionary containing the coordinates of the source pattern and ghost pattern, the luminance of 
        source pattern and ghost pattern, and the ghost strength. If the number of squares detected is not 2,
        then an error message ('ErrMsg') will be stored in the dictionary.
        {
        'ghost_image_1 Source_LocX':float, 'ghost_image_1 Source_LocY':float, 'ghost_image_1 Ghost_LocX':float, 'ghost_image_1 Ghost_LocY':int, 
        'ghost_image_1 Source_Lum':float, 'ghost_image_1 Ghost_Lum':float, 'ghost_image_1 Ghost_Strength_Perc':float, 
        'ghost_image_1 Source_LocX':float, ...
        ...
        }
    The function first loads the configuration file and extracts the relevant parameters. 
    It then finds the location of the source and ghost square patterns by finding their contours.
    It calculates the average luminance of the source and ghost squares then calculates the ghost strength.
    The results are returned in a dictionary.
    """

    # Check if analysis_config is a dictionary
    if not isinstance(analysis_config, dict):
        print("analysis_config is not a dictionary")
        return False
    # Check analysis_config keys and their types
    analysis_keys = {
        'Ghost_CropRect': list,
        'Source_Size': (int, float),
        'save_image': str
    }
    for key, expected_type in analysis_keys.items():
        if key not in analysis_config:
            print(f"Key '{key}' not found in analysis_config")
            return False
        if not isinstance(analysis_config[key], expected_type):
            print(f"Expected type {expected_type} for key '{key}', but got {type(analysis_config[key])}")
            return False
        if key == 'save_image' and analysis_config[key] not in ["True", "False"]:
            print(f"Expected 'save_image' to be 'True' or 'False', but got {analysis_config[key]}")
            return False
        if key == 'Ghost_CropRect' and len(analysis_config[key]) != 4:
            print(f"Expected a list of four elements for key '{key}', but got {type(analysis_config[key])} with length {len(analysis_config[key])}")
            return False    
    #Validate eye_side input
    if not isinstance(eye_side,str):
        print("eye_side must be a string containing 'left' or 'right'")
        return False
    if eye_side != "left" and eye_side != "right":
        print("eye_side must be a string containing 'left' or 'right'")
        return False
    #Validate image_dict
    if not isinstance(image_dict,dict):
        print("image_dict is not a dictionary")
        return False
    #Create dict to store original luminance data before conversion
    luminance_dict = {}
    for key, array in image_dict.items():
        # Check if the value is a 2D numpy array
        if not isinstance(array, np.ndarray) or len(array.shape) != 2:
            print(f"{key} is not a 2D numpy array")
            return False
        # Check that the array is not empty
        if array.size == 0:
            print(f"{key} is empty")
            return False
        # If pixel values are not within the valid range 0-255, convert them
        if np.any(array < 0) or np.any(array > 255) or issubclass(array.dtype.type, np.floating):
            is_luminance = True
            print(f"Pixel in '{key}' are out of the valid range 0-255. Converting...")
            #Store original luminance data in luminance_dict
            luminance_dict[key] = np.copy(array)
            # Make all values positive
            array = array - np.min(array)
            # Normalize the array
            array = array / np.max(array)
            # Apply gamma correction and convert to gray level values
            array = np.power(array, 1/2.2) * 255
            # Convert the array data to 8-bit integer datatype
            image_dict[key] = array.astype(np.uint8)
        else: is_luminance = False
            
    #Set show_result = False to skip plotting 
    #Set show_result = True to show image processing debugging purposes
    show_result = False

    #Define a numpy array with coordinates from which to crop the image, defined by Ghost_CropRect in config.
    cropRect =np.array(analysis_config['Ghost_CropRect']).astype('int')

    #Define dictionary from which to store ghost analysis results
    ghost = {}

    #Loop through each image passed in image_dict to do ghost analysis
    for key, image in image_dict.items():

        #Add dictionary for the current key results
        ghost[key] = {}

        # if show_result is True, define a figure to plot the image processing steps
        if show_result == True: fig, axes = plt.subplots(3, 2, figsize=(8, 11)) 

        #If eye_side is right, flip the image
        if eye_side == "right":
            image = np.fliplr(image)
        if is_luminance == True:
            luminance_dict[key] = np.fliplr(luminance_dict[key])

        #Crop the image to the area defined by cropRect coordinates
        ghost_image = image[cropRect[1]:cropRect[1]+cropRect[3], cropRect[0]:cropRect[0]+cropRect[2]]
        #If is_luminance is True then crop the luminance data to the area defind by the cropRect coordinates
        if is_luminance == True: 
            luminance_dict[key] = luminance_dict[key][cropRect[1]:cropRect[1]+cropRect[3], cropRect[0]:cropRect[0]+cropRect[2]]

        #Make a copy of ghost_image to maintain the original data before image processing
        ghost_image_raw = copy.deepcopy(ghost_image) #Create copy of ghost_image to keep without applying image processing

        #If show_result is True, plot the original image on the figure
        if show_result == True:
            axes[0,0].imshow(ghost_image, cmap = "gray")
            axes[0,0].set_title('Original image')

        #Apply a band pass filter to ghost_image in frequency domain defined by the Source_Size
        filtered_image = util.filter_image_fft(ghost_image, analysis_config["Source_Size"])

        #If show_result is True, plot the filtered image on the figure
        if show_result == True:
            axes[0,1].imshow(filtered_image, cmap = "gray")
            axes[0,1].set_title('Filtered image')    

        #Apply thresholding to the filtered image to get a binary image
        block_size = 49 
        C = 0
        image_thresholded = cv2.adaptiveThreshold(filtered_image.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block_size, C)

        #If show_result is True, plot the thresholded image on the figure
        if show_result == True:
            axes[1,0].imshow(image_thresholded, cmap = "gray")
            axes[1,0].set_title('Thresholded image')

        #Find and draw contours from the thresholded image
        contours0, _ = cv2.findContours(image_thresholded.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour_img = cv2.drawContours(cv2.cvtColor(ghost_image, cv2.COLOR_GRAY2BGR), contours0, -1, (0, 255, 0), 2)

        #If show_result is True, plot the image with contours on the figure
        if show_result == True:
            axes[1,1].imshow(contour_img)
            axes[1,1].set_title('Detected contours')

        #Define imgME image from which to draw the detected squares 
        imgME = image_thresholded

        #Get the detected square locations from the contours
        square_locations = util.get_ghost_locations(contours0,analysis_config["Source_Size"])
        print(f"{len(square_locations)} squares detected")

        #Calculate ghost strength if number of squares is 2
        if len(square_locations)==2:

            #Get the center location of the two squares
            square_centroid1 = [int(np.average(square_locations[0][:, 0])), int(np.average(square_locations[0][:, 1]))]
            square_centroid2 = [int(np.average(square_locations[1][:, 0])), int(np.average(square_locations[1][:, 1]))]

            #Assign the square further to the right as source square and the other square as ghost square
            if (square_centroid1[0]>square_centroid2[0]):
                source_loc = np.array(square_centroid1)
                ghost_loc = np.array(square_centroid2)
            else:
                source_loc = np.array(square_centroid2)
                ghost_loc = np.array(square_centroid1)

            #Get the x and y coordinates of the source and ghost locations
            source_x = source_loc[0] + cropRect[0]
            ghost_x = ghost_loc[0] + cropRect[0]
            source_y = source_loc[1] + cropRect[1]
            ghost_y = ghost_loc[1] + cropRect[1]
            ghost[key]['Source_LocX'] = source_x
            ghost[key]['Source_LocY'] = source_y
            ghost[key]['Ghost_LocX'] = ghost_x
            ghost[key]['Ghost_LocY'] = ghost_y

            if is_luminance == False:
                #If the input data is not luminance, calculate average of square pixels in ghost_image_raw
                ghost[key]['Source_Lum'] = \
                    util.calculate_average_in_square(ghost_image_raw, source_loc[0], source_loc[1], analysis_config['Source_Size'])
                ghost[key]['Ghost_Lum'] = \
                    util.calculate_average_in_square(ghost_image_raw, ghost_loc[0], ghost_loc[1], analysis_config['Source_Size'])
            else:
                #If the input data is luminance, calculate average of square pixels in the original data
                ghost[key]['Source_Lum'] = \
                    util.calculate_average_in_square(luminance_dict[key], source_loc[0], source_loc[1], analysis_config['Source_Size'])
                ghost[key]['Ghost_Lum'] = \
                    util.calculate_average_in_square(luminance_dict[key], ghost_loc[0], ghost_loc[1], analysis_config['Source_Size'])

            #Calculate the ghost strength
            ghost[key]['Ghost_Strength_Perc'] = \
            100*ghost[key]['Ghost_Lum']/ghost[key]['Source_Lum']

            #Draw the contours on imgME
            hull = cv2.convexHull(contours0[0])
            cv2.drawContours(imgME, [hull], 0, 255)
            hull = cv2.convexHull(contours0[1])
            cv2.drawContours(imgME, [hull], 0, 255)

        #If the number of squares is not 2, then store the error message in ghost dict
        elif len(square_locations) > 2:
            ghost[key]['ErrMsg'] = 'More than 3 squares detected'
        elif len(square_locations) == 1:
            ghost[key]['ErrMsg'] = 'Only one square detected'
        else:
            ghost[key]['ErrMsg'] = 'No squares detected'

        #Plot the squares on the image
        for square in square_locations:
            cv2.polylines(imgME, [square.reshape((-1, 1, 2))], True, (0, 255, 0), thickness=2)

        #If show_result is True of save_image is True, plot the raw image
        if show_result == True or analysis_config["save_image"] == "True":
            plt.imshow(ghost_image_raw, cmap = "gray")
            plt.title('Detected polygons')


        #Plot source and ghost location dots
        dot_offset = analysis_config['Source_Size']/2
        for square in square_locations:
            plt.scatter(np.average(square[:,0]), np.average(square[:,1]), color='blue',s=dot_offset)
            plt.scatter(np.average(square[:,0])-dot_offset,
                        np.average(square[:,1]-dot_offset), color='blue',s=dot_offset)
            plt.scatter(np.average(square[:,0])+dot_offset,
                        np.average(square[:,1]+dot_offset), color='blue',s=dot_offset)
            plt.scatter(np.average(square[:,0])-dot_offset,
                        np.average(square[:,1]+dot_offset), color='blue',s=dot_offset)
            plt.scatter(np.average(square[:,0])+dot_offset,
                        np.average(square[:,1]-dot_offset), color='blue',s=dot_offset)

        #If show_result is True, show the figure then clear the figure
        if show_result == True:
            plt.show()
            fig.clear()

        #Create the figure for the detected ghost result
        h1 = plt.figure(key +' Ghost detection', figsize=(8, 6))
        plt.imshow(ghost_image)
        plt.title(key +' Ghost detection')
        plt.axis('off')

        #Plot source and ghost location
        for square in square_locations:
            plt.scatter(np.average(square[:,0]), np.average(square[:,1]), color='blue',s=dot_offset)
            plt.scatter(np.average(square[:,0])-dot_offset,
                        np.average(square[:,1]-dot_offset), color='blue',s=dot_offset)
            plt.scatter(np.average(square[:,0])+dot_offset,
                        np.average(square[:,1]+dot_offset), color='blue',s=dot_offset)
            plt.scatter(np.average(square[:,0])-dot_offset,
                        np.average(square[:,1]+dot_offset), color='blue',s=dot_offset)
            plt.scatter(np.average(square[:,0])+dot_offset,
                        np.average(square[:,1]-dot_offset), color='blue',s=dot_offset)

        
        #Get the date and time to add timestamp the saved image
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        #If save_image is True save the figure
        output_path = os.path.join(r'images',key+f'_ghostDetection {current_time}.png')
        if analysis_config["save_image"] == "True": plt.savefig(output_path)
        plt.close('all')

    ghost = util.flatten_dict(ghost)

    return ghost

def get_pupil_swim(im1:np.ndarray,
                    im2:np.ndarray,
                    analysis_config:dict,
                    station_config:dict) -> dict:

    """
    This function gets the pupil swim between two checkerboard images by calculating the angle  
    variation between checkerboard image im1 and im2.
    Inputs:
        im1: numpy array containing the first checkerboard image 
        im2: numpy array containing the second checkerboard image 
        analysis_config: dictionary containing analysis config parameters
        station_config: dictionary containing staiton config parameters
        
    Output:
        A dictionary containing global and local pupil swim RMS & 90 percentile values, and the number of global 
        pupil swim outliers.
        {
        'GlobalPupilSwim_OutlierNo': int
        'GlobalPupilSwim_RMS_arcmin': float
        'GlobalPupilSwim_P90_arcmin': float
        'LocalPupilSwim_RMS': float
        'LocalPupilSwim_P90': float
        ...
        }
    The function first loads the configuration file and extracts the relevant parameters. 
    It detects each of the corners in the checkerboard images im1 and im2.
    The angular difference is calculated between the corners in im1 and im2 to find the pupil swim.
    The results are returned in a dictionary.
    """

    #Set show_result to True to show image processing steps for debugging purposes
    show_result = False

    #Check if the input arrays are of the correct type
    if not isinstance(im1, np.ndarray):
        print("im1 must be a numpy array")
        return False
    if not isinstance(im2, np.ndarray):
        print("im2 must be a numpy array")
        return False
    #Check if input arrays have equal dimensions
    if im1.shape != im2.shape:
        print("im1 and im2 must have equal dimensions")
        return False
    #Check if input arrays have empty data
    if im1.shape[0] == 0 or im2.shape[0] == 0:
        print("Data is empty")
        return False
    #Check if either array contains any zeros
    if np.any(im1 == 0) or np.any(im2 == 0):
        # Adding a very small number to both arrays
        im1 += 1e-10
        im2 += 1e-10

    # Check if all required keys exist in analysis_config and station_config dictionaries
    required_keys_analysis_config = {
        'checkerboard_columns': int, 
        'checkerboard_rows': int, 
        'checker_width': (int, float), 
        'checker_height': (int, float), 
        'MaxPupilSwimVectorFilter': (int, float), 
        'CheckerDensity': (int, float), 
        'save_image': str
    }
    for key, value_type in required_keys_analysis_config.items():
        if key not in analysis_config:
            print(f"{key} key not found in analysis_config")
            return False
        if not isinstance(analysis_config[key], value_type):
            print(f"{key} in analysis_config must be of type {value_type}")
            return False
        if key == 'save_image' and analysis_config[key] not in ['True', 'False']:
            print(f"{key} in analysis_config must be either 'True' or 'False'")
            return False
    required_keys_station_config = {
        'pixel_size': (int, float), 
        'binning': (int, float), 
        'focal_length': (int, float)
    }
    for key, value_type in required_keys_station_config.items():
        if key not in station_config:
            print(f"{key} key not found in station_config")
            return False
        if not isinstance(station_config[key], value_type):
            print(f"{key} in station_config must be of type {value_type}")
            return False

    checkerboard_columns = analysis_config["checkerboard_columns"]
    checkerboard_rows = analysis_config["checkerboard_rows"]
    checker_width = analysis_config["checker_width"]
    checker_height = analysis_config["checker_height"]

    # Read images
    Points1 = util.get_checkerboard_points(im1, checkerboard_columns*checkerboard_rows, checker_width,analysis_config)
    Points2 = util.get_checkerboard_points(im2, checkerboard_columns*checkerboard_rows, checker_width,analysis_config) 
    checkerPoints1 = sort2grid.sort2grid(Points1)
    checkerPoints2 = sort2grid.sort2grid(Points2)
    if show_result == True:
        plt.imshow(im2)
        plt.scatter(Points1[:,0],Points1[:,1],c="g",s=1.5)
        plt.scatter(Points2[:,0],Points2[:,1],c="r",s=1.5)
        plt.show()
        plt.imshow(im2)
        plt.scatter(checkerPoints2[:,:,0],checkerPoints2[:,:,1],c="r",s=1.25,label="im2 detected points")
        plt.show()
        plt.imshow(im1)
        plt.scatter(checkerPoints1[:,:,0],checkerPoints1[:,:,1],c="g",s=1.25,label="im1 detected points")
        plt.show()
        
    #Shift grid coordinates such that checkerPoints1 and checkerPoints2 are mapped correctly
    shift_range = max(abs(np.shape(checkerPoints1)[0] - np.shape(checkerPoints2)[0]),abs(np.shape(checkerPoints1)[1] - np.shape(checkerPoints2)[1]))
    checkerPoints1, checkerPoints2 = util.shift_grid(checkerPoints1, checkerPoints2, shift_range = 5)

    # Reshape the grids into square grids. Grids must be square for the pupil swim calculation
    # Find the maximum length of the first two dimensions
    max_len = max(checkerPoints1.shape[0], checkerPoints1.shape[1])
    # Create a new array filled with nan values of shape (max_len, max_len, 2)
    new_checkerPoints1 = np.full((max_len, max_len, 2), np.nan)
    new_checkerPoints2 = np.full((max_len, max_len, 2), np.nan)
    # Calculate the start indices for the original data in the new array
    start_idx1 = (max_len - checkerPoints1.shape[0]) // 2
    start_idx2 = (max_len - checkerPoints1.shape[1]) // 2
    # Insert the original data into the new array
    new_checkerPoints1[start_idx1:start_idx1+checkerPoints1.shape[0], start_idx2:start_idx2+checkerPoints1.shape[1]] = checkerPoints1
    new_checkerPoints2[start_idx1:start_idx1+checkerPoints2.shape[0], start_idx2:start_idx2+checkerPoints2.shape[1]] = checkerPoints2
    # Replace the original checkerPoints1 and checkerPoints2 with the new arrays
    checkerPoints1 = new_checkerPoints1
    checkerPoints2 = new_checkerPoints2
    # Create masks for where values are nan in each array
    nan_mask1 = np.isnan(checkerPoints1)
    nan_mask2 = np.isnan(checkerPoints2)
    # If a value is nan in one array but not the other, replace it with nan in both arrays
    checkerPoints1[nan_mask2] = np.nan
    checkerPoints2[nan_mask1] = np.nan

    if analysis_config["save_image"] == "True":
        #Get the date and time to add timestamp the saved image
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        plt.imshow(im1)
        plt.scatter(checkerPoints2[:,:,0],checkerPoints2[:,:,1],c="g",s=0.25,label="im2 detected points")
        plt.scatter(checkerPoints1[:,:,0],checkerPoints1[:,:,1],c="r",s=0.25,label="im1 detected points")
        plt.title("im1")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'images\im1 {current_time}.png',dpi = 800)
        plt.cla()
        plt.imshow(im2)
        plt.scatter(checkerPoints1[:,:,0],checkerPoints1[:,:,1],c="r",s=0.25,label="im1 detected points")
        plt.scatter(checkerPoints2[:,:,0],checkerPoints2[:,:,1],c="g",s=0.25,label="im2 detected points")
        plt.title("im2")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(f'images\im2 {current_time}.png',dpi = 800)
    
    Centroid1X = checkerPoints1[:, :, 0]
    Centroid1Y = checkerPoints1[:, :, 1]

    # calculate difference between checker points in xy
    DIFFx = checkerPoints2[:, :, 0] - checkerPoints1[:, :, 0]
    DIFFy = checkerPoints2[:, :, 1] - checkerPoints1[:, :, 1]

    output = {}

    # filter out outliers if myshape algo can not handle distorted image properly
    Diffr = np.sqrt(DIFFy ** 2 + DIFFx ** 2)
    mask_outlier = Diffr > analysis_config['MaxPupilSwimVectorFilter']
    output['GlobalPupilSwim_OutlierNo'] = np.sum(mask_outlier)


    DIFFx[mask_outlier] = np.nan
    DIFFy[mask_outlier] = np.nan

    # find mask without nans for both of images
    mask_found = ~np.isnan(DIFFx)
    output['GlobalPupilSwim_OutlierNo'] = np.sum(mask_outlier)

    #Get pixel angle in arc-min per pixel
    ppd = 1/ math.degrees(math.atan(station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
    pixel_angle_arcmin = 1/(ppd/60)

    output['GlobalPupilSwim_RMS_arcmin'],output['GlobalPupilSwim_P90_arcmin']= util.get_pupil_swim_stats(DIFFx[mask_found],
                                                                              DIFFy[mask_found],  pixel_angle_arcmin)

    if str(output['GlobalPupilSwim_RMS_arcmin']) == '-1111':
        return output

    # set xy axes scale
    LL = DIFFx.shape[0]
    x = analysis_config['CheckerDensity'] * np.linspace((-LL + 1) / 2, (LL - 1) / 2, LL)
    # Create the meshgrid using numpy.meshgrid
    X, Y = np.meshgrid(x, x)

    # create a model
    DIFF_regression_model = np.column_stack((np.ones(LL * LL), X.flatten(), Y.flatten()))
    mask_found_1D = mask_found.flatten()

    # applying linear regression to remove low orders with points without nans
    DIFFx_1D = DIFFx.flatten()
    DIFFy_1D = DIFFy.flatten()
    DIFFx1D_masked = DIFFx_1D[mask_found_1D]
    DIFFy1D_masked = DIFFx_1D[mask_found_1D]

    DIFF_regression_model_masked = DIFF_regression_model[mask_found_1D,:]
    model_x = LinearRegression().fit(DIFF_regression_model_masked,DIFFx1D_masked)
    model_y = LinearRegression().fit(DIFF_regression_model_masked,DIFFy1D_masked)
    predict_x = model_x.predict(DIFF_regression_model_masked)
    predict_y = model_y.predict(DIFF_regression_model_masked)

    #calculate residual for local pupil swim
    residual_x = (DIFFx1D_masked- predict_x)
    residual_y = (DIFFy1D_masked- predict_y)

    # reshape 1D array back to 2D array with nans
    temp = np.full( DIFFx_1D.shape, np.nan)
    temp[mask_found_1D] = residual_x
    residual_x = temp
    temp = np.nan * DIFFx_1D
    temp[mask_found_1D] = residual_y
    residual_y = temp

    # reshape 1D residual array to 2D array with nans
    DIFFx_r = residual_x.reshape(LL, LL)
    DIFFy_r = residual_y.reshape(LL, LL)

    output['LocalPupilSwim_RMS'],output['LocalPupilSwim_P90']= util.get_pupil_swim_stats(DIFFx_r[mask_found], DIFFy_r[mask_found], pixel_angle_arcmin)

    #If save_image is True save the figure
    if analysis_config["save_image"] == "True":
    
        #Get the date and time to add timestamp the saved image
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')

        # Save intermediate data for plot
        filename_xls = f'images\PS_data {current_time}.xlsx'
        with pd.ExcelWriter(filename_xls,mode="w", engine="xlsxwriter") as writer:
            # use to_excel function and specify the sheet_name and index
            # to store the dataframe in specified sheet
            (pd.DataFrame(checkerPoints1[:,:,0])).to_excel(writer, sheet_name="Image1_Points_X", index=False)
            (pd.DataFrame(checkerPoints1[:,:,1])).to_excel(writer, sheet_name="Image1_Points_Y", index=False)
            (pd.DataFrame(checkerPoints2[:,:,0])).to_excel(writer, sheet_name="Image2_Points_X", index=False)
            (pd.DataFrame(checkerPoints2[:,:,1])).to_excel(writer, sheet_name="Image2_Points_Y", index=False)
            (pd.DataFrame(mask_found)).to_excel(writer, sheet_name="Image1_mask", index=False)
            (pd.DataFrame(mask_found)).to_excel(writer, sheet_name="Image2_mask", index=False)
            (pd.DataFrame(mask_found)).to_excel(writer, sheet_name="MaskUnion", index=False)
            (pd.DataFrame(DIFFx)).to_excel(writer, sheet_name="DiffX", index=False)
            (pd.DataFrame(DIFFy)).to_excel(writer, sheet_name="DiffY", index=False)

        # Result Figures,
        fig = plt.figure(333,figsize=(18,14))
        fig.suptitle('LPS', fontsize=12, y=0.95)
        # Calculate the limits of x and y where nan values are not present
        x_limits = [np.nanmin(X[~np.isnan(DIFFx)]), np.nanmax(X[~np.isnan(DIFFx)])]
        y_limits = [np.nanmin(Y[~np.isnan(DIFFy)]), np.nanmax(Y[~np.isnan(DIFFy)])]
        # Subplot 1: GPS Vector
        plt.subplot(2, 2, 1)
        plt.quiver(X, Y, DIFFx, DIFFy)
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        plt.title('GPS Vector', fontsize=10)  # Adjust fontsize as desired
        # Subplot 2: LPS Magnitude
        plt.subplot(2, 2, 2)
        plt.quiver(X, Y, residual_x, residual_y)
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        plt.title('LPS Vector (residual)')
        # Subplot 3: LPS magnitude in false color
        ax = plt.subplot(2, 2, 3)
        ps_mag2D = pixel_angle_arcmin * np.sqrt(DIFFy ** 2 + DIFFx ** 2)
        cax = ax.pcolormesh(X, Y, ps_mag2D)
        plt.xlim(x_limits)
        plt.ylim(y_limits)
        plt.title('LPS Magnitude (arcmin)', fontsize=10)  # Adjust fontsize as desired
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)  # Adjust the fraction and pad as needed
        # Subplot 3: Histogram of LPS Magnitude
        filename_ps = rf'images\lps {current_time}.png'
        plt.subplot(2, 2, 4)
        plt.hist(ps_mag2D.flatten())
        plt.title(f"Local Pupil Swim (arcmin)\nRMS={output['LocalPupilSwim_RMS']:.2f} arcmin P90={output['LocalPupilSwim_P90']:.2f} arcmin", fontsize=10)  # Adjust fontsize as desired
        plt.savefig(filename_ps)

    return output

def get_fov(Y_dict:dict, fov_zoneA:np.ndarray, analysis_config: dict, station_config: dict) -> dict:
    """
    This function calculates the horizontal and vertical FOV for eyebox region 0 measurements. 
    If data is passed for region 1, FOV is also calculated for region 1 measurements.
    Inputs:
        Y_dict: dictionary of dictionaries containing Y data numpy arrays.
                The Y_dict dictionary shall contain a "region 0" dictionary containing the region 0 Y data 
                numpy arrays, and a "region 1" dictionary containing the region 1 Y data numpy arrays.
                Dictionary must contain at least contain at the least "region 0" dictioinary. 
                "region 0" and "region 1" can contain a single, or multiple arrays for Y data.
                The keys for the numpy arrays can have any name.
                        {
                        "region 0":{"image1":np.ndarray, "image2":np.ndarray,"image3":np.ndarray}
                        "region 1":{"image4":np.ndarray, "image5":np.ndarray,"image6":np.ndarray,"image7":np.ndarray}
                        }
        fov_zoneA: FOV zone mask for zoneA
                   np.ndarray 
        analysis_config: dictionary containing parameters for analysis config file
        station_config: dictionary containing parameters for station config file
        
    Output:
        rolloff_dict: A dictionary of float values for the horizontal FOV, vertical FOV, and percent coverage to the target FOV 
                      for each image. The dictionary will also contain the minimum horizontal & vertical FOV, the minimum
                      percent coverage to the target FOV, the target horizontal and vertical FOV itself.
                      Dictionary will contain at least "region 0" data.
        {
        'region 0 image1 hFOV':float,
        'region 0 image1 vFOV':float,
        'region 0 image1 FOV coverage':float,
        ...
        'region 1 image7 hFOV':float,
        'region 1 image7 vFOV':float,
        'region 1 image7 FOV coverage':float,
        'region 0 min hFOV': float,
        'region 0 min vFOV': float,
        'region 1 min hFOV': float,
        'region 1 min vFOV': float,
        'region 0 min FOV coverage': float,
        'region 1 min FOV coverage':float,
        'target hFOV':float,
        'target vFOV':flaot
        }
    The function first loads the station and analyzer configuration files and extracts the relevant parameters. 
    The horizontal and vertical FOV are then calculated for region 0 using the Artemis definition:  
        1) The average luminance for zoneA in each region 0 EB positions is calculated.
        2) The horizontal and vertical boundaries of the FOV for each of the images in region 0 are determined by using 
           the lum_threshold*average zoneA calculated in step 1.
    The horizontal and vertical FOV boundaries are then calculated for region 1: 
        3) The horizontal and vertical boundaries of the FOV for each of the images in region 1 are determined by using
           the lum_threshold*average zoneA calculated in step 1.
    """

     # Check if the variable is a dictionary with at least "region 0"
    if isinstance(Y_dict, dict) and "region 0" in Y_dict:
        pass
    else:
        print('Y_dict is not a dictionary containing at least "region 0"')
        return False
    #Validate zoneA mask
    #Check if zoneA mask is a boolean numpy array
    if isinstance(fov_zoneA,np.ndarray) and np.isin(fov_zoneA, [True,False]).all():
        pass
    else:
        print(f"zoneA mask is not a boolean numpy array")
        return False
    # Validate Y_dict
    for region_key, region_dict in Y_dict.items():
        for array_key, array in region_dict.items():
            if not isinstance(array, np.ndarray):
                print(f"{region_key} {array_key} must be a numpy array")
                return False
            # Check if input arrays have empty data
            if array.shape[0] == 0:
                print(f"{region_key} {array_key} is empty")
                return False
            #Check if input array is 8 bit image data
            array = util.check_if_8bit_image(array)
            if len(array.shape) == 2: pass
            else:
                print(f"{region_key} {array_key} is not a 2D or 3D array")
                return False  
            # Check if zoneA mask is the same size as the Y_data array
            if np.shape(array) != np.shape(fov_zoneA):
                print(f"zoneA mask is not the same shape as {region_key} {array_key}")
                return False

    #Set show_result to True or False to show plots for debugging purposes
    show_result = False

    #Make list to store average luminance for each array in region 0
    region0_zoneA_averages = []
    #Iterate over each Y_image in Y_dict region 0:
    for key, array in Y_dict["region 0"].items():
        #Use util.get_FOV_zone_data function to apply fov_zone mask to Y_image input data
        masked_brightness = util.get_FOV_zone_data(array, fov_zoneA)
        
        #Calculate the mean brightness value for the masked data, ignoring nan values
        mean_masked_brightness = np.nanmean(masked_brightness)

        #Append mean brightness value to zone_averages
        region0_zoneA_averages.append(mean_masked_brightness)

    #Get the mean of zoneA luminance across all arrays in region 0
    region0_zoneA_average = np.mean(region0_zoneA_averages)

    #Initialize dictionary to store FOV results
    fov_dict = {}

    #Get the current eyebox region from Y_dict
    for region_key, region_dict in Y_dict.items():
        fov_dict[region_key] = {}

        #Get the current data array from the current eyebox region
        for array_key, array in region_dict.items():
            if show_result == True:
                plt.imshow(array)
                plt.show()
            fov_dict[region_key][array_key] = {}

            #Calculate the luminance threshold to detect the FOV boundaries 
            threshold = region0_zoneA_average*analysis_config["lum_threshold"]

            # Initialize variables to store the min and max boundary points
            min_boundary_x = None
            max_boundary_x = None
            min_boundary_y = None
            max_boundary_y = None

            # Calculate the average luminance for each column in the array
            column_avg = np.mean(array, axis=0)
            # Find the min and max boundary points for the horizontal direction
            for i, avg in enumerate(column_avg):
                if avg > threshold:
                    if min_boundary_x is None or i < min_boundary_x:
                        min_boundary_x = i
                    if max_boundary_x is None or i > max_boundary_x:
                        max_boundary_x = i

            # Calculate the average luminance for each row in the array
            row_avg = np.mean(array, axis=1)
            # Find the min and max boundary points for the vertical direction
            for i, avg in enumerate(row_avg):
                if avg > threshold:
                    if min_boundary_y is None or i < min_boundary_y:
                        min_boundary_y = i
                    if max_boundary_y is None or i > max_boundary_y:
                        max_boundary_y = i

            if show_result == True or analysis_config["save_image"] == "True":
                plt.imshow(array[int(min_boundary_y):int(max_boundary_y),int(min_boundary_x):int(max_boundary_x)])
                 #Get the date and time to add timestamp the saved image
                current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                plt.savefig(f'images\{region_key} {array_key} FOV boundary {current_time}.png'
                            ,dpi = 400, facecolor=(1,1,1,1), transparent=False)
                if show_result == True: plt.show()

            #Convert boundary coordinates to coordinates relative to sensor center
            min_boundary_x = min_boundary_x - (np.shape(array)[1]/2)
            max_boundary_x = max_boundary_x - (np.shape(array)[1]/2)
            min_boundary_y = min_boundary_y - (np.shape(array)[0]/2)
            max_boundary_y = max_boundary_y - (np.shape(array)[0]/2)

            #Convert boundary coordinates to FOV coordinates
            min_boundary_x_FOV = math.degrees(math.atan(min_boundary_x*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            max_boundary_x_FOV = math.degrees(math.atan(max_boundary_x*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            min_boundary_y_FOV = math.degrees(math.atan(min_boundary_y*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            max_boundary_y_FOV = math.degrees(math.atan(max_boundary_y*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))

            #Calculate and store the hFOV, vFOV, and FOV coverage
            hFOV = max_boundary_x_FOV - min_boundary_x_FOV
            vFOV = max_boundary_y_FOV - min_boundary_y_FOV
            fov_dict[region_key][array_key]["hFOV"] = hFOV
            fov_dict[region_key][array_key]["vFOV"] = vFOV
            fov_coverage = hFOV*vFOV/(analysis_config["target_hFOV"]*analysis_config["target_vFOV"])
            fov_dict[region_key][array_key]["FOV coverage"] = fov_coverage

        #Find and store the min hFOV, min vFOV, min FOV coverage for each region
        hFOV_list = [fov_dict[region_key][array_key]["hFOV"] for array_key in fov_dict[region_key]]
        vFOV_list = [fov_dict[region_key][array_key]["vFOV"] for array_key in fov_dict[region_key]]
        FOV_coverage_list = [fov_dict[region_key][array_key]["FOV coverage"] for array_key in fov_dict[region_key]]
        fov_dict[region_key]["min hFOV"] = min(hFOV_list)
        fov_dict[region_key]["min vFOV"] = min(vFOV_list)
        fov_dict[region_key]["min FOV coverage"] = min(FOV_coverage_list)
        
    #Store the target hFOV and vFOV
    fov_dict["target hFOV"] = analysis_config["target_hFOV"]
    fov_dict["target vFOV"] = analysis_config["target_vFOV"]

    fov_dict = util.flatten_dict(fov_dict)

    return fov_dict

def get_gamut(cx_cy_dict:dict, analysis_config:dict, ref_gamut = "P3-D65") -> float:
    '''
        Get the color gamut for the given RGB CIE xy values by calculating the overlap area of their gamut triangle 
        relative to the reference gamut RGB CIE xy values.
        input: 
            cx_cy_dict: dictionary containing numpy arrays for the CIE xy data for each RGB color primary.
                        Key values must be named "R", "G", and "B"
                        {
                        "R":[CIEx,CIEy],
                        "G":[CIEx,CIEy],
                        "B":[CIEx,CIEy],
                        }
            ref_gamut: string for reference color gamut. Must be "P3-D65", "DCI-P3", "sRGB", or "Adobe RGB".
                       "P3-D65" is selected by default. 
            analysis_config: dictionary containg analysis config parameters

        output:
            A float value for the percent gamut coverage relative to the reference color gamut.

        The function calculates the color gamut for the given RGB CIE xy values with respect to a reference gamut
        by calculating the overlap area.                                                         
    '''

    ref_gamut_dict = {
                      "P3-D65":{"R":[0.68,.32],"G":[.265,.69],"B":[.15,.06]},
                      "DCI-P3":{"R":[0.68,.32],"G":[.265,.69],"B":[.15,.06]},
                      "sRGB":{"R":[0.64,.33],"G":[.30,.60],"B":[.15,.06]},
                      "Adobe RGB":{"R":[0.64,.33],"G":[.21,.71],"B":[.150,.06]}
                     }
    #Validate input cx_cy_dict array
    for key, array in cx_cy_dict.items():
        #Check if the array is a numpy array with 2 elements
        if not (isinstance(array, np.ndarray) and array.size == 2 and np.issubdtype(array.dtype, np.floating)):
            print("{key} must be a numpy array containing 2 float values for CIE x and y")
            return False
    #Validate input ref_gamut
    if ref_gamut not in ref_gamut_dict.keys():
        print('ref_gamut must be a value of "P3-D65", "DCI-P3", "sRGB", or "Adobe RGB"')

    ref_gamut = ref_gamut_dict[ref_gamut]
    
    # Get the points for the input and reference gamuts
    input_points = [cx_cy_dict[color] for color in ['R', 'G', 'B']]
    ref_points = [ref_gamut[color] for color in ['R', 'G', 'B']]

    #Calculate the gamut coverage
    numerator = (((cx_cy_dict["R"][0]-cx_cy_dict["B"][0])*(cx_cy_dict["G"][1]-cx_cy_dict["B"][1]))
                -((cx_cy_dict["G"][0]-cx_cy_dict["B"][0])*(cx_cy_dict["R"][1]-cx_cy_dict["B"][1])))
    denominator = (((ref_gamut["R"][0]-ref_gamut["B"][0])*(ref_gamut["G"][1]-ref_gamut["B"][1]))
                -((ref_gamut["G"][0]-ref_gamut["B"][0])*(ref_gamut["R"][1]-ref_gamut["B"][1])))
    gamut_coverage = numerator/denominator

    # If analysis_config["save_image"] is True, plot the CIE color space
    if analysis_config["save_image"] == "True":
        colour.plotting.plot_chromaticity_diagram_CIE1931(standalone=False)
        # Plot the input and reference gamuts
        plt.fill(*zip(*input_points), alpha=0)
        plt.fill(*zip(*ref_points), alpha=0)
        # Plot the input and reference gamuts
        plt.plot(*zip(*input_points, input_points[0]), label='Input Gamut')
        plt.plot(*zip(*ref_points, ref_points[0]), label='Reference Gamut')
        plt.legend()
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Color Gamut')
        #Get the date and time to add timestamp the saved image
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # Save the plot
        #plt.show()
        plt.savefig(f'images\color gamut {current_time}.png',dpi = 400, facecolor=(1,1,1,1), transparent=False)
    return gamut_coverage

def get_sequential_contrast(brightIm:np.ndarray,darkIm:np.ndarray,fov_zones:dict) -> dict:
    """
    This function calculates the average sequential contrast from a bright in dark image for each FOV zone.
    Inputs:
        brightIm: numpy array containing the bright image
        darkIm: numpy array containing the dark image 
        fov_zones: dict of FOV zone masks {"zoneA":np.ndarray,"zoneB":np.ndarray,"zoneC":np.ndarray}
        
    Output:
        A dictionary containing the average sequential contrast for each FOV zone.
        {
        'zoneA sequential contrast':float, 'zoneB sequential contrast':float, 'zoneC sequential contrast':float, ...
        }
    The function first calculates the average luminance of a FOV zone in the bright and dark images separately. 
    Then the function calculates the sequential contrast by dividing the average luminance of the bright image by 
    the average luminance of the dark image.
    """
    #Check if the input arrays are of the correct type
    if not isinstance(brightIm, np.ndarray):
        print("brightIm must be a numpy array")
        return False
    if not isinstance(darkIm, np.ndarray):
        print("darkIm must be a numpy array")
        return False
    #Check if input arrays have equal dimensions
    if brightIm.shape != darkIm.shape:
        print("brightIm and darkIm must have equal dimensions")
        return False
    #Check if input arrays have empty data
    if brightIm.shape[0] == 0 or darkIm.shape[0] == 0:
        print("Data is empty")
        return False
    #Validate fov_zones
    for zone in fov_zones.keys():
        #Check if fov_zone mask is a boolean numpy array
        if isinstance(fov_zones[zone],np.ndarray) and np.isin(fov_zones[zone], [True,False]).all():
            pass
        else:
            print(f"{zone} mask is not a boolean numpy array")
            return False
        #Check if fov_zone mask is same size as the Y_data arrays
        for image in [brightIm,darkIm]:
            #Check if the input array is of the correct dimensions
            if np.shape(image) != np.shape(fov_zones[zone]):
                print(f"{zone} mask is not the same shape as image")
                return False

    #Create empty dictionary to store the sequential contrast for each FOV zone
    contrast_dict = {}
    
    #Iterate over each FOV zone in the fov_zone dictionary
    for fov_zone in fov_zones.keys():
        contrast_dict[fov_zone] = {}
        
        #Use util.get_FOV_zone_data function to apply fov_zone mask to brightIm and darkIm images
        brightIm_masked = util.get_FOV_zone_data(brightIm, fov_zones[fov_zone])
        darkIm_masked = util.get_FOV_zone_data(darkIm, fov_zones[fov_zone])
        
        #Calculate the mean brightness value for the brightIm & darkIm masked data
        brightIm_mean = np.nanmean(brightIm_masked)
        darkIm_mean = np.nanmean(darkIm_masked)

        #Calculate the sequential contrast
        sequential_contrast = brightIm_mean/darkIm_mean
        
        #Store the sequential contrast for the current FOV zone
        contrast_dict[fov_zone]["sequential contrast"] = sequential_contrast

    #Flatten dictionary
    contrast_dict = util.flatten_dict(contrast_dict)

    return contrast_dict

def get_color_difference(measurement_data_folder:str, reference_data_folder:str, analysis_config: dict) -> dict:
    '''
        Get the see through transmittance and color difference metrics calculated in CTTanalysis.  
        input: 
            measurement_data_folder: string containing the filepath to the measurement data folder.
                                     The folder must contain the .xml file for CTTanalysis and a subfolder 
                                     named "Data" containing the measurement image files.
            reference_data_folder: string containing the filepath to the reference data folder.
                                   The folder must contain the reference image files for CTTanalysis.
            analysis_config: dictionary containg analysis config parameters

        output:
            A dictionary containing the see through transmission, color difference, and yellow index metrics:
            {
            'Transmission_A_Radiometric': float
            'Transmission_B_Radiometric': float
            'TransmissionAngular_A_Radiometric': float
            'TransmissionAngular_B_Radiometric': float
            'Transmission_A_Photometric': float
            'Transmission_B_Photometric': float
            'TransmissionAngular_A_Photometric': float
            'TransmissionAngular_B_Photometric': float
            'Color_difference_deltaEab_A': float
            'Color_difference_deltaEab_B': float
            'Color_differenceAngular_deltaEab_A': float
            'Color_differenceAngular_deltaEab_B':float
            'Yellow_Index_A': float
            'Yellow_Index_B': float
            }

        The function calls the CTTanalysis module to calculate the see through transmittance, color difference,
        and yellow index metrics with the provided measurement and reference data.                                                        
    '''
    # Validate the input arguments
    if not isinstance(measurement_data_folder, str) or not os.path.isdir(measurement_data_folder):
        print("measurement_data_folder is not a valid directory path")
        return False
    if not glob.glob(os.path.join(measurement_data_folder, "*.xml")):
        print("measurement_data_folder does not contain a .xml file")
        return False
    if not os.path.isdir(os.path.join(measurement_data_folder, "Data")):
        print('measurement_data_folder does not contain a subfolder named "Data"')
        return False
    if not isinstance(reference_data_folder, str) or not os.path.isdir(reference_data_folder):
        print("reference_data_folder is not a valid directory path")
        return False
    if not glob.glob(os.path.join(reference_data_folder, "*.[jp][np][g]")):
        print("reference_data_folder does not contain image files")
        return False
    if not isinstance(analysis_config, dict):
        print("analysis_config is not a dictionary")
        return False
    if 'save_image' not in analysis_config:
        print("analysis_config does not contain 'save_image'")
        return False
    if analysis_config['save_image'] not in ['True', 'False']:
        print("analysis_config['save_image'] is not 'True' or 'False'")
        return False
    
    result_metrics = CTTanalysis.CTTanalysis(measurement_data_folder, reference_data_folder, analysis_config)

    return result_metrics

def get_distortion(image: np.ndarray, test_pattern: np.ndarray, analysis_config:dict) -> dict:
    """
    This function calculates the distortion in a measured image compared to the projected test pattern.
    The distortion is calculated using the definition from the ICDM 19.7.
    Inputs:
        image: numpy array containing the measured image
        test_pattern: numpy array containing the reference test pattern
        analysis_config: dictionary containing analysis config parameters
    Output:
        A dictionary containing an array of the distortion percentage values for each measurement point 
        in the measured image, the abs mean, abs max, and std dev of the distortion.
        {
        'distortion grid': np.ndarray,
        'distortion mean': float,
        'distortion max': float,
        distortion std dev': float,
        'distortion (%) p95': float,
        }
    The function first detects the distortion measurement points in the measured image and test pattern. 
    The measured image points are then mapped to the test pattern points.
    The distance is calculated for the ideal and measured points relative to the center.
    The distortion for each point is calculated as 100*(actual distance-ideal distance)/(ideal distance).
    """
    # Validate image and test_pattern
    if not isinstance(image, np.ndarray):
        print("image must be a numpy array.")
        return False
    if not isinstance(test_pattern, np.ndarray):
        print("test_pattern must be a numpy array.")
        return False
    if image.ndim < 2 or test_pattern.ndim < 2:
        print("image and test_pattern must have 2 dimensional image data.")
        return False
    if not np.issubdtype(image.dtype, np.number) or not np.issubdtype(test_pattern.dtype, np.number):
        print("image and test_pattern must contain numerical data.")
        return False


    # Set show_result to True to plot out image processing steps for debugging
    show_result = False

    DISTORTION_LEN_RATIO = 1
    # Get the shape of the test pattern
    W, H = test_pattern.shape[0], test_pattern.shape[1]

    #Get the number of grid columns and rows from analysis_config
    grid_columns = analysis_config["grid_columns"]
    grid_rows = analysis_config["grid_rows"]
    #Get the total number points in the grid
    total_points = grid_columns*grid_rows

    # Get the centroids of points in the test pattern and the image
    xy_test_pattern = sort2grid.get_point_centroids(test_pattern, MASK=None, sigma = 0, expected_points=total_points)
    xy_source = sort2grid.get_point_centroids(image, MASK=None, threshold=.95, expected_points=total_points, sort_by='sum')
    
    # Map the coordinates from the test pattern to the source image to get the ideal positions
    ideal_pos = util.coordinate_mapping(xy_source, xy_test_pattern, xy_test_pattern, zoom=1.00, show_result=show_result)
    ideal_pos = np.array(ideal_pos)

    # Use sort2grid to make a grid from ideal_pos and find the center location 
    # After the previous coordinate mapping, target center and source center positions are the same
    ideal_pos_grid = sort2grid.sort2grid(ideal_pos)
    source_pos_grid = sort2grid.sort2grid(xy_source)
    center = ideal_pos_grid[ideal_pos_grid.shape[0]//2, ideal_pos_grid.shape[1]//2,:]
    center_x, center_y = center[0], center[1]

    # Calculate the euclidean distance of the source positions and ideal positions from the center position
    ideal_d = (((ideal_pos_grid[:,:,0] - center_x)**2) + ((ideal_pos_grid[:,:,1] - center_y)**2))**0.5
    source_d = (((source_pos_grid[:,:,0] - center_x)**2) + ((source_pos_grid[:,:,1] - center_y)**2))**0.5

    # Calculate the distortion
    distortion_grid = np.nan_to_num(((source_d - ideal_d)/ideal_d)*100)

    # Return the distortion results as a dictionary
    result_dict = {
                    'distortion (%) grid': distortion_grid,
                    'distortion (%) mean': np.mean(np.abs(distortion_grid)),
                    'distortion (%) max': np.amax(np.abs(distortion_grid)),
                    'distortion (%) std dev': np.std(distortion_grid),
                    'distortion (%) p95': np.percentile(np.abs(distortion_grid), 95)
                    }

    return result_dict

def get_masks(image: np.ndarray, analysis_config: dict, station_config: dict) -> dict:
    """
    This function returns FOV zone masks for the input image. 
    The masks are for Artemis with parameters defined in the analyzers.config file.
    The function also return a list of x,y,w,h parameters for a cropping angle 
    of the active area full FOV in the iamge.
    Inputs:
        image: numpy array containing the measured image
        analysis_config: dictionary containing analysis config parameters
        station_config: dictionary containing station config parameters
    Output:
        A dictionary containing the FOV zone masks as boolean numpy arrays.
        {
        'full FOV': np.ndarray,
        'zoneA': np.ndarray,
        'zoneB': np.ndarray,
        'zoneC': np.ndarray
        }
        A list containing integers for parameters of a cropping rectangle for the full FOV
        after image rotation.
        [x,y,w,h]

    The function first detects the display area in the input image to find the full FOV boundaries.
    Then masks for zoneA, zoneB, and zoneC are generated using the parameters defined in analyzers.config.
    A plot of the zones is saved if save_image set to True in config.
    """
    # Validate the input image
    if not isinstance(image, np.ndarray):
        print("image is not a numpy array")
        return False
    if image.ndim == 3 and image.dtype == np.uint8:
        # Convert RGB to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif not image.ndim == 2:
        print("image is not a 2D numpy array")
        return False
    if not np.issubdtype(image.dtype, np.number):
        print("image does not contain numerical data")
        return False
    # Validate the analysis_config dictionary
    required_analysis_keys = ["rotation_angle", "target_hFOV", "target_vFOV", 
                              "zoneB_h1", "zoneB_h2", "zoneB_h3", "zoneB_h4", 
                              "zoneB_v1", "zoneB_v2", "zoneB_v3", "zoneB_v4", "zoneA_FOV", "save_image"]
    if not all(key in analysis_config for key in required_analysis_keys):
        print("analysis_config is missing one or more required keys")
        return False
    if not isinstance(analysis_config["rotation_angle"], (int, float)):
        print("rotation_angle in analysis_config is not a numerical value")
        return False
    if not all(isinstance(analysis_config[key], (int, float)) and analysis_config[key] >= 0 for key in required_analysis_keys[1:12]):
        print("One or more values in analysis_config are not non-negative numerical values")
        return False
    if analysis_config["zoneA_FOV"] >= min(analysis_config["target_hFOV"], analysis_config["target_vFOV"]):
        print("zoneA_FOV in analysis_config is not smaller than target_hFOV and target_vFOV")
        return False
    if analysis_config["save_image"] not in ["True", "False"]:
        print("save_image in analysis_config is not 'True' or 'False'")
        return False
    # Validate the station_config dictionary
    required_station_keys = ["pixel_size", "binning", "focal_length"]
    if not all(key in station_config for key in required_station_keys):
        print("station_config is missing one or more required keys")
        return False
    if not isinstance(station_config["pixel_size"], (int, float)) or station_config["pixel_size"] <= 0:
        print("pixel_size in station_config is not a positive numerical value")
        return False
    if not isinstance(station_config["binning"], int) or station_config["binning"] <= 0:
        print("binning in station_config is not a positive integer")
        return False
    if not isinstance(station_config["focal_length"], (int, float)):
        print("focal_length in station_config is not a numerical value")
        return False
    
    # Set show_result to True to plot out image processing steps for debugging
    show_result = False

    #Save a copy of the original image data before processing
    original_image = np.copy(image)

    #Rotate image with config parameter
    image = util.rotate_array(image,analysis_config["rotation_angle"])

    # Check if the image is a floating-point image
    if image.dtype.kind == 'f':
        # Normalize the image to the range [0, 255] and convert to 8-bit integer
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #Initialize the mask dict
    masks_dict = {}

    ###################################################################################################################################################
    #Find the rectangular boundary of the display area in the measurement image

    # Apply thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresh = cv2.GaussianBlur(thresh, (155, 155), 0)

    if show_result == True:
        plt.imshow(thresh)
        plt.show()

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if show_result == True:
        # Draw bounding boxes around the contours
        image_with_boxes = thresh.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (255,255,255), 3)
        plt.imshow(image_with_boxes)
        plt.show()

    # Target aspect ratio
    target_aspect_ratio = analysis_config["target_hFOV"] / analysis_config["target_vFOV"]
    #Target angle area = 
    target_angle_area = analysis_config["target_hFOV"]*analysis_config["target_vFOV"]
    # Initial aspect ratio window (you can adjust this value)
    window = 0.2
    # Set the timeout (seconds)
    timeout = 60
    start_time = time.time()
    while True:
        # Filter contours based on aspect ratio
        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h

            #Get contour dimensions in angle
            contour_x1 = x - (np.shape(image)[1]/2)
            contour_x2 = x + w - (np.shape(image)[1]/2)
            contour_y1 = y - (np.shape(image)[0]/2)
            contour_y2 = y + h - (np.shape(image)[0]/2)
            contour_x1_FOV = math.degrees(math.atan(contour_x1*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            contour_x2_FOV = math.degrees(math.atan(contour_x2*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            contour_y1_FOV = math.degrees(math.atan(contour_y1*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            contour_y2_FOV = math.degrees(math.atan(contour_y2*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            contour_w_angle = contour_x2_FOV - contour_x1_FOV
            contour_h_angle = contour_y2_FOV - contour_y1_FOV

            # Calculate the angular area of the contour
            contour_angle_area = contour_w_angle * contour_h_angle
            # Check if the contour is within the window for both aspect ratio and angular area
            #if abs(aspect_ratio - target_aspect_ratio) <= window and abs(1 - (contour_angle_area/target_angle_area)) <= window:
            if abs(1 - (contour_angle_area/target_angle_area)) <= window:
                # print("window",window)
                # print("aspect ratio delta",abs(aspect_ratio - target_aspect_ratio))
                # print("angular area delta",abs(1 - (contour_angle_area/target_angle_area)))
                # print("contour angular area",contour_angle_area)
                # print("x, y, w, h",x, y, w, h)
                # print("w_angle, h_angle",contour_w_angle,contour_h_angle)
                filtered_contours.append(cnt)
        # If only one contour is found or the timeout is reached, break the loop
        if len(filtered_contours) == 1:
            break
        if time.time() - start_time > timeout:
            print("get_masks timed out. Active area not found.")
            return False
        # If no contours found, widen the window
        if len(filtered_contours) == 0:
            window *= 1.1
        # If more than one contour found, narrow the window
        if len(filtered_contours) > 1:
            window *= 0.9
    # The final contour
    final_contour = filtered_contours[0]

    # If a contour was found, draw the bounding rectangle
    if final_contour is not None:
        x, y, w, h = cv2.boundingRect(final_contour)
        cv2.rectangle(image, (x, y), (x+w, y+h), (255, 255, 255), 2)

    if show_result == True:
        # Display the image
        plt.imshow(image)
        plt.show()

    # Create a blank image of the same size as the original image
    mask_full_FOV = np.zeros(image.shape[:2], dtype=np.uint8)
    #Mask off the area outside the detected contour region
    mask_full_FOV[:,:x] = 255
    mask_full_FOV[:y,:] = 255
    mask_full_FOV[:,x+w:] = 255
    mask_full_FOV[y+h:,:] = 255

    mask_full_FOV = mask_full_FOV.astype(bool)

    #Store the rectangular boundary as the full FOV mask.
    masks_dict["full FOV"] = mask_full_FOV

    #Create roi list for x, y, w, h
    roi = [x, y, w, h]

    ###################################################################################################################################################
    #Generate the mask for zoneC. Crop the corners of the FOV defined in config

    corner_widths = np.array([analysis_config["zoneB_h1"],analysis_config["zoneB_h2"],analysis_config["zoneB_h3"],analysis_config["zoneB_h4"]])
    corner_widths = (corner_widths/analysis_config["target_hFOV"])*w
    corner_lengths = np.array([analysis_config["zoneB_v1"],analysis_config["zoneB_v2"],analysis_config["zoneB_v3"],analysis_config["zoneB_v4"]])
    corner_lengths = corner_lengths/analysis_config["target_vFOV"]*h

    w = w - 1
    h = h - 1 
    top_left = np.array([[[x, y], [corner_widths[0]+x, y], [x, corner_lengths[0]+y]]], dtype=np.int32)
    top_right = np.array([[[w-corner_widths[1]+x, y], [w+x, y], [w+x, corner_lengths[1]+y]]], dtype=np.int32)
    bottom_left = np.array([[[x, h+y-corner_lengths[2]], [x, h+y], [corner_widths[2]+x, h+y]]], dtype=np.int32) 
    bottom_right = np.array([[[w+x, h+y], [w+x, h-corner_lengths[3]+y], [w-corner_widths[3]+x,h+y]]], dtype=np.int32)

    if show_result == True:
        # Draw the triangles
        cv2.fillPoly(image, top_left, (0, 0, 0))
        cv2.fillPoly(image, top_right, (0, 0, 0))
        cv2.fillPoly(image, bottom_left, (0, 0, 0))
        cv2.fillPoly(image, [bottom_right], (0, 0, 0))

        # Display the image
        plt.imshow(image)
        plt.show()

    # Create a blank image of the same size as the original image
    mask_zoneC = np.zeros(image.shape[:2], dtype=np.uint8)
    # Draw the triangles on the mask
    cv2.fillPoly(mask_zoneC, top_left, 255)
    cv2.fillPoly(mask_zoneC, top_right, 255)
    cv2.fillPoly(mask_zoneC, bottom_left, 255)
    cv2.fillPoly(mask_zoneC, bottom_right, 255)
    # Mask off the areas outside the contour (display) region
    mask_zoneC[0:,0:x] = 255
    mask_zoneC[0:y,0:] = 255
    mask_zoneC[0:,x+w:] = 255
    mask_zoneC[y+h:,0:] = 255
    # Convert the mask to a boolean mask
    mask_zoneC = mask_zoneC.astype(bool)
    if show_result == True:
        # Display the mask
        plt.imshow(mask_zoneC, cmap='gray')
        plt.show()

    #Store the mask for zoneC in dict
    masks_dict["zoneC"] = mask_zoneC

    ###################################################################################################################################################
    #Generate the mask for zoneA. Center FOV circular region with diameter defined in config

    zoneA_diameter_FOV = analysis_config["zoneA_FOV"]
    zoneA_diameter_relative = zoneA_diameter_FOV/analysis_config["target_hFOV"]
    zoneA_diameter_pixels = zoneA_diameter_relative*w

    mask_zoneA = np.zeros(image.shape[:2], dtype=np.uint8)
    mask_zoneA[:,:] = 255
    # Get the detected contour
    center_x = x + w//2
    center_y = y + h//2
    # Draw the circle on the mask
    cv2.circle(mask_zoneA, (center_x, center_y), int(zoneA_diameter_pixels // 2), 0, -1)
    # Convert the mask to a boolean mask
    mask_zoneA = mask_zoneA.astype(bool)
    if show_result == True:
        # Display the mask
        plt.imshow(mask_zoneA, cmap='gray')
        plt.show()

    #Store the mask for zoneA in dict
    masks_dict["zoneA"] = mask_zoneA

    ###################################################################################################################################################
    #Generate the mask for zoneB. Defined as the area outside zoneA and inside zoneC.

    # Create the mask for zoneB
    mask_zoneB = np.bitwise_not(np.bitwise_and(np.bitwise_not(mask_zoneC), mask_zoneA))
    if show_result == True:
        # Display the mask
        plt.imshow(mask_zoneB, cmap='gray')
        plt.show()

    #Store the mask for zoneB in dict
    masks_dict["zoneB"] = mask_zoneB

    ###################################################################################################################################################
    #Rotate the image and masks back to the original orientation of the input image. Crop the image and masks back to original resolution
    # Get the center of the original image
    original_center = (np.array(original_image.shape[:2][::-1]) - 1) / 2.0
    # Rotate the image and masks back to the original orientation
    rotation_angle = -analysis_config["rotation_angle"]
    image = util.rotate_array(image, rotation_angle)
    for key in masks_dict:
        masks_dict[key] = util.rotate_array(masks_dict[key], rotation_angle)
    # Get the center of the rotated image
    rotated_center = (np.array(image.shape[:2][::-1]) - 1) / 2.0
    # Calculate the top left corner of the cropping rectangle
    top_left = np.round(rotated_center - original_center).astype(int)
    # Calculate the bottom right corner of the cropping rectangle
    bottom_right = top_left + np.array(original_image.shape[:2][::-1])
    # Crop the image and masks to the original resolution
    image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    for key in masks_dict:
        masks_dict[key] = masks_dict[key][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    ###################################################################################################################################################
    #Save output image result if save_image set to True.

    if analysis_config["save_image"] == "True":
        f"images\artemis masks.png"

    if analysis_config["save_image"] == "True":
        # Create a blank RGB image
        background_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb_image = np.zeros((*image.shape, 3), dtype=np.uint8)
        rgb_image[:, :, :] = background_image[:, :, np.newaxis]
        # Erode the masks and subtract from the original masks to get the outlines
        outline_width = int(background_image.shape[0]*0.002899)
        outline_zoneA = binary_dilation(masks_dict["zoneA"] & ~binary_erosion(masks_dict["zoneA"]),iterations = outline_width)
        outline_zoneC = binary_dilation(masks_dict["zoneC"] & ~binary_erosion(masks_dict["zoneC"]),iterations = outline_width)
        outline_fullFOV = binary_dilation(masks_dict["full FOV"] & ~binary_erosion(masks_dict["full FOV"]),iterations = outline_width)
        # Draw the outlines on the image
        rgb_image[outline_zoneA] = [255, 0, 0]  # Red for zoneA
        rgb_image[outline_zoneC] = [0, 0, 255]  # Blue for zoneC
        rgb_image[outline_fullFOV] = [0,255,0] # Green for full FOV
        # Set the edge pixels of rgb_image to 0
        rgb_image[:outline_width, :] = 0
        rgb_image[-outline_width:, :] = 0
        rgb_image[:, :outline_width] = 0
        rgb_image[:, -outline_width:] = 0
        # Create a hatched pattern for Zone B
        zoneB_mask = masks_dict["zoneC"] & ~masks_dict["zoneA"]
        hatched_pattern = np.zeros_like(background_image)
        hash_width = outline_width
        for i in range(0, hatched_pattern.shape[0], hash_width*5):  # Adjust the stride to change the density of the hatching
            hatched_pattern[i:i+hash_width, :] = 255
        hatched_zoneB = cv2.bitwise_and(hatched_pattern, ~masks_dict["zoneB"].astype(np.uint8) * 255)
        hatched_zoneB = np.where(hatched_zoneB==2,0,hatched_zoneB)*255
        rgb_image[hatched_zoneB == 255] = [0, 255, 255]  # Cyan for Zone B
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb_image)
        ax.set_title('Masks')
        # Create the legend
        red_patch = mpatches.Patch(color='red', label='Zone A')
        blue_patch = mpatches.Patch(color='blue', label='Zone C')
        green_patch = mpatches.Patch(color='green', label='full FOV')
        cyan_patch = mpatches.Patch(color='cyan', label='Zone B')
        ax.legend(handles=[red_patch, green_patch, blue_patch, cyan_patch], loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.set_xlim(2, original_image.shape[1] - 2)
        ax.set_ylim(original_image.shape[0] - 2, 2)
        plt.tight_layout()
        # Save the plot
        plt.savefig(f"images/masks {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.png", bbox_inches='tight')
        plt.cla()

    return masks_dict, roi

def get_gamma(gammaIm: np.ndarray, 
              brightIm: np.ndarray, 
              darkIm: np.ndarray, 
              gamma_target_Im : np.ndarray,
              analysis_config: dict) -> dict:
    """
    This function returns the gamma coefficient and luminance data from the input images. 
    Inputs:
        gammaIm: numpy array containing the measured gamma image
        brightIm: numpy array containing the measured gray 255 gamma image
        darkIm: numpy array containing the measured dark image (gray 0)
        analysis_config: dictionary containing the analysis_config parameters
    Output:
        A dictionary containing the gamma coefficient and a dataframe with the luminance 
        data from the input images.
        {
        'gamma_coeff': float,
        'gamma_df': float,
        }
    The function first detects the centers of the gamma squares in the test pattern and measured images.
    Then the average luminance values are calculated for the rois in gammaIm, brightIm, and darkIm.
    The gamma is calculated by subtracting the background from gammaIm and brightIm values, normalizing 
    the gammaIm values to brightIm, then extracting the coefficient from curve generated from plotting 
    gray level vs normalized luminance.
    A plot of the images with rois and gamma curve is saved is save_image is set to true in config.
    """

    # Validate input arguments
    if not(isinstance(analysis_config, dict)):
        print("analysis_config should be a dictionary")
    # Check if analysis_config didn't already have parameters selected from "Gamma". If it didn't, select them.
    if "Gamma" in analysis_config:
        analysis_config = analysis_config["Gamma"]
    # Check if analysis_config is a dictionary containing the keys save_image and roi_area_percentage
    if not ("save_image" in analysis_config 
            or "roi_area_percentage" in analysis_config):
        print("analysis_config should contain the keys save_image and roi_area_percentage.")
        return False
    # Check if gammaIm is a 2D numpy array containing numerical data
    if not isinstance(gammaIm, np.ndarray) or len(gammaIm.shape) != 2 or not np.issubdtype(gammaIm.dtype, np.number):
        print("gammaIm should be a 2D numpy array containing numerical data.")
        return False
    # Check if brightIm is a 2D numpy array containing numerical data
    if not isinstance(brightIm, np.ndarray) or len(brightIm.shape) != 2 or not np.issubdtype(brightIm.dtype, np.number):
        print("brightIm should be a 2D numpy array containing numerical data.")
        return False
    # Check if darkIm is a 2D numpy array containing numerical data
    if not isinstance(darkIm, np.ndarray) or len(darkIm.shape) != 2 or not np.issubdtype(darkIm.dtype, np.number):
        print("darkIm should be a 2D numpy array containing numerical data.")
        return False
    # Check if gamma_target_Im is a 3D numpy array containing integer data with values from 0-255
    if not (isinstance(gamma_target_Im, np.ndarray) 
            or len(gamma_target_Im.shape) != 3 
            or not np.issubdtype(gamma_target_Im.dtype, np.integer) 
            or np.max(gamma_target_Im) > 255 
            or np.min(gamma_target_Im) < 0):
        print("gamma_target_Im should be a 3D numpy array with values from 0-255.")
        return False
    # Check if gammaIm, brightIm, and darkIm all have the same axis lengths
    if gammaIm.shape != brightIm.shape or gammaIm.shape != darkIm.shape:
        print("gammaIm, brightIm, and darkIm should all have the same axis lengths.")
        return False

    ###################################################################################################################################################
    #Get the gamma square center locations in gamma_target_Im and brightIm

    #Convert gamma_target_Im to grayscale
    gamma_target_grayscale = cv2.cvtColor(gamma_target_Im, cv2.COLOR_BGR2GRAY)
    # Find 2nd smallest value in gamma_target_Im to get the gray0 value of the teset pattern
    gray0 = np.unique(gamma_target_Im.flatten())[1]
    # Apply binary threshold
    _, thresh = cv2.threshold(gamma_target_grayscale, gray0-1, 255, cv2.THRESH_BINARY)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contoursIm = np.zeros((gamma_target_grayscale.shape[0],gamma_target_Im.shape[1]))
    # Initialize an empty list to store the corners
    box_centers = []

    # Iterate over the contours
    for contour in contours:
        # Get bounding box for each contour
        x, y, w, h = cv2.boundingRect(contour)
        
        # Get the center of the bounding box
        center = [y+(h//2), x+(w//2)]
        
        # Add the corners and their corresponding gray level values to the list
        box_centers.append([center[0], center[1]])
        
        # Draw bounding box on the original image
        cv2.rectangle(contoursIm, (x, y), (x+w, y+h), (255, 255, 255), 2)

    # Convert the list of corners to a numpy array
    box_centers = np.array(box_centers)

    #Get the centroids of the boxes in brightIm
    expected_points = len(contours)
    brightIm_centroids = sort2grid.get_point_centroids(brightIm,None,expected_points,.95,sort_by='area')

    #Convert to grid
    box_centers = sort2grid.sort2grid(box_centers)
    brightIm_centroids = sort2grid.sort2grid(brightIm_centroids)

    ###################################################################################################################################################
    #Get the roi data for gray_level, gamma_luminance, ref_luminance, dark_luminance

    #Initialize arrays for gray_level, gamma_luminance, ref_luminance, dark_luminance
    gray_level = np.zeros((brightIm_centroids.shape[0],brightIm_centroids.shape[1]))
    gamma_luminance = np.zeros((brightIm_centroids.shape[0],brightIm_centroids.shape[1]))
    ref_luminance = np.zeros((brightIm_centroids.shape[0],brightIm_centroids.shape[1]))
    dark_luminance = np.zeros((brightIm_centroids.shape[0],brightIm_centroids.shape[1]))

    #Get roi dimensions
    roi_area_percentage = analysis_config["roi_area_percentage"]
    target_x,target_y,target_w,target_h = cv2.boundingRect(contours[0])
    roi_scalar = (np.ptp(brightIm_centroids,0)/np.ptp(box_centers,0))[0][1]
    roi_w = int(w*roi_scalar * np.sqrt(roi_area_percentage))
    roi_h = int(h*roi_scalar * np.sqrt(roi_area_percentage))

    #Make new roi image to visualize roi's
    roi_Im = np.copy(gammaIm) 
    # Iterate over box_centers to get roi data
    for row_i in range(brightIm_centroids.shape[0]):
        for column_i in range(brightIm_centroids.shape[1]):
            
            #Get the gamma value from gamma_target_Im
            gray_level[row_i, column_i] = gamma_target_Im[int(box_centers[row_i,column_i,1]),
                                                     int(box_centers[row_i,column_i,0])][0]
            # Calculate the top-left corner of the ROI
            roi_x = int(brightIm_centroids[row_i, column_i, 0]) - roi_w // 2
            roi_y = int(brightIm_centroids[row_i, column_i, 1]) - roi_h // 2

            #Get the gamma luminance data in the roi from gammaIm
            gamma_luminance[row_i, column_i] = np.mean(gammaIm[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w])
            
            #Get the reference luminance data in the roi from brightIm
            ref_luminance[row_i, column_i] = np.mean(brightIm[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w])

            #Get the dark luminance data in the roi from darkIm
            dark_luminance[row_i, column_i] = np.mean(darkIm[roi_y:roi_y+roi_h,roi_x:roi_x+roi_w])

            # Draw the ROI on the image
            cv2.rectangle(roi_Im, (roi_x, roi_y), (roi_x + roi_w, roi_y + roi_h), (255, 255, 255), 2)

    ###################################################################################################################################################
    #Arrange the gray & luminance data into a dataframe and calculate the gamma

    # Flatten the arrays and stack them along a new axis
    stacked = np.stack((gray_level.flatten(), gamma_luminance.flatten(), 
                        ref_luminance.flatten(), dark_luminance.flatten()), axis=-1)
    # Convert the stacked array to a DataFrame
    gamma_df = pd.DataFrame(stacked, columns=['gray_level', 'gamma_luminance', 'ref_luminance', 'dark_luminance'])
    # Sort the DataFrame by the 'gray_level' column in ascending order
    gamma_df = gamma_df.sort_values(by='gray_level')
    # Reset the index of the DataFrame
    gamma_df = gamma_df.reset_index(drop=True)

    #Calculate gamma luminance normalized to reference luminance after subtracting background
    gamma_df["gamma_luminance_norm"] = ((gamma_df["gamma_luminance"]-gamma_df["dark_luminance"])/
                                        (gamma_df["ref_luminance"]-gamma_df["dark_luminance"]))

    gamma = util.get_gamma_coeff(gamma_df["gray_level"],gamma_df["gamma_luminance_norm"])

    ###################################################################################################################################################
    #Save a plot of the images and gamma if save_image set to true in config

    # Function to draw ROIs on an image
    def draw_rois(ax, centroids, roi_w, roi_h):
        for row_i in range(centroids.shape[0]):
            for column_i in range(centroids.shape[1]):
                # Calculate the top-left corner of the ROI
                roi_x = int(centroids[row_i, column_i, 0]) - roi_w // 2
                roi_y = int(centroids[row_i, column_i, 1]) - roi_h // 2
                # Draw the ROI on the image
                rect = mpatches.Rectangle((roi_x, roi_y), roi_w, roi_h, linewidth=1, edgecolor='r', facecolor='none')
                ax.add_patch(rect)
    # If save_image set to true save a plot of the images with rois & gamma
    if analysis_config["save_image"] == "True":
        fig, axs = plt.subplots(1, 4, figsize=(20, 5))
        
        # Plot gammaIm with the rois drawn on the image
        im1 = axs[0].imshow(gammaIm, cmap='viridis')
        draw_rois(axs[0], brightIm_centroids, roi_w, roi_h)
        axs[0].set_title('gammaIm')
        fig.colorbar(im1, ax=axs[0], orientation='vertical')

        # Plot brightIm with the rois drawn on the image
        im2 = axs[1].imshow(brightIm, cmap='viridis')
        draw_rois(axs[1], brightIm_centroids, roi_w, roi_h)
        axs[1].set_title('brightIm')
        fig.colorbar(im2, ax=axs[1], orientation='vertical')

        # Plot darkIm with the rois drawn on the image
        im3 = axs[2].imshow(darkIm, cmap='viridis')
        draw_rois(axs[2], brightIm_centroids, roi_w, roi_h)
        axs[2].set_title('darkIm')
        fig.colorbar(im3, ax=axs[2], orientation='vertical')

        # Plot gamma_df["gray_level"] vs gamma_df["gamma_luminance_norm"]
        axs[3].plot(gamma_df["gray_level"], gamma_df["gamma_luminance_norm"], marker = ".",  label=f'gamma={round(gamma,2)}')
        axs[3].set_xlabel('gray_level')
        axs[3].set_ylabel('gamma_luminance_norm')
        axs[3].legend()
        axs[3].set_title('gray_level vs gamma_luminance_norm')
        plt.tight_layout()

        # Save the plot
        plt.savefig(f"images/gamma {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.png", bbox_inches='tight')
        plt.cla()
        plt.clf()

    #Store the results in a dict
    gamma_dict = {}
    gamma_dict["gamma_df"] = gamma_df
    gamma_dict["gamma_coeff"] = gamma

    return gamma_dict

def get_angular_resolution(image: np.ndarray, station_config: dict, analysis_config: dict) -> dict:
    """
    This function returns the resolution in cycles/degree and arcmin calculated from the LSF
    of an input image.
    Inputs:
        image: numpy array containing the measured line image
        station_config: dictionary containing the station_config parameters
        analysis_config: dictionary containing the analysis_config parameters
    Output:
        A dictionary containing a float value for the resolution in cycles/degree.
        {
        'resolution (cyc/degree)': float
        'resolution (arcmin)': float
        }
    The function first gets the line spread function from the MTF_cal.py module.
    Then the full width half max of the LSF function is converted to angular space 
    to calculate the resolution in cycles/degree.  The function returns a dictionary 
    with the resolution value.
    """

    #set show_result to True to see LSF plot for debugging purpose
    show_result = False

    #Get the slope of the line in the image from MTF_cal
    slope = MTF_cal.get_line_slope(image)

    # If the absolute slope is less than 1, rotate the image 90 degrees
    if abs(slope) < 1:
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        slope = -1 / slope if slope != 0 else float('inf')
    
    #Get the LSF from MTF_cal
    lsf, _ = MTF_cal.slanted_image_to_bin(image, 1, slope)

    if show_result == True:
        plt.plot(np.arange(len(lsf)),lsf)
        plt.show()

    # Find the maximum of the data and its corresponding index
    max_val = np.max(lsf)
    max_idx = np.argmax(lsf)
    # Find the half maximum
    half_max = max_val*(analysis_config["FWHM_threshold"])
    # Find the indices where the data is greater than the half maximum
    half_max_indices = np.where(lsf > half_max)[0]

    #Get the FWHM in angle units, convert indices to be realtive to camera sensor center
    min_hm_i = half_max_indices[0] - (np.shape(image)[1]/2)
    max_hm_i = half_max_indices[-1] - (np.shape(image)[1]/2)
    min_hm_angle = math.degrees(math.atan(min_hm_i*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
    max_hm_angle = math.degrees(math.atan(max_hm_i*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))

    #Get the FWHM in angular space and calculate the resolution in cycles/degree
    FWHM_angle = max_hm_angle - min_hm_angle
    FWHM_arcmin = FWHM_angle*60
    cyc_per_deg = 1/FWHM_angle

    if show_result == True:
        plt.plot(np.arange(len(lsf)),lsf)
        plt.scatter(half_max_indices,lsf[half_max_indices],c="r")
        plt.show()

    #Store the result in a dictionary
    resolution_dict = {"resolution (cyc/deg)":cyc_per_deg}
    resolution_dict["resolution (arcmin)"] = FWHM_arcmin

    return resolution_dict

def get_crop_rect(images: dict or np.ndarray, rotation_angle: float, roi: list) -> dict:
    """
    This function crops the input images to the given roi and rotation angle.
    The function returns the cropped images.
    Inputs:
        images: dictionary of images or single image to crop.
        rotation_angle: angle to rotate the images before cropping
        roi: region of interest parameters in a list [x, y, width, height]
    Output:
        An image or dictionary of images cropped the given roi and rotation angle parameters.
        The dictionary will have the same labels as the input dictionary.
        Example:
        {
        'image1':np.ndarray,
        'image2':np.ndarray,
        'image3':np.ndarray
        }
    The function first rotates the input image based on rotation_angle using util.rotate_array.
    The input image is then cropped to the roi parameters and stored in the return dictionary.
    The process is repeated if the input was a dictionary of images.
    """
    # Validate rotation_angle
    if not isinstance(rotation_angle, (int, float)):
        raise ValueError("rotation_angle must be a numerical value")
    # Validate roi
    if not isinstance(roi, list) or len(roi) != 4 or not all(isinstance(i, int) for i in roi):
        raise ValueError("roi must be a list of 4 integers")
    x, y, w, h = roi[0], roi[1], roi[2], roi[3]

    # Check if images is a dictionary or a single image
    if isinstance(images, dict):
        cropped_images = {}
        for key, image in images.items():
            # Validate image
            if not isinstance(image, np.ndarray) or len(image.shape) != 2:
                raise ValueError("images must be a dictionary of 2-dimensional numpy arrays")
            # Rotate the image
            rotated_image = util.rotate_array(image, rotation_angle)
            # Crop the image
            cropped_image = rotated_image[y:y+h,x:x+w]
            # Store the cropped image in the dictionary
            cropped_images[key] = cropped_image
        return cropped_images
    
    elif isinstance(images, np.ndarray) and len(images.shape) == 2:
        # Rotate the image
        rotated_image = util.rotate_array(images, rotation_angle)
        # Crop the image
        cropped_image = rotated_image[y:y+h,x:x+w]
        return cropped_image
    else:
        raise ValueError("images must be a 2-dimensional numpy array or a dictionary of 2-dimensional numpy arrays")

def get_pupil_efficiency(DE_image:np.ndarray, 
                         WG_images:dict, 
                         fov_zones:dict,
                         analysis_config:dict, 
                         DE_exposure:float = None, 
                         WG_exposures:dict = None):
    """
    This function calculates the pupil efficiency metrics.
    The function can calculate efficiency for a single or multiple WG images.
    In the case of image data (.bmp, .png, .tiff) exposure values should be provided
    to normalize the data to exposure time. For luminance data, keep exposures set to None.
    Inputs:
        DE_image: 2d numpy array containing a single DE image
        WG_images: dict containing 2d numpy arrays. Any number of arrays. Example:
                    {"image1":np.ndarray, "image2":np.ndarray, "image3":np.ndarray}
        fov_zones: dict of FOV zone masks 
                   Dictionary must contain all zones: 
                    {"zoneA":np.ndarray,"zoneB":np.ndarray,"zoneC":np.ndarray, "full FOV":np.ndarray}
        analysis_config: dictionary containing the analysis_config parameters
        DE_exposure: optional float value for DE image camera exposure time
        WG_exposures: optional dict of float values for the WG images' camera exposure time.
                      Must match the keys passed to WG_images. Example:
                      {"image1":float, "image2":float, "image3":float}
    Output:
        A dictionary of float values for the calculated efficiency for each WG image for each 
        FOV zone.  The mean efficiency for each zone is also returned.
        Example:
        {
        'zoneA image1':np.ndarray, 'zoneB image1':np.ndarray, 'zoneC image1':np.ndarray, 'full FOV image1':np.ndarray,
        'zoneA image2':np.ndarray, 'zoneB image2':np.ndarray, 'zoneC image2':np.ndarray, 'full FOV image2':np.ndarray,
        'zoneA image3':np.ndarray, 'zoneB image3':np.ndarray, 'zoneC image3':np.ndarray, 'full FOV image3':np.ndarray,
        'zoneA mean':np.ndarray, 'zoneB mean':np.ndarray, 'zoneC mean':np.ndarray, 'full FOV mean':np.ndarray,
        }
    The function calculates the efficiency for each image in WG_images by dividing the image by the DE_image,
    multiplied by the "transmission_ratio" scalar provided in analysis_config.  The efficiency is calculated 
    for each of the FOV zones provided in fov_zones.  If exposure values are provided in DE_exposure and WG_exposures,
    the image data is normalized by their exposure times.  Do not provided exposure times if the input data is luminance.
    """

    # Check if DE_image is a 2D numpy array with numerical data
    if not isinstance(DE_image, np.ndarray) or DE_image.ndim != 2 or not np.issubdtype(DE_image.dtype, np.number):
        print("DE_image must be a 2D numpy array with numerical data")
        return False
    # Check if WG_images is a dictionary containing 2D numpy arrays with numerical data, and check that each array is the same size
    if not isinstance(WG_images, dict):
        print("WG_images must be a dictionary")
        return False
    first_WG_image_shape = None
    for img_key, img in WG_images.items():
        if not isinstance(img, np.ndarray) or img.ndim != 2 or not np.issubdtype(img.dtype, np.number):
            print(f"WG_images[{img_key}] must be a 2D numpy array with numerical data")
            return False
        if first_WG_image_shape is None:
            first_WG_image_shape = img.shape
        elif img.shape != first_WG_image_shape:
            print("All WG_images must have the same shape")
            return False
    # Check if fov_zones contains 2D numpy arrays with boolean data, and check that each array is the same size as DE_image or the first WG_image
    if not isinstance(fov_zones, dict):
        print("fov_zones must be a dictionary")
        return False
    for zone_key, zone in fov_zones.items():
        if not isinstance(zone, np.ndarray) or zone.ndim != 2 or zone.dtype != bool:
            print(f"fov_zones[{zone_key}] must be a 2D numpy array with boolean data")
            return False
        if zone.shape != DE_image.shape and zone.shape != first_WG_image_shape:
            print(f"fov_zones[{zone_key}] must have the same shape")
    # Check if analysis_config is a dictionary containing the key "transmission_ratio", which is a float value
    if not isinstance(analysis_config, dict):
        print("analysis_config must be a dictionary")
        return False
    if "transmission_ratio" not in analysis_config or not isinstance(analysis_config["transmission_ratio"], float):
        print("analysis_config must contain the key 'transmission_ratio' with a float value")
        return False
    # Check if DE_exposure and WG_exposures are None or contain a numerical value, and both variables must be None or both contain a numerical value
    if DE_exposure is not None and WG_exposures is None:
        print("DE_exposure and WG_exposures must both be None or both contain a numerical value")
        return False
    if DE_exposure is None and WG_exposures is not None:
        print("DE_exposure and WG_exposures must both be None or both contain a numerical value")
        return False
    if DE_exposure is not None and not isinstance(DE_exposure, (int, float)):
        print("DE_exposure must be None or a numerical value")
        return False
    if WG_exposures is not None and not isinstance(WG_exposures, dict):
        print("WG_exposures must be None or a dictionary")
        return False
    if WG_exposures is not None:
        for img_key, exposure in WG_exposures.items():
            if not isinstance(exposure, (int, float)):
                print(f"WG_exposures[{img_key}] must be a numerical value")
                return False

    #Check if DE_image and fov_zones are the same size as WG_images. If not, resize them.
    first_WG_image_shape = list(WG_images.values())[0].shape
    if DE_image.shape != first_WG_image_shape:
        DE_image = transform.resize(DE_image, first_WG_image_shape)
    for fov_zone_key, fov_zone_item in fov_zones.items():
        if fov_zone_item.shape != first_WG_image_shape:
            fov_zones[fov_zone_key] = transform.resize(fov_zone_item, first_WG_image_shape)

    efficiency_dict = {}

    #If no exposures are provided, it is assumed input data is luminance and does not need normalization.
    if DE_exposure == None and WG_exposures == None:

        for fov_zone_key, fov_zone_item in fov_zones.items():
            efficiency_dict[fov_zone_key] = {}

            for WG_image_key, WG_image_item in WG_images.items():

                WG_zone_data = util.get_FOV_zone_data(WG_image_item,fov_zone_item)
                DE_zone_data = util.get_FOV_zone_data(DE_image,fov_zone_item)
                efficiency_array = WG_zone_data*analysis_config["transmission_ratio"]/DE_zone_data
                inf_mask = np.isfinite(efficiency_array)
                efficiency_mean = np.nanmean(efficiency_array[inf_mask])
                efficiency_dict[fov_zone_key][WG_image_key] = efficiency_mean

    #If exposures were provided, normalize image data to exposure time.
    else:

        for fov_zone_key, fov_zone_item in fov_zones.items():
            efficiency_dict[fov_zone_key] = {}

            for WG_image_key, WG_image_item in WG_images.items():

                WG_zone_data = util.get_FOV_zone_data(WG_image_item,fov_zone_item)
                DE_zone_data = util.get_FOV_zone_data(DE_image,fov_zone_item)
                efficiency_array = (WG_zone_data*WG_exposures[WG_image_key]*analysis_config["transmission_ratio"])/(DE_zone_data*DE_exposure)
                inf_mask = np.isfinite(efficiency_array)
                efficiency_mean = np.nanmean(efficiency_array[inf_mask])
                efficiency_dict[fov_zone_key][WG_image_key] = efficiency_mean

    #Calculate average efficiency for each FOV zone
    for fov_zone_key in fov_zones.keys():

        fov_zone_efficiencies = list(efficiency_dict[fov_zone_key].values())
        fov_zone_efficiency_mean = np.mean(fov_zone_efficiencies)
        efficiency_dict[fov_zone_key]["mean"] = fov_zone_efficiency_mean

    efficiency_dict = util.flatten_dict(efficiency_dict)

    return efficiency_dict

def get_polarization(df: pd.DataFrame, title='', angle_interest=0, angle_offset=0, save_image=False):
    '''
    Calculate the polarization peak angle and extinction ratio based on transmitted power vs linear polarizer angle data.
    input: 
        df: A dataframe containing two series. 
            df['angle'] is the angle of linear polarizer's orientation, typically read from a rotation stage
            df['power'] is the transmitted light power through the linear polarizer, typically read from a photodiode
        title: A string for the title of the data set for plot and file naming purposes. 
            If left empty, then the current timestamp will be used
        angle_offset: A float value from calibrating the angle read from the rotation stage compared to the defined reference direction of polarizer
            Such calibration can be performed using a Polarzing Beam Splitter with fixed mechanical relation to the reference direction
        angle_interest: A float value defining the angle of interest, typically specified in Engineering Requirements Specifications (ERS)
        save_image: A boolean value to determine whether to save the plot for visualizing the data and the curve fit
    output:
        Dictionary containing the angle of peak transmission 'theta_peak', the ratio b/w max transmission and min transmission 'extinction_ratio',
        and ratio b/w transmission at the angle of interest and its perpendicular 'power_ratio_interest'                                                 
    '''
    if not isinstance(df, pd.DataFrame) or 'angle' not in df.columns or 'power' not in df.columns:
        print("df must be a data frame containing angle and power")
        return False
    if not isinstance(title, str):
        print("title must be a string")
        return False
    if not isinstance(angle_interest, (float, int)):
        print("angle_interest must be a numerical value")
        return False
    if not isinstance(angle_offset, (float, int)):
        print("angle_offset must be a numerical value")
        return False
    if not isinstance(save_image, bool):
        print("save_image must be a boolean value")
        return False

    # Get the x and y values
    xdata = df['angle'].values
    ydata = df['power'].values

    # Handle the title
    if(title==''): title='polarization_'+datetime.now().strftime('%Y-%m-%d_%H-%M-%S') 
    else: title='polarization_'+title

    # Adjust x values and normalize y values
    for i in range(len(xdata)):
        x_adjusted = xdata[i]-angle_offset
        if(x_adjusted<-90): xdata[i] = x_adjusted + 180
        elif(x_adjusted>=90): xdata[i] = x_adjusted - 180
        else: xdata[i] = x_adjusted
    y_max=max(ydata)
    ydata=ydata/y_max

    # Define the fitting function using a lambda function
    func = lambda x, x_0, A, B: A * (np.cos(np.radians(x - x_0)))**2 + B

    # Fit the function to the data
    bounds = ((-90, 0, 0), (90, np.inf, np.inf))
    popt, pcov = curve_fit(func, xdata, ydata, p0=[0, 1, 0], bounds=bounds)

    # Calculate the output values
    polarization_dict = {}
    polarization_dict['theta_peak']=popt[0]
    polarization_dict['extinction_ratio']=(popt[1]/popt[2])+1
    polarization_dict['power_ratio_interest']=func(angle_interest, popt[0], popt[1], popt[2])/func(angle_interest+90, popt[0], popt[1], popt[2])

    if(save_image):
        # Create a scatter plot of the original data
        plt.scatter(xdata, ydata, label=title+' data')

        # Plot the fitted curve
        xfit = np.linspace(min(xdata), max(xdata), 1000)
        yfit = func(xfit, *popt)
        plt.plot(xfit, yfit, 'r-', label='Fitted curve')

        # Add stuff to the plot
        plt.legend()
        plt.grid()
        plt.title(f"theta_peak = {polarization_dict['theta_peak']:.2f} degrees, extinction_ratio = {polarization_dict['extinction_ratio']:.2f}"+"\n"+\
            f"power({angle_interest}deg)/power({angle_interest+90}deg) = {polarization_dict['power_ratio_interest']:.2f}")
        plt.xlabel('linear polarizer angle/degrees')
        plt.ylabel('normalized transmission power')
        #Define the path 
        path = "images"
        save_path = os.path.join(path, title+'.jpg')
        #Save the image
        plt.savefig(save_path,dpi=400)
        print(f"Image saved at {save_path}")
    return polarization_dict

def get_luminous_efficacy(brightness, power, station_config: dict):
    '''
    Calculate the luminous efficacy based on the brightness and power of a specific image.
    input: 
        brightness: the average luminance (in nits) in the full FOV to match the solid angle calculated below
        power: total eletric power driving the LEDs for the corresponding image brightness
    output:
        A float value for the luminous efficacy in lm/W.
    The function loads FOV data and pupil data from station_config, and takes brightness and power from input                                                       
    '''
    if not isinstance(brightness, (float, int)):
        print("brightness must be a numerical value")
        return False
    if not isinstance(power, (float, int)):
        print("power must be a numerical value")
        return False
    # Validate the station_config dictionary
    required_station_keys = ["vFoV", "hFoV", "pupil_width", "pupil_height"]
    if not all(key in station_config for key in required_station_keys):
        print("station_config is missing one or more required keys")
        return False
    if not isinstance(station_config["vFoV"], (int, float)) or station_config["vFoV"] <= 0:
        print("vFoV in station_config is not a positive numerical value")
        return False
    if not isinstance(station_config["hFoV"], (int, float)) or station_config["hFoV"] <= 0:
        print("hFoV in station_config is not a positive numerical value")
        return False
    if not isinstance(station_config["pupil_width"], (int, float)) or station_config["pupil_width"] <= 0:
        print("pupil_width in station_config is not a positive numerical value")
        return False
    if not isinstance(station_config["pupil_height"], (int, float)) or station_config["pupil_height"] <= 0:
        print("pupil_height in station_config is not a positive numerical value")
        return False
    # Convert hFOV and vFOV to radii in radians
    hfov_rad = np.radians(station_config["hFoV"] / 2.0)
    vfov_rad = np.radians(station_config["vFoV"] / 2.0)
    
    # Calculate the solid angle using the formula
    solid_angle = 2.0 * (np.sin(hfov_rad) * np.arctan(np.tan(vfov_rad) * np.cos(hfov_rad)) + \
           np.sin(vfov_rad) * np.arctan(np.tan(hfov_rad) * np.cos(vfov_rad)))
    pupil_area=station_config["pupil_width"]*station_config["pupil_height"]/1E6 #pupil sizes are in mm, pupil area in m^2
    etendue=solid_angle*pupil_area
    luminous_flux=etendue*brightness
    return luminous_flux/power
    
def get_exit_pupil(image: np.ndarray, analysis_config: dict, title='', check_config=False, save_image=False):
    '''
    Calculate the exit pupil length, width, and locations relative to the reference datums.
    input: 
        image: the image of the exit pupil
        title: a string used to name the saved images
        analysis_config: a dictionary containing the config parameters
            rotation_angle: angle to rotate the image so that the ROI window is horizontal
            x_window: a list indicating the start and end of the ROI window in x direction (after rotation)
            y_window: a list indicating the start and end of the ROI window in y direction (after rotation)
            origin: a list indicating the [x, y] coordinates of the origin datum (after rotation)
            datum: a list indicating the [x, y] coordinates of the other reference datum (after rotation)
            radius: an int or float value indicating the radius of each datum (in terms of pixels)
            pixel_size: an int or float value indicating the conversion from pixels to microns
        check_config: a boolean flag to switch to config parameter checking mode
            If true, the function rotates the image and draws the ROI rectangle and datum circles on it before saving it
            If false, the function calculates the exit pupil metrics using the config parameters passed in the argument
        save_image: a boolean flag to indicate whether to save images
    output:
        A dictionary containing exit pupil metrics:
            'exit_pupil_length': the length of the exit pupil in mm, along the x-direction of ROI window                                                  
            'exit_pupil_width': the width of the exit pupil in mm, along the y-direction of ROI window
            'exit_pupil_loc_h': the location of the exit pupil's center, along the h-direction (from the origin to the other reference datum)
            'exit_pupil_loc_v': the location of the exit pupil's center, along the perpendicular direction                                                 
    '''
    # Normalize the 8-bit grayscale image
    image = image/np.max(image)
    image = np.clip(image*255, 0, 255)
    image = image.astype(np.uint8)
    image = cv2.normalize(image, None, 0, 2**8, cv2.NORM_MINMAX)  

    # Get analysis config parameters from analysis_config
    rotation_angle = analysis_config["rotation_angle"]
    x_window = analysis_config["x_window"]
    y_window = analysis_config["y_window"]
    origin = analysis_config["origin"]
    datum = analysis_config["datum"]
    radius = analysis_config["radius"]
    pixel_size = analysis_config["pixel_size"] * 1e-3 #pixel size in millimeters

    # Rotate the image 
    image = util.rotate_array(image, rotation_angle)

    if(check_config==True):  
        # Visualize the rectangle window and circle datums for checking the configuration
        # Iterate to optimize the parameters in analyzers.config
        cv2.circle(image, tuple(origin), radius, 0, 1)
        cv2.circle(image, tuple(datum), radius, 0, 1)
        cv2.rectangle(image, (x_window[0], y_window[0]), \
            (x_window[1], y_window[1]), 0, 1)
        plt.imshow(image)
        plt.show()
        return
  
    #Crop the image for processing
    image = image[y_window[0]:y_window[1], x_window[0]:x_window[1]]
    h, w = image.shape

    # Initlize the results dictionary
    exit_pupil_dict={}

    # Initialize the x_profile and y_profile arrays with zeros
    x_profile = np.zeros(w)
    y_profile = np.zeros(h)

    sum_x = sum_y = 0
    # Calculate the average of pixels for each x coordinate within the window, and count up sum_x for grayscale centroid
    for x in range(w):
        x_profile[x] = np.mean(image[0:h, x])
        sum_x = sum_x + x * x_profile[x]
    # Calculate the average of pixels for each y coordinate within the window
    for y in range(h):
        y_profile[y] = np.mean(image[y, 0:w])
        sum_y = sum_y + y * y_profile[y]

    # Calculate 20% of the maximum pixel value in x_profile and y_profile
    x_threshold = 0.2 * np.max(x_profile)
    y_threshold = 0.2 * np.max(y_profile)
    # Calculate the exit pupil length and width
    x_coordinates = [i for i, x in enumerate(x_profile) if x > x_threshold]
    exit_pupil_length = len(x_coordinates)
    y_coordinates = [i for i, y in enumerate(y_profile) if y > y_threshold]
    exit_pupil_width = len(y_coordinates)    

    # Convert the length and width to units in millimeters
    exit_pupil_dict['exit_pupil_length'] = pixel_size * exit_pupil_length
    exit_pupil_dict['exit_pupil_width'] = pixel_size * exit_pupil_width

    # Calculate the grayscale centroid within ROI
    sum_grayscale = np.sum(image[0:h, 0:w])
    x_centroid = sum_x * len(y_profile) / sum_grayscale
    y_centroid = sum_y * len(x_profile) / sum_grayscale
    # Use vector calculus dot product and cross product to convert the coordinates
    OD = np.sqrt((origin[0]-datum[0])**2+(origin[1]-datum[1])**2)
    h_centroid = ((origin[0]-x_window[0]-x_centroid)*(origin[0]-datum[0])+(origin[1]-y_window[0]-y_centroid)*(origin[1]-datum[1]))/OD
    v_centroid = ((origin[0]-x_window[0]-x_centroid)*(origin[1]-datum[1])-(origin[1]-y_window[0]-y_centroid)*(origin[0]-datum[0]))/OD
    exit_pupil_dict['exit_pupil_loc_h'] = pixel_size * h_centroid
    exit_pupil_dict['exit_pupil_loc_v'] = pixel_size * v_centroid    

    if(save_image):
        # Draw a rectangle showing the boundary of the exit pupil
        cv2.rectangle(image, (min(x_coordinates), min(y_coordinates)), \
            (max(x_coordinates), max(y_coordinates)), (255, 0, 0), 1)
        # Draw the crosshair to mark the exit pupil center
        cv2.drawMarker(image, (int(x_centroid), int(y_centroid)), (0, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=10, thickness=1)
        #Get the date and time
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_name = os.path.join('images', title+f'_exit_pupil_visualization_{current_time}.jpg')
        plt.imshow(image)
        plt.title(title+f'_Exit Pupil visualization\n20%max boundary and grayscale centroid location')
        plt.savefig(image_name)
        print(f"Exit pupil boundary and center are drawn and saved at {image_name}")   

        # Create a new figure
        plt.figure()
        # Create the first subplot for x_profile
        plt.subplot(1, 2, 1)  # 1 row, 2 columns, first plot    
        # Plot x_profile
        plt.plot(range(x_window[0], x_window[1]), x_profile, label=f'x_profile, x_center={x_centroid: .1f}')
        # Add a horizontal line at the threshold
        plt.axhline(y=x_threshold, color='r', linestyle='--', label='Threshold (20% of max)')
        # Set the x-axis limits
        plt.xlim(x_window[0], x_window[1])
        # Add labels and title
        plt.xlabel('ROI_x in pixels')
        plt.ylabel('Pixel grayscale value')
        plt.title(f'Length={exit_pupil_length}pixels={pixel_size*exit_pupil_length:.2f}mm')
        # Add a legend and a grid
        plt.legend()
        plt.grid()

        # Create the second subplot for y_profile
        plt.subplot(1, 2, 2)  # 1 row, 2 columns, second plot    
        # Plot y_profile
        plt.plot(range(y_window[0], y_window[1]), y_profile, label=f'y_profile, y_center={y_centroid:.1f}')
        # Add a horizontal line at the threshold
        plt.axhline(y=y_threshold, color='r', linestyle='--', label='Threshold (20% of max)')
        # Set the x-axis limits
        plt.xlim(y_window[0], y_window[1])
        # Add labels and title
        plt.xlabel('ROI_y in pixels')
        plt.ylabel('Pixel grayscale value')
        plt.title(f'Width={exit_pupil_width}pixels={pixel_size*exit_pupil_width: .2f}mm')
        # Add a legend and grid and a super title
        plt.legend()
        plt.grid()
        plt.suptitle(title+f'_Exit pupil location h={pixel_size*h_centroid: .2f}mm, v={pixel_size*v_centroid: .2f}mm')

        #Get the date and time
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        image_name = os.path.join('images', title+f'_exit_pupil_ROI_profiles_{current_time}.jpg')
        plt.savefig(image_name)

        print(f"Exit pupil grayscale profile and metrics are saved at {image_name}")
    
    return exit_pupil_dict

def get_border_brightness(Y_image:np.ndarray, fov_zones:dict, title='', save_image=False) ->float:

    '''
    Calculate border brightness for a Y tristimulus image, normalized by the average brightness inside FOV
    input: 
        Y_image: an 2D array representing a Y tristimulus image
        fov_zones: dict of FOV zone masks {"full FOV": np.ndarray, "zoneA":np.ndarray,"zoneB":np.ndarray,"zoneC":np.ndarray}
        save_image: a boolean to flag whether to save the processed image
    output:
        A float value of the maximum brightness outside of FOV divided by the average brightness in zoneA                                                            
    '''
    if not isinstance(Y_image, np.ndarray):
        print('Y_dict is not a dictionary')
        return False
    #Check if full FOV is in fov_zones
    if not "full FOV" in fov_zones:
        print("fov_zones is not a dictionary containing at least 'full FOV'")
        return False    
    #Check if fov_zone["full FOV"] mask is a boolean numpy array
    if isinstance(fov_zones["full FOV"],np.ndarray) and np.isin(fov_zones["full FOV"], [True,False]).all():
        pass
    else:
        print("full FOV mask is not a boolean numpy array")
        return False
    #Check if fov_zone mask is same size as the Y_image
    if np.shape(Y_image) != np.shape(fov_zones["full FOV"]):
        print("full FOV mask is not the same shape as Y_image")

    # Normalize Y_image and mask to 8-bit grayscale
    image = Y_image/np.max(Y_image)
    image = np.clip(image*255, 0, 255)
    image = image.astype(np.uint8)
    image = cv2.normalize(image, None, 0, 2**8, cv2.NORM_MINMAX)  

    mask = fov_zones['full FOV'].copy()
    mask = np.clip(mask*255, 0, 255)
    mask = mask.astype(np.uint8)
    mask = cv2.normalize(mask, None, 0, 2**8, cv2.NORM_MINMAX)  

    # Resize the Y_image and the mask by 8-fold for effective averaging and better repeatability
    h, w = Y_image.shape
    resize = (int(w/8),int(h/8))
    image = cv2.resize(image, resize, interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, resize, interpolation=cv2.INTER_AREA)
    mask = (mask>0) # convert the array back to boolean
    #Use util.get_FOV_zone_data function to apply full FOV mask to Y_image input data
    masked_brightness = util.get_FOV_zone_data(image, mask)    
    #Calculate the mean brightness value for the masked data, ignoring nan values
    mean_brightness_fov = np.nanmean(masked_brightness)
    #Use util.get_FOV_zone_data function to get the data outside of FOV
    masked_brightness = util.get_FOV_zone_data(image, ~mask)    
    #Calculate the mean brightness value for the masked data, ignoring nan values
    max_brightness_outfov = np.nanmax(masked_brightness)

    if(save_image):
        #Get the date and time and set the image file name
        current_time = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')  
        image_name = os.path.join('images', title+f'_border_brightness_{current_time}.jpg')
        plt.imshow(masked_brightness)
        plt.title(title+f'_border brightness outside FOV')             
        plt.savefig(image_name)
    return max_brightness_outfov/mean_brightness_fov


def get_masks_DSA(image: np.ndarray, analysis_config: dict, station_config: dict) -> dict:
    """
    This function returns FOV zone masks for the input image. 
    The masks are for Artemis DSA with parameters defined in the analyzers.config file.
    The function also return a list of x,y,w,h parameters for a cropping angle 
    of the active area full FOV in the iamge.
    Inputs:
        image: numpy array containing the measured image
        analysis_config: dictionary containing analysis config parameters
        station_config: dictionary containing station config parameters
    Output:
        A dictionary containing the FOV zone masks as boolean numpy arrays.
        {
        'full FOV': np.ndarray,
        'zoneA': np.ndarray,
        'zoneB': np.ndarray,
        'zoneC': np.ndarray
        }
        A list containing integers for parameters of a cropping rectangle for the full FOV
        after image rotation.
        [x,y,w,h]

    This funtion is the same as get_masks, but with the full FOV mask area defined using the parameters
    target_hFOV and target_vFOV defined in the config file.

    The function first detects the display area in the input image to find the full FOV boundaries.
    The full FOV mask is generated by forming a rectangle centered at the center of the detected display area, with 
    width and height defined by target_hFOV and target_vFOV.
    Then masks for zoneA, zoneB, and zoneC are generated using the parameters defined in analyzers.config.
    A plot of the zones is saved if save_image set to True in config.
    """
    # Validate the input image
    if not isinstance(image, np.ndarray):
        print("image is not a numpy array")
        return False
    if image.ndim == 3 and image.dtype == np.uint8:
        # Convert RGB to grayscale
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif not image.ndim == 2:
        print("image is not a 2D numpy array")
        return False
    if not np.issubdtype(image.dtype, np.number):
        print("image does not contain numerical data")
        return False
    # Validate the analysis_config dictionary
    required_analysis_keys = ["rotation_angle", "target_hFOV", "target_vFOV", 
                              "zoneB_h1", "zoneB_h2", "zoneB_h3", "zoneB_h4", 
                              "zoneB_v1", "zoneB_v2", "zoneB_v3", "zoneB_v4", "zoneA_FOV", "save_image"]
    if not all(key in analysis_config for key in required_analysis_keys):
        print("analysis_config is missing one or more required keys")
        return False
    if not isinstance(analysis_config["rotation_angle"], (int, float)):
        print("rotation_angle in analysis_config is not a numerical value")
        return False
    if not all(isinstance(analysis_config[key], (int, float)) and analysis_config[key] >= 0 for key in required_analysis_keys[1:12]):
        print("One or more values in analysis_config are not non-negative numerical values")
        return False
    if analysis_config["zoneA_FOV"] >= min(analysis_config["target_hFOV"], analysis_config["target_vFOV"]):
        print("zoneA_FOV in analysis_config is not smaller than target_hFOV and target_vFOV")
        return False
    if analysis_config["save_image"] not in ["True", "False"]:
        print("save_image in analysis_config is not 'True' or 'False'")
        return False
    # Validate the station_config dictionary
    required_station_keys = ["pixel_size", "binning", "focal_length"]
    if not all(key in station_config for key in required_station_keys):
        print("station_config is missing one or more required keys")
        return False
    if not isinstance(station_config["pixel_size"], (int, float)) or station_config["pixel_size"] <= 0:
        print("pixel_size in station_config is not a positive numerical value")
        return False
    if not isinstance(station_config["binning"], int) or station_config["binning"] <= 0:
        print("binning in station_config is not a positive integer")
        return False
    if not isinstance(station_config["focal_length"], (int, float)):
        print("focal_length in station_config is not a numerical value")
        return False
    
    # Set show_result to True to plot out image processing steps for debugging
    show_result = False

    #Save a copy of the original image data before processing
    original_image = np.copy(image)

    #Rotate image with config parameter
    image = util.rotate_array(image,analysis_config["rotation_angle"])

    # Check if the image is a floating-point image
    if image.dtype.kind == 'f':
        # Normalize the image to the range [0, 255] and convert to 8-bit integer
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    #Initialize the mask dict
    masks_dict = {}

    ###################################################################################################################################################
    #Find the rectangular boundary of the display area in the measurement image

    # Apply thresholding
    _, thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    #thresh = cv2.GaussianBlur(thresh, (155, 155), 0)

    if show_result == True:
        plt.imshow(thresh)
        plt.show()

    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if show_result == True:
        # Draw bounding boxes around the contours
        image_with_boxes = image.copy()
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(image_with_boxes, (x, y), (x+w, y+h), (255,255,255), 3)
        plt.imshow(image_with_boxes)
        plt.show()

    # Target aspect ratio
    target_aspect_ratio = analysis_config["target_hFOV"] / analysis_config["target_vFOV"]
    #Target angle area = 
    target_angle_area = analysis_config["target_hFOV"]*analysis_config["target_vFOV"]
    # Initial aspect ratio window (you can adjust this value)
    window = 0.2
    # Set the timeout (seconds)
    timeout = 60
    start_time = time.time()
    while True:
        # Filter contours based on aspect ratio
        filtered_contours = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            aspect_ratio = w / h

            #Get contour dimensions in angle
            contour_x1 = x - (np.shape(image)[1]/2)
            contour_x2 = x + w - (np.shape(image)[1]/2)
            contour_y1 = y - (np.shape(image)[0]/2)
            contour_y2 = y + h - (np.shape(image)[0]/2)
            contour_x1_FOV = math.degrees(math.atan(contour_x1*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            contour_x2_FOV = math.degrees(math.atan(contour_x2*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            contour_y1_FOV = math.degrees(math.atan(contour_y1*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            contour_y2_FOV = math.degrees(math.atan(contour_y2*station_config["pixel_size"]*station_config["binning"]/station_config["focal_length"]))
            contour_w_angle = contour_x2_FOV - contour_x1_FOV
            contour_h_angle = contour_y2_FOV - contour_y1_FOV

            # Calculate the angular area of the contour
            contour_angle_area = contour_w_angle * contour_h_angle
            # Check if the contour is within the window for both aspect ratio and angular area
            #if abs(aspect_ratio - target_aspect_ratio) <= window and abs(1 - (contour_angle_area/target_angle_area)) <= window:
            if abs(1 - (contour_angle_area/target_angle_area)) <= window:
                filtered_contours.append(cnt)
        # If only one contour is found or the timeout is reached, break the loop
        if len(filtered_contours) == 1:
            break
        if time.time() - start_time > timeout:
            print("get_masks timed out. Active area not found.")
            return False
        # If no contours found, widen the window
        if len(filtered_contours) == 0:
            window *= 1.1
        # If more than one contour found, narrow the window
        if len(filtered_contours) > 1:
            window *= 0.9
    # The final contour
    final_contour = filtered_contours[0]

    # If a contour was found, draw the bounding rectangle
    if final_contour is not None:
        contour_x, contour_y, contour_w, contour_h = cv2.boundingRect(final_contour)
        cv2.rectangle(image, (contour_x, contour_y), (contour_x+contour_w, contour_y+contour_h), (255, 255, 255), 2)

    if show_result == True:
        # Display the image
        plt.imshow(image)
        plt.show()

    #Define x, y, w, h of the full FOV mask, centered around the center of the bounding rectangle
    contour_center_x = contour_x + round(contour_w/2)
    contour_center_y = contour_y + round(contour_h/2)
    #w = round(2*math.tan((analysis_config["target_hFOV"]))*station_config["focal_length"]/(contour_center_x*station_config["pixel_size"]*station_config["binning"]))
    #h = round(2*math.tan((analysis_config["target_vFOV"]))*station_config["focal_length"]/(contour_center_y*station_config["pixel_size"]*station_config["binning"]))
    w = round(2*station_config["focal_length"] * math.tan(math.radians(analysis_config["target_hFOV"] / 2)) / (station_config["pixel_size"] * station_config["binning"]))
    h = round(2*station_config["focal_length"] * math.tan(math.radians(analysis_config["target_vFOV"] / 2)) / (station_config["pixel_size"] * station_config["binning"]))
    x = contour_center_x - round(w/2)
    y = contour_center_y - round(h/2)
    print(contour_x, contour_y, contour_w, contour_h)
    print(x,y,w,h)

    # Create a blank image of the same size as the original image
    mask_full_FOV = np.zeros(image.shape[:2], dtype=np.uint8)
    #Mask off the area outside the detected contour region
    mask_full_FOV[:,:x] = 255
    mask_full_FOV[:y,:] = 255
    mask_full_FOV[:,x+w:] = 255
    mask_full_FOV[y+h:,:] = 255

    mask_full_FOV = mask_full_FOV.astype(bool)

    #Store the rectangular boundary as the full FOV mask.
    masks_dict["full FOV"] = mask_full_FOV

    #Create roi list for x, y, w, h
    roi = [x, y, w, h]

    ###################################################################################################################################################
    #Generate the mask for zoneC. Crop the corners of the FOV defined in config

    corner_widths = np.array([analysis_config["zoneB_h1"],analysis_config["zoneB_h2"],analysis_config["zoneB_h3"],analysis_config["zoneB_h4"]])
    corner_widths = (corner_widths/analysis_config["target_hFOV"])*w
    corner_lengths = np.array([analysis_config["zoneB_v1"],analysis_config["zoneB_v2"],analysis_config["zoneB_v3"],analysis_config["zoneB_v4"]])
    corner_lengths = corner_lengths/analysis_config["target_vFOV"]*h

    w = w - 1
    h = h - 1 
    top_left = np.array([[[x, y], [corner_widths[0]+x, y], [x, corner_lengths[0]+y]]], dtype=np.int32)
    top_right = np.array([[[w-corner_widths[1]+x, y], [w+x, y], [w+x, corner_lengths[1]+y]]], dtype=np.int32)
    bottom_left = np.array([[[x, h+y-corner_lengths[2]], [x, h+y], [corner_widths[2]+x, h+y]]], dtype=np.int32) 
    bottom_right = np.array([[[w+x, h+y], [w+x, h-corner_lengths[3]+y], [w-corner_widths[3]+x,h+y]]], dtype=np.int32)

    if show_result == True:
        # Draw the triangles
        cv2.fillPoly(image, top_left, (0, 0, 0))
        cv2.fillPoly(image, top_right, (0, 0, 0))
        cv2.fillPoly(image, bottom_left, (0, 0, 0))
        cv2.fillPoly(image, [bottom_right], (0, 0, 0))

        # Display the image
        plt.imshow(image)
        plt.show()

    # Create a blank image of the same size as the original image
    mask_zoneC = np.zeros(image.shape[:2], dtype=np.uint8)
    # Draw the triangles on the mask
    cv2.fillPoly(mask_zoneC, top_left, 255)
    cv2.fillPoly(mask_zoneC, top_right, 255)
    cv2.fillPoly(mask_zoneC, bottom_left, 255)
    cv2.fillPoly(mask_zoneC, bottom_right, 255)
    # Mask off the areas outside the contour (display) region
    mask_zoneC[0:,0:x] = 255
    mask_zoneC[0:y,0:] = 255
    mask_zoneC[0:,x+w:] = 255
    mask_zoneC[y+h:,0:] = 255
    # Convert the mask to a boolean mask
    mask_zoneC = mask_zoneC.astype(bool)
    if show_result == True:
        # Display the mask
        plt.imshow(mask_zoneC, cmap='gray')
        plt.show()

    #Store the mask for zoneC in dict
    masks_dict["zoneC"] = mask_zoneC

    ###################################################################################################################################################
    #Generate the mask for zoneA. Center FOV circular region with diameter defined in config

    zoneA_diameter_FOV = analysis_config["zoneA_FOV"]
    zoneA_diameter_relative = zoneA_diameter_FOV/analysis_config["target_hFOV"]
    zoneA_diameter_pixels = zoneA_diameter_relative*w

    mask_zoneA = np.zeros(image.shape[:2], dtype=np.uint8)
    mask_zoneA[:,:] = 255
    # Get the detected contour
    center_x = x + w//2
    center_y = y + h//2
    # Draw the circle on the mask
    cv2.circle(mask_zoneA, (center_x, center_y), int(zoneA_diameter_pixels // 2), 0, -1)
    # Convert the mask to a boolean mask
    mask_zoneA = mask_zoneA.astype(bool)
    if show_result == True:
        # Display the mask
        plt.imshow(mask_zoneA, cmap='gray')
        plt.show()

    #Store the mask for zoneA in dict
    masks_dict["zoneA"] = mask_zoneA

    ###################################################################################################################################################
    #Generate the mask for zoneB. Defined as the area outside zoneA and inside zoneC.

    # Create the mask for zoneB
    mask_zoneB = np.bitwise_not(np.bitwise_and(np.bitwise_not(mask_zoneC), mask_zoneA))
    if show_result == True:
        # Display the mask
        plt.imshow(mask_zoneB, cmap='gray')
        plt.show()

    #Store the mask for zoneB in dict
    masks_dict["zoneB"] = mask_zoneB

    ###################################################################################################################################################
    #Rotate the image and masks back to the original orientation of the input image. Crop the image and masks back to original resolution
    # Get the center of the original image
    original_center = (np.array(original_image.shape[:2][::-1]) - 1) / 2.0
    # Rotate the image and masks back to the original orientation
    rotation_angle = -analysis_config["rotation_angle"]
    image = util.rotate_array(image, rotation_angle)
    for key in masks_dict:
        masks_dict[key] = util.rotate_array(masks_dict[key], rotation_angle)
    # Get the center of the rotated image
    rotated_center = (np.array(image.shape[:2][::-1]) - 1) / 2.0
    # Calculate the top left corner of the cropping rectangle
    top_left = np.round(rotated_center - original_center).astype(int)
    # Calculate the bottom right corner of the cropping rectangle
    bottom_right = top_left + np.array(original_image.shape[:2][::-1])
    # Crop the image and masks to the original resolution
    image = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    for key in masks_dict:
        masks_dict[key] = masks_dict[key][top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

    ###################################################################################################################################################
    #Save output image result if save_image set to True.

    if analysis_config["save_image"] == "True":
        f"images\artemis masks.png"

    if analysis_config["save_image"] == "True":
        # Create a blank RGB image
        background_image = cv2.normalize(original_image, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        rgb_image = np.zeros((*image.shape, 3), dtype=np.uint8)
        rgb_image[:, :, :] = background_image[:, :, np.newaxis]
        # Erode the masks and subtract from the original masks to get the outlines
        outline_width = int(background_image.shape[0]*0.002899)
        outline_zoneA = binary_dilation(masks_dict["zoneA"] & ~binary_erosion(masks_dict["zoneA"]),iterations = outline_width)
        outline_zoneC = binary_dilation(masks_dict["zoneC"] & ~binary_erosion(masks_dict["zoneC"]),iterations = outline_width)
        outline_fullFOV = binary_dilation(masks_dict["full FOV"] & ~binary_erosion(masks_dict["full FOV"]),iterations = outline_width)
        # Draw the outlines on the image
        rgb_image[outline_zoneA] = [255, 0, 0]  # Red for zoneA
        rgb_image[outline_zoneC] = [0, 0, 255]  # Blue for zoneC
        rgb_image[outline_fullFOV] = [0,255,0] # Green for full FOV
        # Set the edge pixels of rgb_image to 0
        rgb_image[:outline_width, :] = 0
        rgb_image[-outline_width:, :] = 0
        rgb_image[:, :outline_width] = 0
        rgb_image[:, -outline_width:] = 0
        # Create a hatched pattern for Zone B
        zoneB_mask = masks_dict["zoneC"] & ~masks_dict["zoneA"]
        hatched_pattern = np.zeros_like(background_image)
        hash_width = outline_width
        for i in range(0, hatched_pattern.shape[0], hash_width*5):  # Adjust the stride to change the density of the hatching
            hatched_pattern[i:i+hash_width, :] = 255
        hatched_zoneB = cv2.bitwise_and(hatched_pattern, ~masks_dict["zoneB"].astype(np.uint8) * 255)
        hatched_zoneB = np.where(hatched_zoneB==2,0,hatched_zoneB)*255
        rgb_image[hatched_zoneB == 255] = [0, 255, 255]  # Cyan for Zone B
        # Create the plot
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(rgb_image)
        ax.set_title('DSA Masks')
        # Create the legend
        red_patch = mpatches.Patch(color='red', label='Zone A')
        blue_patch = mpatches.Patch(color='blue', label='Zone C')
        green_patch = mpatches.Patch(color='green', label='full FOV')
        cyan_patch = mpatches.Patch(color='cyan', label='Zone B')
        ax.legend(handles=[red_patch, green_patch, blue_patch, cyan_patch], loc='upper right', bbox_to_anchor=(1.15, 1))
        ax.set_xlim(2, original_image.shape[1] - 2)
        ax.set_ylim(original_image.shape[0] - 2, 2)
        plt.tight_layout()
        # Save the plot
        plt.savefig(f"images/DSA masks {datetime.now().strftime('%Y-%m-%d %H-%M-%S')}.png", bbox_inches='tight')
        plt.cla()

    return masks_dict, roi

def get_keystone_undistortion(image: np.ndarray, analysis_config: dict, station_config: dict):
    '''
    Correct parallax keystone distortion due to cant angle for an image
    input: 
        image: an image stored as np.ndarray
        analysis_config: dictionary containing parameters for the undistortion
        station_config: dictionary containing station parameters relevant for the undistortion
    output:
        An undistorted image of the same size as the input image                                                       
    '''

    if analysis_config['EyeSide'] == 'Left':
        sign = -1
    else:
        sign = 1
    
    if image.ndim > 2:
        h, w, d = image.shape
    else:
        h, w = image.shape
        d = 1
    efl = station_config['focal_length']  # 22.9
    binning = station_config['binning'] # 4
    dw = 0.5 * w * binning * station_config['pixel_size']  # Width of p_in in space
    sin_c = np.sin(sign * np.radians(analysis_config['CantAngle']))
    cos_c = np.cos(sign * np.radians(analysis_config['CantAngle']))

    # Define 4 corners of a rectangle on waveguide plane:
    if analysis_config['EyeSide'] == 'Left':
        p_in = [[w / 4, h / 4], [w / 4, 3 * h / 4], [3 * w / 4, h / 4], [3 * w / 4, 3 * h / 4]]
        # Projected onto viewing plane become:
        p_out = [[w / 4 * (2 - cos_c), h / 2 * (efl - dw * sin_c) / (2 * efl - dw * sin_c)],
                 [w / 4 * (2 - cos_c), h / 2 * (3 * efl - dw * sin_c) / (2 * efl - dw * sin_c)],
                 [w / 4 * (2 + cos_c), h / 2 * efl / (2 * efl - dw * sin_c)],
                 [w / 4 * (2 + cos_c), h / 2 * (3 * efl - 2 * dw * sin_c) / (2 * efl - dw * sin_c)]]
    else:
        p_in = [[w / 4, 3 * h / 4], [w / 4, h / 4], [3 * w / 4, 3 * h / 4], [3 * w / 4, h / 4]]
        # Projected onto viewing plane become:
        p_out = [[w / 4 * (2 - cos_c), h / 2 * (3 * efl - dw * sin_c) / (2 * efl - dw * sin_c)],
                 [w / 4 * (2 - cos_c), h / 2 * (efl - dw * sin_c) / (2 * efl - dw * sin_c)],
                 [w / 4 * (2 + cos_c), h / 2 * (3 * efl - 2 * dw * sin_c) / (2 * efl - dw * sin_c)],
                 [w / 4 * (2 + cos_c), h / 2 * efl / (2 * efl - dw * sin_c)]]

    image_corr = np.empty_like(image)
    # Projective transformation
    Transf = cv2.getPerspectiveTransform(np.float32(p_out), np.float32(p_in))
    unDistortedExtend = cv2.perspectiveTransform(np.float32([[0, 0], [0, h], [w, h], [w, 0]]).reshape(-1, 1, 2),
                                                 Transf).squeeze()
    sizeX, sizeY = np.ceil(np.max(unDistortedExtend, axis=0) - np.min(unDistortedExtend, axis=0))
    transl = np.eye(3)
    transl[0, 2] = -np.min(unDistortedExtend[:, 0])
    transl[1, 2] = -np.min(unDistortedExtend[:, 1])
    Transf = transl @ Transf

    if d > 1:
        for k in range(image.shape[2]):
            image_corr[:, :, k] = cv2.warpPerspective(image[:, :, k], Transf, (w, h))
    else:
        image_corr = cv2.warpPerspective(image, Transf, (int(sizeX), int(sizeY)))

    # Truncate edges to make image_corr the same size as image
    trim = list(map(lambda x, y: round((x - y) / 2), image_corr.shape, image.shape))
    if trim != [0, 0]:
        image_corr = image_corr[trim[0]:trim[0]+h, trim[1]:trim[1]+w]

    return image_corr
    