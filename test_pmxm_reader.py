# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:54:02 2023

@author: xzd6089
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import image_utils
import cv2
import analyzer_portable as analyzer
from external.prometric.pmxm_reader_utils import pmxm_reader_utils

ix = -1
iy = -1


def convert2img(array):
    array[array < 0] = 0
    array_max = np.max(array)
    image = np.uint8(array*(245.0/array_max))
    cv2.imwrite('tempimage.bmp',image)
    image = cv2.imread('tempimage.bmp',cv2.IMREAD_GRAYSCALE)   #this is to avoid a weird bug by cv2.rectangle, the image has to be reloaded in order to plot
    return image
 
def find_measurementIDs(listofmeasurement, keyword):
    indexarray = []
    for measurement in listofmeasurement:
        if keyword in measurement[1]:
            indexarray.append(measurement[0])
    return indexarray
            
def readXYZ_radiant(path, keyword):
    result = []
    listofmeasurement = pmxm_reader.get_meas_list(path)
    index = find_measurementIDs(listofmeasurement, keyword)
    for ID in index:
        XYZ = pmxm_reader.get_xyz_arrays(pmxm_file = path, meas_id = ID)
        result.append(XYZ)
    return result


pmxm_reader = pmxm_reader_utils()

listofmeasurement = pmxm_reader.get_meas_list('G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\Capstone_CS1 220-21-300 V1M2.S31_11092023\Lumus_SN75017_Capstone_0deg_noND_wiPolar.pmxm')

#xyz = pmxm_reader.get_xyz_arrays(pmxm_file = 'G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\Capstone_CS1 220-21-300 V1M2.S31_11092023\Lumus_SN75017_Capstone_0deg_noND_wiPolar.pmxm', meas_id = 1)

path = 'G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\Capstone_CS1 220-21-300 V1M2.S31_11092023\Lumus_SN75017_Capstone_0deg_noND_wiPolar.pmxm'
keyword = 'Green Neg'

xyz = readXYZ_radiant(path, keyword)

Y = xyz[0][1]

data = convert2img(Y) 

crossdata = readXYZ_radiant(path, 'Green Cross')
crossimage = convert2img(crossdata[0][1])


rows, cols = data.shape

# Create x and y coordinates for the plot
x = np.arange(cols)
y = np.arange(rows)

# Create a meshgrid from x and y
X, Y = np.meshgrid(x, y)

# Plot the 2D profile using a contour plot
plt.contour(X, Y, data, cmap='viridis')
plt.colorbar()  # Add a colorbar for reference
plt.title('2D Profile')
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.show()


template = cv2.imread('crosshair_template_2.png',cv2.IMREAD_GRAYSCALE)
coords = image_utils.get_crosshairs(crossimage,template)
dist_coef = image_utils.distortion_coefficients("distortion_87deg_v2Lens.xlsx")

undistort,K_mtx = image_utils.undistort(data, dist_coef)
cv2.imwrite('undistort.bmp',undistort)

center_coord = (np.mean(coords[:,0]),np.mean(coords[:,1]))

#Find contrast~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

current_directory = os.getcwd()
obj = analyzer.portable_analyzer(current_directory)
checkerdata  = readXYZ_radiant(path, 'Green Pos')
checkerimage = convert2img(checkerdata[0][1])
negcheckerdata = readXYZ_radiant(path, 'Green Neg')
negcheckerimage = convert2img(negcheckerdata[0][1])


#Find corners of checkers
checkerboard_size = (17, 17)
ret, corners = cv2.findChessboardCorners(checkerimage, checkerboard_size, None)

# If corners are found, draw them on the image
if ret:
    # Draw the corners on the image
    cv2.drawChessboardCorners(checkerimage, checkerboard_size, corners, ret)

    # Display the image with corners
    cv2.imshow('Corners', checkerimage)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Corners not found in the image.")



#End find corners of checkers

checkerimage,K_mtx = image_utils.undistort(checkerimage, dist_coef)
negcheckerimage,K_mtx = image_utils.undistort(negcheckerimage, dist_coef)

obj.load_checker_negchecker(checkerimage,negcheckerimage)
obj.meancontrast(18, 18, squaresize_m = 37.5, squaresize_n = 38,  center_m = center_coord[0] - 34, center_n = center_coord[1] + 30)

cv2.imwrite('checker.png',obj.checker)
cv2.imwrite('negchecker.png',obj.negchecker)

analyzer.save_to_csv(obj.contrastcsv, "Ucalibrated_contrastmap.csv", mode = 'w')
meandata = [
    ["Mean contrast",obj.meancontrast],
    ["Harmonic mean contrast",obj.hmeancontrast]
    ]
analyzer.save_to_csv(meandata, "Ucalibrated_contrastmap.csv", mode = 'a')
