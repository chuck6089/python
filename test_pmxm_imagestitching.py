# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 15:54:02 2023

@author: xzd6089
"""

# %% import lib

import os
import numpy as np
import matplotlib.pyplot as plt
import image_utils
import analyzer_portable as analyzer
from external.prometric.pmxm_reader_utils import pmxm_reader_utils
import cv2

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

# %% contrast & crosshair

pmxm_reader = pmxm_reader_utils()

listofmeasurement = pmxm_reader.get_meas_list('G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\Capstone_CS1 220-21-300 V1M2.S31_11092023\Lumus_SN75017_Capstone_0deg_noND_wiPolar.pmxm')

#xyz = pmxm_reader.get_xyz_arrays(pmxm_file = 'G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\Capstone_CS1 220-21-300 V1M2.S31_11092023\Lumus_SN75017_Capstone_0deg_noND_wiPolar.pmxm', meas_id = 1)


path = 'G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\Capstone_CS1 220-21-300 V1M2.S31_11092023\Lumus_Capstone_0deg_ND.pmxm'
#slbpath = 'G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\SLB\Lumus_SN75017_Projector_ND2.pmxm'
slbpath = 'G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\SLB\Lumus_SN75017_Projector_ND2_aftColorCal_4X4binning.pmxm'

keyword = 'Green Neg'

xyz = readXYZ_radiant(path, keyword)

Y = xyz[0][1]

data = convert2img(Y) 

# crossdata = readXYZ_radiant(path, 'Green Cross')
# crossimage = convert2img(crossdata[0][1])


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
#coords = image_utils.get_crosshairs(crossimage,template)
dist_coef = image_utils.distortion_coefficients("distortion_87deg_v2Lens.xlsx")

undistort,K_mtx = image_utils.undistort(data, dist_coef)
cv2.imwrite('undistort.bmp',undistort)

#center_coord = (np.mean(coords[:,0]),np.mean(coords[:,1]))

#Find contrast~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

current_directory = os.getcwd()
obj = analyzer.portable_analyzer(current_directory)
checkerdata  = readXYZ_radiant(path, 'White Pos')
checkerimage = convert2img(checkerdata[0][1])
negcheckerdata = readXYZ_radiant(path, 'White Neg')
negcheckerimage = convert2img(negcheckerdata[0][1])


#checkerimage,K_mtx = image_utils.undistort(checkerimage, dist_coef)
#negcheckerimage,K_mtx = image_utils.undistort(negcheckerimage, dist_coef)

# checkerimage = image_utils.rotate_image(checkerimage,-3)
# negcheckerimage = image_utils.rotate_image(negcheckerimage,-3)


# checkerimage = cv2.flip(checkerimage,0)
# negcheckerimage = cv2.flip(negcheckerimage,0)




obj.load_checker_negchecker(checkerimage,negcheckerimage)
#obj.meancontrast(18, 18, squaresize_m = 37.5, squaresize_n = 38,  center_m = 781.8, center_n = 680.5)


# analyzer.save_to_csv(obj.contrastcsv, "Ucalibrated_contrastmap.csv", mode = 'w')
# meandata = [
#     ["Mean contrast",obj.meancontrast],
#     ["Harmonic mean contrast",obj.hmeancontrast]
#     ]
# analyzer.save_to_csv(meandata, "Ucalibrated_contrastmap.csv", mode = 'a')



#Getting SLB contrast and calibrate the contrast

slbcheckerdata  = readXYZ_radiant(slbpath, 'Blue Pos')
slbcheckerimage = convert2img(slbcheckerdata[0][1])
slbnegcheckerdata = readXYZ_radiant(slbpath, 'Blue Neg')
slbnegcheckerimage = convert2img(slbnegcheckerdata[0][1])


#slbcheckerimage,K_mtx = image_utils.undistort(slbcheckerimage, dist_coef)
#slbnegcheckerimage,K_mtx = image_utils.undistort(slbnegcheckerimage, dist_coef)

slbcheckerimage = image_utils.rotate_image(slbcheckerimage,-3)
slbnegcheckerimage = image_utils.rotate_image(slbnegcheckerimage,-3)


slbcheckerimage = cv2.flip(slbcheckerimage,0)
slbnegcheckerimage = cv2.flip(slbnegcheckerimage,0)


obj.load_SLB(slbcheckerimage,slbnegcheckerimage,slbnegcheckerimage)
#obj.SLB_contrast(18, 18, squaresize_m = 38, squaresize_n = 38,  center_m = 828, center_n = 696)
#obj.meancontrast(18, 18, squaresize_m = 37.5, squaresize_n = 38,  center_m = 781.8, center_n = 680.5)
obj.SLB_contrast(16, 16, squaresize_m = 37.8, squaresize_n = 37.8,  center_m = 832, center_n = 694)
#obj.SLB_contrast(16, 16, squaresize_m = 34, squaresize_n = 33,  center_m = 880, center_n = 625)
obj.meancontrast(16, 16, squaresize_m = 38, squaresize_n = 38,  center_m = 779.8, center_n = 680.5)


obj.contrastcalibration()


print("SLB Checkerboard mean contrast is {:.5f}".format(obj.slb_meancontrast))
print("SLB Checkerboard harmonic mean contrast is {:.5f}".format(obj.slb_hmeancontrast))
print("Ucalibrated Checkerboard mean contrast is {:.5f}".format(obj.meancontrast))
print("Ucalibrated Checkerboard harmonic mean contrast is {:.5f}".format(obj.hmeancontrast))
print("Calibrated Checkerboard mean contrast is {:.5f}".format(obj.calibrated_meancontrast))
print("Calibrated Checkerboard harmonic mean contrast is {:.5f}".format(obj.calibrated_hmeancontrast))


cv2.imwrite('checker.png',obj.checker)
cv2.imwrite('negchecker.png',obj.negchecker)
cv2.imwrite('slbchecker.png',obj.slbchecker)
cv2.imwrite('slbnegchecker.png',obj.slbnegchecker)


analyzer.save_to_csv(obj.slb_contrastcsv, "SLB_contrastmap.csv", mode = 'w')
analyzer.save_to_csv(obj.contrastcsv, "Ucalibrated_contrastmap.csv", mode = 'w')
analyzer.save_to_csv(obj.calibrated_contrastcsv, "Calibrated_contrastmap.csv", mode = 'w')


meandata = [
    ["Mean contrast",obj.meancontrast],
    ["Harmonic mean contrast",obj.hmeancontrast]
    ]
analyzer.save_to_csv(meandata, "Ucalibrated_contrastmap.csv", mode = 'a')

slbmeandata = [
    ["SLB_Meancontrast",obj.slb_meancontrast],
    ["SLB_Harmonic mean contrast",obj.slb_hmeancontrast]
    ]
analyzer.save_to_csv(slbmeandata, "SLB_contrastmap.csv", mode = 'a')

calibrated_meandata = [
    ["Calibrated_Meancontrast",obj.calibrated_meancontrast],
    ["Calibrated_Harmonic mean contrast",obj.calibrated_hmeancontrast]
    ]
analyzer.save_to_csv(calibrated_meandata, "Calibrated_contrastmap.csv", mode = 'a')
 

# %%test image stitching by crosshair


path1 = 'G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\Capstone_CS1 220-21-300 V1M2.S31_11092023\Lumus_Capstone_+5deg_ND.pmxm'
path2 = 'G:\Shared drives\Display & Optics - GRWG\Optical Metrology\IQT2 data\Capstone_CS1 220-21-300 V1M2.S31_11092023\Lumus_Capstone_-5deg_ND.pmxm'

cross1 = readXYZ_radiant(path1, 'Green Cross')
cross2 = readXYZ_radiant(path2, 'Green Cross')

cross1 = convert2img(cross1[0][1])
cross2 = convert2img(cross2[0][1])

stitched_img = image_utils.stitchimage_crosshair(cross1,cross2,template)

cv2.imwrite('stitched.png',stitched_img)

