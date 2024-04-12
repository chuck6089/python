# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 11:59:10 2017

@author: zhixu
"""

from scipy.optimize import curve_fit
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import cv2
#import tkFileDialog as FD



def seddist(r,k1,k2,k3):
    return 1 + k1 * np.power(r,2) + k2 * np.power(r,4) + k3 * np.power(r,6)

def fisheyetheta(theta,k1,k2,k3,k4):
    return 1 + k1 * np.power(theta,2) + k2 * np.power(theta,4) + k3 * np.power(theta,6) + k4 * np.power(theta,8)


def readxls(filename, col_name_1, col_name_2):
    df = pd.read_excel(filename)
    r = df[col_name_1]
    y = df[col_name_2]
    return r,y


def distortion_coefficients(distortionfile, r_col_name = 'Angle(rad)', y_col_name = 'Ref_height_ratio', Ftheta = False):
    r, y = readxls(distortionfile, r_col_name, y_col_name)
     
        
    if(Ftheta):
        #thetaX = [x * 180/3.1415926 for x in r]  #degrees
        thetaX = r
        thetaY = y
        popt, pcov = curve_fit(fisheyetheta, thetaX, thetaY)
        k1,k2,k3,k4 = popt
        plt.plot(thetaX, thetaY, 'b+', label='data')
        plt.plot(thetaX, fisheyetheta(thetaX, *popt), 'r-',label='fit')
        print("F-Theta")
        print('K1:',k1)
        print('K2:',k2)
        print('K3',k3)
        print('K4',k4)
        print(popt)
    else:                                        
        popt, pcov = curve_fit(seddist, r, y)
        k1,k2,k3 = popt
        plt.plot(r, y, 'b+', label='data')
        plt.plot(r, seddist(r, *popt), 'r-',label='fit')
        print("F-Tan")
        print('K1:',k1)
        print('K2:',k2)
        print('K3:',k3)
        print(popt)        
    print(distortionfile)
    return popt

def undistort(image, dist_coef, Ftheta = False, pixelsize = 22, EFL = 23.8):  #pixelsize unit it um, EFL unit in mm, Use fisheye(Ftheta) distortion model if Ftheta is true
    h, w = image.shape[0],image.shape[1]
    K_mtx = np.array([[EFL*1000/pixelsize,0,w/2],
                     [0,EFL*1000/pixelsize,h/2],
                     [0,0,1]])
    if not Ftheta:
        dist = np.array([dist_coef[0],dist_coef[1],0,0,dist_coef[2]])
        undistorted = cv2.undistort(image, K_mtx, dist, None, K_mtx)
    else:
        dist = np.array([dist_coef[0],dist_coef[1],dist_coef[2],dist_coef[3]])
        undistorted = cv2.fisheye.undistortImage(image, K_mtx, dist, K_mtx)
    return undistorted,K_mtx

    

if __name__ == "__main__":
    filename = 'distortion_87deg_v2Lens.xlsx'
    image = cv2.imread('crosshairs_image.png')
    Ftheta = True
    coeffs = distortion_coefficients(filename, 'Angle(rad)','Ref_height_ratio(Ftheta)', Ftheta)
    undistorted, K_mtx = undistort(image, coeffs, Ftheta = True, pixelsize = 22, EFL = 23.8)