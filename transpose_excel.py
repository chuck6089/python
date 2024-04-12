# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 14:08:36 2024

@author: xzd6089
"""

import tkinter as tk
from tkinter import filedialog
import os

import pandas as pd

def transpose_excel(input_file, output_file):
    # Read the Excel file into a DataFrame
    df = pd.read_excel(input_file)
    
    # Transpose the DataFrame
    df_transposed = df.transpose()
    
    # Write the transposed DataFrame to a new Excel file
    df_transposed.to_excel(output_file, index= True, header= False)

def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    initial_dir = "X:\IQT2-Display\Devices\Lumus\Lumus_SN75010\SCS33\Repeatability_AutoDP processing"

    filetypes = (("XLS files", "*.xlsx"),("CSV files",".csv"), ("Text files", "*.txt"))
    filename = filedialog.askopenfilename(initialdir=initial_dir, filetypes=filetypes, title="Choose MTF raw file")
    
    return filename

def get_directory_filename_and_extension(file_path):
    directory = os.path.dirname(file_path)
    filename_with_extension = os.path.basename(file_path)
    filename, extension = os.path.splitext(filename_with_extension)
    return directory, filename, extension

# Example usage
input_file = choose_file()
directory,filename,extension = get_directory_filename_and_extension(input_file)
output_file = directory + '\\' + filename + '_rows.xlsx'
transpose_excel(input_file, output_file)
