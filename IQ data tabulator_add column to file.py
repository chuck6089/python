# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 13:26:43 2024

@author: xzd6089
"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import csv

def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    initial_dir = "X:\IQT2-Display\Devices\Lumus\Lumus_SN75014\AutoDP processing\10 ADK3 left WG_April2024"

    filetypes = (("CSV files",".csv"), ("XLS files", "*.xlsx"), ("Text files", "*.txt"))
    filename = filedialog.askopenfilename(initialdir=initial_dir, filetypes=filetypes, title="Choose IQT csv file")
    
    return filename

def get_directory_filename_and_extension(file_path):
    directory = os.path.dirname(file_path)
    filename_with_extension = os.path.basename(file_path)
    filename, extension = os.path.splitext(filename_with_extension)
    return directory, filename, extension

def create_excel_file(filename):
    if not os.path.exists(filename):
        df = pd.DataFrame()
        df.to_excel(filename, index=False)
        print(f"Excel file '{filename}' created successfully.")
    else:
        print(f"Excel file '{filename}' already exists.")

Metrics_list_path = r"G:\Shared drives\Display & Optics - GRWG\Optical Metrology\software\pyAutoDP\Req_ID_mapping_Capstone.csv"
Metrics_tabulate_list_path = r"G:\Shared drives\Display & Optics - GRWG\Optical Metrology\software\pyAutoDP\Metrics to tabulate.xlsx"

ReqID_mapping = pd.read_csv(Metrics_list_path)


path = choose_file()

directory,filename,extension = get_directory_filename_and_extension(path)

outputfile = r"C:\Users\xzd6089\Documents\Data\Capstone\IQT_metrics_tabulated.xlsx"


if not os.path.exists(outputfile):
    #create_excel_file(outputfile)
    Metrics_to_tabulate = pd.read_excel(Metrics_tabulate_list_path)
    description_column = []

    for Req_ID in Metrics_to_tabulate['Req_ID']:
        Metric = ReqID_mapping.loc[ReqID_mapping['Req_ID'] == Req_ID, 'Metrics'].iloc[0]
        description_column.append(Metric)
        
    Metrics_to_tabulate.insert(1,'Metrics description',description_column)
else:
    Metrics_to_tabulate = pd.read_excel(outputfile)

#input_data = pd.read_excel(path,sheet_name = "Primary with flare")

input_data = pd.read_csv(path)

#ReqID_column = Metrics_to_tabulate['Req_ID']



DUTs = input_data['dut_id']

for DUT in DUTs:  #insert a row for each DUT
    value_column = []
    for metric in Metrics_to_tabulate['Metrics description']:
        value = input_data.loc[input_data['dut_id'] == DUT, metric].iloc[0]
        value_column.append(value)
    Metrics_to_tabulate[DUT] = value_column
    
Metrics_to_tabulate.to_excel(outputfile ,index = False)