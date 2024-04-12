# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:51:47 2024

@author: xzd6089

Secondary eyebox MTF data parser

"""

import pandas as pd
import tkinter as tk
from tkinter import filedialog
import os
import csv
import datetime

def choose_file():
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    initial_dir = "X:\SYS-STMTF\Capstone PP1\Schott"

    filetypes = (("CSV files", "*.csv"), ("Text files", "*.txt"))
    filename = filedialog.askopenfilename(initialdir=initial_dir, filetypes=filetypes, title="Choose MTF raw file")
    
    return filename

def get_directory_filename_and_extension(file_path):
    directory = os.path.dirname(file_path)
    filename_with_extension = os.path.basename(file_path)
    filename, extension = os.path.splitext(filename_with_extension)
    return directory, filename, extension

def swap_columns_inplace(df, col1, col2):  #swap two columns
    """
    Swap the positions of two columns in a DataFrame in place.

    Parameters:
        df (DataFrame): The DataFrame containing the columns.
        col1 (str): Name of the first column to swap.
        col2 (str): Name of the second column to swap.
    """
    col1_index = df.columns.get_loc(col1)
    col2_index = df.columns.get_loc(col2)
    df.iloc[:, [col1_index, col2_index]] = df.iloc[:, [col2_index, col1_index]]
    

def change_column_name_inplace(df, old_name, new_name):  #change column names
    """
    Change the name of a column in a DataFrame in place.

    Parameters:
        df (DataFrame): The DataFrame containing the column.
        old_name (str): Name of the column to be renamed.
        new_name (str): New name for the column.
    """
    df.rename(columns={old_name: new_name}, inplace=True)

def get_file_creation_time(file_path):
    # Get creation time of the file in seconds since the epoch
    creation_time_seconds = os.path.getctime(file_path)
    
    # Convert creation time to a datetime object
    creation_time = datetime.datetime.fromtimestamp(creation_time_seconds)
    
    # Convert datetime object to a formatted string
    creation_time_str = creation_time.strftime('%Y-%m-%d %H:%M:%S')
    
    return creation_time_str

delimited = True

path = choose_file()

directory,filename,extension = get_directory_filename_and_extension(path)

#path = r"X:\SYS-STMTF\Capstone PP1\Schott\Measurement_capstone_primary_sys_stmtf_labarflex_ge01_SCS30_20240318_170531\Measurement_capstone_primary_sys_stmtf_labarflex_ge01_SCS30_20240318_170531.csv"

#drop the first 60 rows

creation_time = get_file_creation_time(path)

outputfile = r"C:\Users\xzd6089\Documents\Data\Capstone\MTF_secondaryeyebox_metrics_tabulated.xlsx"

if delimited:
    df = pd.read_csv(path,skiprows= 35,delimiter=';')  #read dataframe with delimiter
else:
    df = pd.read_csv(path,skiprows= 35)  #read dataframe without delimiter


#Split 'Label column' in to two columns: 'position' and 'field angle'
df['Label'] = df['Label'].str.replace('Pos_','Pos')
df['Label'] = df['Label'].str.replace('On_Axis','OnAxis')
#df['Label'] = df['Label'].str.replace('DSA_','DSA')

df[['Position', 'Field angle']] = df['Label'].str.split('_', expand=True)

df.drop(columns=['Label'], inplace = True)

#move last 2 columns into 4th and 5th columns of dataframe

last_two_columns = df.iloc[:, -2:]

df = pd.concat([df.iloc[:, :2], last_two_columns, df.iloc[:, 2:]], axis=1)

df = df.iloc[:, :-2]


#Re-label the Run Label columns with Red/Green/White

# df['Run Label'].replace(1,'Blue',inplace = True)
# df['Run Label'].replace(2,'Green',inplace = True)
# df['Run Label'].replace(3,'Red',inplace = True)

#df.rename(columns={'Run Label': 'Color'}, inplace=True)

#Calculate the average and stdev for each field angle

column_names = df.columns.tolist()

#remove the useless string 

string_to_remove = ' [norm.]';
for i in range(len(column_names)):
    column_name = column_names[i]
    column_name_new = column_name.replace(string_to_remove, "")
    column_name_new = column_name_new.replace(' Dir1 ', '_V')
    column_name_new = column_name_new.replace(' Dir2 ', '_H')
    column_name_new = column_name_new.replace('lp/°', "lp/deg")
    df.rename(columns={column_name: column_name_new}, inplace=True)
    
column_names = df.columns.tolist()

#swap the columns in df

swap_columns_inplace(df, 'MTF_H@ 7.5 lp/deg', 'MTF_V@ 7.5 lp/deg')
swap_columns_inplace(df, 'MTF_H@ 15 lp/deg', 'MTF_V@ 15 lp/deg')
swap_columns_inplace(df, 'CRA X ["]', 'CRA Y ["]')
swap_columns_inplace(df, 'Power_H@ 7.5 lp/deg [dpt]', 'Power_V@ 7.5 lp/deg [dpt]')

#change column names
# change_column_name_inplace(df, 'CRA X ["]', 'Columnnametochange')
# change_column_name_inplace(df, 'CRA Y ["]', 'CRA X ["]')
# change_column_name_inplace(df, 'Columnnametochange', 'CRA Y ["]')

#Calculate the mean and stdev for each color


#summary = df.groupby('Entrance X [mm]').agg({column_names[15]: ['mean', 'std'],column_names[16]: ['mean', 'std'], column_names[17]: ['mean', 'std'], column_names[18]: ['mean', 'std']}, skipna=True)

column_indices = [15,16,17,18]
summary =  df.iloc[:, column_indices].mean(skipna=True)
summary = summary.to_frame().transpose()

#summary = summary.iloc[:, ::2]  #drop odd index column of the std

#summary = summary.iloc[1:]
#summary.index = [0]

OnAxis_df = df[df['Field angle'] ==  'OnAxis'] #getting CRA

summary['CRA X ["]']  = OnAxis_df['CRA X ["]'][0]
summary['CRA Y ["]']  = -OnAxis_df['CRA Y ["]'][0]
summary.insert(0, 'Time', creation_time)

# insert_df = {
#     'CRA X ["]': ['',OnAxis_df['CRA X ["]'][0]],
#     'CRA Y ["]': ['',OnAxis_df['CRA X ["]'][0]]
#     }

#insert_df = pd.DataFrame(insert_df)

#summary = pd.concat([summary, insert_df], axis=1)

if delimited:
    df_pre = pd.read_csv(path, delimiter=';', nrows=10)
else:
    df_pre = pd.read_csv(path, nrows=10)

#sample_ID = df_pre.loc[df_pre['Measurement ID'] == 'Barcode', 'capstone_primary'].iloc[0]

sample_ID = df_pre[df_pre.columns[1]][df_pre[df_pre.columns[0]] == 'Barcode'].iloc[0]  #use column index instead column name

summary.insert(0, 'Sample ID',sample_ID)

# current_index = summary.index.tolist()
# current_index = ['sampleID',sample_ID] + current_index
#summary.index = current_index

#save reformatted table as xls file
xlspath = directory + "/" + filename + ".xlsx"
summary_path = directory + "/summary_secondary" + ".csv"

df.to_excel(xlspath ,index=False)
summary.to_csv(summary_path ,index= False)


#Write sampleID into the summary csv file
# rows = []
# with open(summary_path, 'r', newline='') as csvfile:
#     csv_reader = csv.reader(csvfile)
#     for row in csv_reader:
#         rows.append(row)

# # Modify the value in cell A2
# rows[0][0] = 'Sample_ID'
# rows[1][0] = sample_ID  # Assuming A2 is the second row (0-based indexing) and first column

# # Write the modified content back to the CSV file
# with open(summary_path, 'w', newline='') as csvfile:
#     csv_writer = csv.writer(csvfile)
#     csv_writer.writerows(rows)

#save to tabulated excel:
if not os.path.exists(outputfile):
    summary.to_excel(outputfile, index=False)
else:
    df_tosave = pd.read_excel(outputfile)
    df_toadd = pd.read_csv(summary_path)
    df_tosave = pd.concat([df_tosave, df_toadd], ignore_index=True)
    df_tosave.to_excel(outputfile, index=False)