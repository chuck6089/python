# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 11:51:47 2024

@author: xzd6089
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

def flatten_df(df):
    """
    Flatten the DataFrame to one row by concatenating all rows into the first row and 
    append the index value to column names for each row.

    Parameters:
        df (DataFrame): The DataFrame to be flattened.

    Returns:
        DataFrame: Flattened DataFrame.
    """
    # Create separate DataFrames for each row
    row_dfs = [df.iloc[[i]].rename(columns=lambda x: str(df.index[i]) + '_' + str(x)) for i in range(len(df))]
    
    for dff in row_dfs:
        dff.index = [0]

    # Concatenate the DataFrames along columns (axis=1)
    flattened_df = pd.concat(row_dfs, axis=1)

    return flattened_df


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

creation_time = get_file_creation_time(path)

ZoneA = ['OnAxis', 'TP1', 'TP2', 'TP3', 'TP4','TP5']
ZoneB = ['TP6','TP7','TP8','TP9']
Region0 = ['Pos1','Pos2','Pos3','Pos4','Pos5']
Region1 = ['Pos6','Pos7','Pos8','Pos9','Pos10','Pos11','Pos12','Pos13']

#path = r"X:\SYS-STMTF\Capstone PP1\Schott\Measurement_capstone_primary_sys_stmtf_labarflex_ge01_SCS30_20240318_170531\Measurement_capstone_primary_sys_stmtf_labarflex_ge01_SCS30_20240318_170531.csv"

#drop the first 60 rows
outputfile = r"C:\Users\xzd6089\Documents\Data\Capstone\MTF_metrics_tabulated.xlsx"

if delimited:
    df = pd.read_csv(path,skiprows= 61,delimiter=';')  #read dataframe with delimiter
else:
    df = pd.read_csv(path,skiprows= 61)  #read dataframe without delimiter


#Split 'Label column' in to two columns: 'position' and 'field angle'
df['Label'] = df['Label'].str.replace('Pos_','Pos')
df['Label'] = df['Label'].str.replace('On_Axis','OnAxis')
df['Label'] = df['Label'].str.replace('TP_','TP')

df[['Position', 'Field angle']] = df['Label'].str.split('_', expand=True)

df.drop(columns=['Label'], inplace = True)

#move last 2 columns into 4th and 5th columns of dataframe

last_two_columns = df.iloc[:, -2:]

df = pd.concat([df.iloc[:, :2], last_two_columns, df.iloc[:, 2:]], axis=1)

df = df.iloc[:, :-2]


#Re-label the Run Label columns with Red/Green/White

df['Run Label'].replace(1,'Blue',inplace = True)
df['Run Label'].replace(2,'Green',inplace = True)
df['Run Label'].replace(3,'Red',inplace = True)

df.rename(columns={'Run Label': 'Color'}, inplace=True)

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

#Calculate the mean and stdev for each color

summary = df.groupby('Color').agg({column_names[16]: ['mean', 'std'], column_names[17]: ['mean', 'std'], column_names[18]: ['mean', 'std'], column_names[19]: ['mean', 'std']}, skipna=True)

#calculate for CRA
OnAxis_df = df[df['Field angle'] ==  'OnAxis']

CRA_summary = OnAxis_df.groupby('Color').agg({column_names[20]: ['mean', 'std'], column_names[21]: ['mean', 'std']}, skipna = True)              

#calculate for power
power_df = OnAxis_df[OnAxis_df['Position'] == 'Pos1']
power_summary = power_df[['Power_V@ 7.5 lp/deg [dpt]','Power_H@ 7.5 lp/deg [dpt]']].mean(skipna=True)
power_summary = power_summary.to_frame().transpose()

summary = pd.concat([summary, CRA_summary], axis=1)

#split df by ZoneA/B and Region0/1:
df_ZoneA_region0 = df[df['Position'].isin(Region0) & df['Field angle'].isin(ZoneA)]
df_ZoneA_region1 = df[df['Position'].isin(Region1) & df['Field angle'].isin(ZoneA)]
df_ZoneB_region0 = df[df['Position'].isin(Region0) & df['Field angle'].isin(ZoneB)]
df_ZoneB_region1 = df[df['Position'].isin(Region1) & df['Field angle'].isin(ZoneB)]

mtf_summary_ZoneA_region0 = df_ZoneA_region0.groupby('Color').agg({column_names[16]: ['mean', 'std'], column_names[17]: ['mean', 'std'], column_names[18]: ['mean', 'std'], column_names[19]: ['mean', 'std']}, skipna=True)
mtf_summary_ZoneA_region1 = df_ZoneA_region1.groupby('Color').agg({column_names[16]: ['mean', 'std'], column_names[17]: ['mean', 'std'], column_names[18]: ['mean', 'std'], column_names[19]: ['mean', 'std']}, skipna=True)
mtf_summary_ZoneB_region0 = df_ZoneB_region0.groupby('Color').agg({column_names[16]: ['mean', 'std'], column_names[17]: ['mean', 'std'], column_names[18]: ['mean', 'std'], column_names[19]: ['mean', 'std']}, skipna=True)    
mtf_summary_ZoneB_region1 = df_ZoneB_region1.groupby('Color').agg({column_names[16]: ['mean', 'std'], column_names[17]: ['mean', 'std'], column_names[18]: ['mean', 'std'], column_names[19]: ['mean', 'std']}, skipna=True)

mtf_summary_ZoneA_region0 = mtf_summary_ZoneA_region0.add_suffix('_ZoneA_Reg0')
mtf_summary_ZoneA_region1 = mtf_summary_ZoneA_region1.add_suffix('_ZoneA_Reg1')
mtf_summary_ZoneB_region0 = mtf_summary_ZoneB_region0.add_suffix('_ZoneB_Reg0')
mtf_summary_ZoneB_region1 = mtf_summary_ZoneB_region1.add_suffix('_ZoneB_Reg1')

summary = pd.concat([mtf_summary_ZoneA_region0, mtf_summary_ZoneA_region1, mtf_summary_ZoneB_region0, mtf_summary_ZoneB_region1], axis=1)
summary = pd.concat([summary, CRA_summary], axis=1)

#flatten the summary table to include R/G/B
summary_flat = summary
columns_to_remove = summary_flat.columns[1::2]  # Start from the 3rd column and select every odd-indexed column
summary_flat = summary_flat.drop(columns=columns_to_remove)

summary_flat = flatten_df(summary_flat)

summary_flat = summary_flat.reset_index(drop=True)

#summary_flat = pd.concat([summary_flat, power_summary], axis=1)


if delimited:
    df_pre = pd.read_csv(path, delimiter=';', nrows=10)
else:
    df_pre = pd.read_csv(path, nrows=10)

#sample_ID = df_pre.loc[df_pre['Measurement ID'] == 'Barcode', 'capstone_primary'].iloc[0]

sample_ID = df_pre[df_pre.columns[1]][df_pre[df_pre.columns[0]] == 'Barcode'].iloc[0]  #use column index instead column name

# current_index = summary.index.tolist()
# current_index = ['sampleID',sample_ID] + current_index
#summary.index = current_index

#save reformatted table as xls file
xlspath = directory + "/" + filename + ".xlsx"
summary_path = directory + "/summary" + ".csv"

df.to_excel(xlspath ,index=False)
summary.to_csv(summary_path ,index= True)


#Write sampleID into the summary csv file
rows = []
with open(summary_path, 'r', newline='') as csvfile:
    csv_reader = csv.reader(csvfile)
    for row in csv_reader:
        rows.append(row)

# Modify the value in cell A2
rows[0][0] = 'Sample_ID'
rows[1][0] = sample_ID  # Assuming A2 is the second row (0-based indexing) and first column

# Write the modified content back to the CSV file
with open(summary_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerows(rows)
    
#save the flattend summary
summary_flat_path = directory + "/summary_1row" + ".csv"
summary_flat.to_csv(summary_flat_path ,index= True)

summary_flat = pd.read_csv(summary_flat_path)

# Remove the desired row and column

# index_to_remove = 2
summary_flat.drop(0, inplace=True) #drop 1st row
summary_flat = summary_flat.drop(summary_flat.columns[0], axis=1) # drop 1st column
summary_flat.index = [0]
summary_flat = pd.concat([summary_flat, power_summary], axis=1)
summary_flat.insert(0, 'Time',creation_time)
summary_flat.insert(0, 'Sample ID',sample_ID)


# # Write the modified DataFrame back to the CSV file
summary_flat.to_csv(summary_flat_path, index=False)

#save to tabulated excel:
if not os.path.exists(outputfile):
    summary_flat.to_excel(outputfile, index=False)
else:
    df_tosave = pd.read_excel(outputfile)
    df_tosave = pd.concat([df_tosave, summary_flat], ignore_index=True)
    df_tosave.to_excel(outputfile, index=False)