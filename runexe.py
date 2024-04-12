# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:32:05 2024

@author: xzd6089
"""


import subprocess
import os
import matplotlib_inline

# Replace with the actual path to your executable file
executable_file = r'C:\Users\xzd6089\Documents\Data\Capstone\Coloruniformity-D52948357\uniAutorun.exe'

exepath, filename = os.path.split(executable_file)

#directory = r'X:\IQT2-Display\Devices\Lumus\Lumus_SN75010\SCS3\2mm Aperture\Lumus_SN75010_SCS3_2mm_Lumus v4_PupilPosition1_X0Y0_-5deg_20240125'
directory = "C:\Data\Coloruniformity test"

os.chdir(exepath)

# Example command to run the executable with arguments
command = [executable_file, '-p', 'Capstone','-d "', directory,'"']

gettodir = 'cd ' + exepath

commandout = filename + ' -p Capstone -d "' + directory + '"'   

print(gettodir)
print(commandout)

# Run the command
#process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# # Check the result
# if process.returncode == 0:
#     print("Command executed successfully.")
#     print("Output:", process.stdout)
# else:
#     print("Error executing command.")
    
