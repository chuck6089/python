# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 14:32:05 2024

@author: xzd6089
"""


import subprocess
import os
import matplotlib_inline

# Replace with the actual path to your executable file
executable_file = r'C:\Users\xzd6089\Documents\AutoDP\IQT\PyAutoDP_Capstone_03052024\D54454939 - IQT2-Disp left-and-right WGs, pre-generated distortion maps, etc\pyAutoDP.exe'

exepath, filename = os.path.split(executable_file)

#directory = r'X:\IQT2-Display\Devices\Lumus\Lumus_SN75010\SCS3\2mm Aperture\Lumus_SN75010_SCS3_2mm_Lumus v4_PupilPosition1_X0Y0_-5deg_20240125'
dutdir = r"X:\IQT2-Display\Devices\Lumus\Lumus_SN75010\AutoDP_processing\SCS8"
slbdir = r"X:\IQT2-Display\Devices\Lumus\Lumus_SN75010\Projector\SLB"

dutdir = r"X:\IQT2-Display\Devices\Lumus\Lumus_SN75010\AutoDP_processing\SCS8"

os.chdir(exepath)

# Example command to run the executable with arguments
command = [executable_file, '-c -p', '"Capstone PreP1 right"','-d "', dutdir,'"', '-b "', slbdir, '"']

gettodir = 'cd ' + exepath

commandout = filename + ' -c -p ' + '"Capstone PreP1 right"' ' -d "' + dutdir + '"' + ' -b "'+ slbdir + '"'  

print(gettodir)
print(commandout)

#Run the command
process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

# # Check the result
# if process.returncode == 0:
#     print("Command executed successfully.")
#     print("Output:", process.stdout)
# else:
#     print("Error executing command.")
    
 