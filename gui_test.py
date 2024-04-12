from tkinter import Tk

from met_tool_class import met_tool_class
from pyautodp_app_window import pyautodp_app_window

ini_file = ".\\Data_processing.ini"
tool_params = {"ini_file": ini_file}
tool_obj = met_tool_class(**tool_params)

warnings = [tool_obj.check_prj_select()]

win_params = {
    "tool_obj": tool_obj,
    "warnings": [w[1] for w in warnings if w[0]],
}

root = Tk()
window = pyautodp_app_window(root, **win_params)
root.mainloop()
