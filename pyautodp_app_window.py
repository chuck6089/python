# -*- coding: utf-8 -*-

__authors__ = "Colin Chow (colinmec), Alden"
__version__ = "gui-0.00"

import tkinter as tk
import tkinter.font
import tkinter.ttk

from os import getcwd
from os.path import sep as filesep, split as path_split

import tkfilebrowser
from file_utils import is_subdir
from PIL import Image, ImageTk
from ttkthemes import ThemedStyle


class pyautodp_app_window:

    # Initialize the app window here
    def __init__(self, master, **params):

        ####################################################################################################
        # Window layout and theme
        self.master = master
        self.master.title("Py-AutoDP")

        # No drag-scaling allowed; vector = (width, height)
        self.master.geometry("1200x800+100+100")
        self.master.minsize(1200, 800)
        self.master.maxsize(1200, 800)

        # Create a left-right divider line in the window
        line_area = tk.Canvas(self.master)
        line_area.create_line(600, 0, 600, 800, width=1, fill="gray")
        line_area.pack(fill=tk.BOTH, expand=True)

        # Creating a Font object of "TkDefaultFont" and override default
        self._defaultFont = tk.font.nametofont("TkDefaultFont")
        self._defaultFont.configure(size=10)

        # Configure overall theme appearance
        self._theme = ThemedStyle()
        self._theme.theme_use("plastik")
        if "advanced" not in self._theme.themes:
            self._theme.set_theme_advanced(
                "plastik",
                brightness=1.25,
                saturation=1.0,
                hue=1.0,
                preserve_transparency=False,
                output_dir=None,
                advanced_name="advanced",
            )

        # Configure start button appearance
        self._button_style = tk.ttk.Style()

        ####################################################################################################
        # Tool_obj variables
        self.tool_obj = params["tool_obj"]
        self.tool_obj.prnt_s = self.append_status_text
        self.tool_obj.prnt_w = self.append_warning_text
        self.tool_obj.show_plot = (
            self.show_plot
            if hasattr(self.tool_obj, "save_plots") and self.tool_obj.save_plots
            else None
        )

        ####################################################################################################
        # Variables for window initialization
        self._prj_options = list(self.tool_obj.prj_options.keys())
        self._cur_sel_dir = self.tool_obj.dat_src_dir
        self._auto_slb = tk.BooleanVar(self.master, value=self.tool_obj.auto_slb)
        self._save_interm_data = tk.BooleanVar(
            self.master, value=self.tool_obj.save_interm_data
        )
        self._stage_sql_data = tk.BooleanVar(
            self.master, value=self.tool_obj.stage_sql_data
        )
        self._stage_interm_data = tk.BooleanVar(
            self.master, value=self.tool_obj.stage_interm_data
        )
        self._comment = tk.StringVar(self.master, value=self.tool_obj.comment)
        self._prj_select = tk.StringVar(self.master, value=self._prj_options[0])

        # ID word keep check-boxes
        self._id_checks = [tk.BooleanVar(self.master) for i in range(10)]

        self.tool_obj.id_word_keep = [
            a for a in self.tool_obj.id_word_keep if a <= len(self._id_checks)
        ]

        for i in range(len(self._id_checks)):
            if i + 1 in self.tool_obj.id_word_keep:
                self._id_checks[i].set(True)
            self._id_checks[i].trace("w", self.update_tool_obj_var)

        # Initialize default project selection
        if self.tool_obj.prj_select != "":
            self._prj_select.set(self.tool_obj.prj_select)
        else:
            self.update_tool_obj_var()

        # Setup core classes thru met_tool_class:
        self.tool_obj.setup_tool()

        # Counter for start button state machine
        self._start_button_state = tk.IntVar(self.master, value=2)
        self._load_database_state = tk.IntVar(self.master, value=1)

        if not self.tool_obj.disable_load_db:
            self._load_database_state.set(
                self._load_database_state.get() + self.tool_obj.get_load_db_module()
            )

        ####################################################################################################
        # Initialize input/output fields and buttons
        self._create_labels()
        self._create_project_dropdown()
        self._create_autoqueue_button()
        self._create_data_select_button()
        self._create_data_select_text()
        self._create_sample_word_checkboxes()
        self._create_new_slb_text()
        self._create_meas_list()
        self._create_warning_text()
        self._create_load_database_button()
        self._create_manual_slb_button()
        self._create_slb_chosen_text()
        self._create_auto_slb_checkbox()
        self._create_save_interm_data_checkbox()
        self._create_stage_sql_data_checkbox()
        self._create_stage_interm_data_checkbox()
        self._create_start_button()
        self._create_comment_field()
        self._create_status_text()
        self._create_plot_canvas()

        ####################################################################################################
        # If there is any passed-down warning
        if "warnings" in params and len(params["warnings"]) > 0:
            for w in params["warnings"]:
                self.append_warning_text(w)

        # Making sure everything is ready to go
        self.master.update_idletasks()
        self.autoqueue_callback(keep_warnings=True)

        # Tracer for self._prj_select. Placed here due to dependency on warning text area.
        self._prj_select.trace("w", self.update_tool_obj_project)

        # End of __init__

    ########################################################################################################
    # Create static labels

    def _create_labels(self):
        label_prj = tk.Label(self.master, text="Project / Measurement purpose")
        label_prj.place(relx=0.025, rely=0.01, anchor="nw")

        label_data_select = tk.Label(self.master, text="Folder(s) selected:")
        label_data_select.place(relx=0.22, rely=0.09, anchor="nw")

        label_meas_list = tk.Label(self.master, text="Sample measurements found:")
        label_meas_list.place(relx=0.025, rely=0.41, anchor="nw")

        label_id_select = tk.Label(self.master, text="Sample ID word select:")
        label_id_select.place(relx=0.025, rely=0.245, anchor="nw")

        label_new_slb = tk.Label(
            self.master, text="New system baseline calibration (SLB) found:"
        )
        label_new_slb.place(relx=0.025, rely=0.29, anchor="nw")
        if self.tool_obj.no_slb:
            label_new_slb.configure(state="disabled")

        tk.Label(self.master, text="Warnings:").place(
            relx=0.025, rely=0.81, anchor="nw"
        )
        tk.Label(self.master, text="Comment (Optional):").place(
            relx=0.52, rely=0.17, anchor="nw"
        )
        tk.Label(self.master, text="Status:").place(relx=0.52, rely=0.21, anchor="nw")

        for i in range(len(self._id_checks)):
            tk.Label(self.master, text=str(i + 1)).place(
                relx=0.150 + 0.025 * i, rely=0.227, anchor="nw"
            )

    ########################################################################################################
    # Create input/output fields and controls

    def _create_project_dropdown(self):
        self._prj_dropdown = tk.ttk.OptionMenu(
            self.master, self._prj_select, self._prj_select.get(), *self._prj_options
        )
        self._prj_dropdown["menu"].config(bg="white")
        self._prj_dropdown.place(
            relx=0.025, rely=0.04, width=200, height=30, anchor="nw"
        )

    def _create_autoqueue_button(self):
        self._autoq_button = tk.ttk.Button(
            self.master, text="AUTO-QUEUE", command=self.autoqueue_callback
        )
        self._autoq_button.place(
            relx=0.025, rely=0.12, width=200, height=75, anchor="nw"
        )

    def _create_data_select_button(self):
        data_button = tk.ttk.Button(
            self.master, text="Select input folder(s)", command=self.select_data
        )
        data_button.place(relx=0.22, rely=0.04, width=200, height=30, anchor="nw")

    def _create_data_select_text(self):
        self.data_select_text = tk.Text(
            self.master,
            state="disabled",
            relief=tk.GROOVE,
            borderwidth=0,
            bg="#FAFAFA",
            highlightthickness=1,
            highlightbackground="#B3B3B3",
        )
        self.data_select_text.place(
            relx=0.22, rely=0.12, width=314, height=75, anchor="nw"
        )

    def _create_sample_word_checkboxes(self):
        for i, box in enumerate(self._id_checks):
            tk.Checkbutton(self.master, variable=box).place(
                relx=0.148 + 0.025 * i, rely=0.25, anchor="nw"
            )

    def _create_new_slb_text(self):
        self.new_slb_text = tk.Text(
            self.master,
            state="disabled",
            relief=tk.GROOVE,
            borderwidth=0,
            highlightthickness=1,
            highlightbackground="#B3B3B3",
        )
        self.new_slb_text.place(
            relx=0.025, rely=0.32, width=548, height=60, anchor="nw"
        )

    def _create_meas_list(self):
        columns = {
            "dut_id": [" DUT ID", 160],
            "eyeside": [" Eye Side", 60],
            "project": [" Project", 100],
            "folder": [" Folder", 210],
            "": ["", 0],
        }

        self._list_frame = tk.Frame(self.master)
        self._list_frame.place(relx=0.025, rely=0.44, anchor="nw")
        self.meas_list = tk.ttk.Treeview(
            self._list_frame, height=12, show="headings", selectmode="extended"
        )
        self.meas_list["columns"] = tuple(columns.keys())

        for a in columns.keys():
            self.meas_list.column(a, width=columns[a][1], stretch=False)
            self.meas_list.heading(a, text=columns[a][0], anchor=tk.W)

        scroll_y = tk.ttk.Scrollbar(self._list_frame, command=self.meas_list.yview)
        scroll_y.pack(side=tk.RIGHT, fill=tk.Y)
        scroll_x = tk.ttk.Scrollbar(
            self._list_frame, orient="horizontal", command=self.meas_list.xview
        )
        scroll_x.pack(side=tk.BOTTOM, fill=tk.X)

        self.meas_list.configure(
            xscrollcommand=scroll_x.set, yscrollcommand=scroll_y.set
        )
        self.meas_list.pack(fill=tk.BOTH, expand=True)

        # To activate horizontal scroll bar, f**K tkinter
        self.meas_list.column(list(columns.keys())[-1], minwidth=1, stretch=True)

    def _create_warning_text(self):
        self._warning_text = tk.Text(
            self.master,
            state="disabled",
            relief=tk.GROOVE,
            borderwidth=0,
            bg="#FAFAFA",
            highlightthickness=1,
            highlightbackground="#B3B3B3",
        )
        self._warning_text.place(
            relx=0.025, rely=0.84, width=440, height=100, anchor="nw"
        )

    def _create_load_database_button(self):
        self._load_db_button = tk.ttk.Button(
            self.master,
            command=self.move_load_database_state,
        )
        self._load_db_button.place(
            relx=0.405, rely=0.84, width=90, height=60, anchor="nw"
        )

        self.move_load_database_state(steps=0)

    def _create_manual_slb_button(self):
        state = "disabled" if self._auto_slb.get() else "normal"
        self._manual_slb_button = tk.ttk.Button(self.master, text="Manual select SLB")
        self._manual_slb_button.configure(command=self.manual_select_slb, state=state)
        self._manual_slb_button.place(
            relx=0.52, rely=0.04, width=135, height=30, anchor="nw"
        )
        if self.tool_obj.no_slb:
            self._manual_slb_button.configure(state="disabled")

    def _create_slb_chosen_text(self):
        self.slb_chosen_text = tk.Text(
            self.master,
            state="disabled",
            relief=tk.GROOVE,
            borderwidth=0,
            bg="#FAFAFA",
            highlightthickness=1,
            highlightbackground="#B3B3B3",
        )
        self.slb_chosen_text.place(
            relx=0.52, rely=0.085, width=260, height=50, anchor="nw"
        )

    def _create_auto_slb_checkbox(self):
        self._auto_slb_checkbox = tk.Checkbutton(
            self.master, text="Auto-select SLB", variable=self._auto_slb
        )
        self._auto_slb_checkbox.place(relx=0.64, rely=0.04, anchor="nw")
        self._auto_slb.trace("w", self.switch_manual_slb_state)
        if self.tool_obj.no_slb:
            self._auto_slb_checkbox.configure(state="disabled")

    def _create_save_interm_data_checkbox(self):
        self._save_interm_checkbox = tk.Checkbutton(
            self.master, text="Save intermediate data"
        )
        self._save_interm_checkbox.configure(variable=self._save_interm_data)
        self._save_interm_checkbox.place(relx=0.75, rely=0.04, anchor="nw")
        self._save_interm_data.trace("w", self.update_tool_obj_var)

    def _create_stage_sql_data_checkbox(self):
        self._stage_sql_data_checkbox = tk.Checkbutton(
            self.master, text="Stage SQL data"
        )
        self._stage_sql_data_checkbox.configure(variable=self._stage_sql_data)
        self._stage_sql_data_checkbox.place(relx=0.75, rely=0.08, anchor="nw")
        self._stage_sql_data.trace("w", self.update_tool_obj_var)

    def _create_stage_interm_data_checkbox(self):
        self._stage_interm_data_checkbox = tk.Checkbutton(
            self.master, text="Stage processed data"
        )
        self._stage_interm_data_checkbox.configure(variable=self._stage_interm_data)
        self._stage_interm_data_checkbox.place(relx=0.75, rely=0.12, anchor="nw")
        self._stage_interm_data.trace("w", self.update_tool_obj_var)

    def _create_start_button(self):
        self._start_button = tk.ttk.Button(self.master, style="_button_style.TButton")
        self._start_button.place(
            relx=0.905, rely=0.042, width=85, height=85, anchor="nw"
        )
        self._start_button.configure(command=self.move_start_button_state)
        self.move_start_button_state(steps=0)

    def _create_comment_field(self):
        self._comment_field = tk.Entry(
            self.master,
            relief=tk.GROOVE,
            borderwidth=0,
            textvariable=self._comment,
            highlightthickness=1,
            highlightbackground="#B3B3B3",
        )
        self._comment_field.place(
            relx=0.63, rely=0.17, width=415, height=25, anchor="nw"
        )
        self._comment.trace("w", self.update_tool_obj_var)

    def _create_status_text(self):
        self._status_text = tk.Text(
            self.master,
            state="disabled",
            relief=tk.GROOVE,
            borderwidth=0,
            bg="#FAFAFA",
            highlightthickness=1,
            highlightbackground="#B3B3B3",
        )
        self._status_text.place(
            relx=0.52, rely=0.24, width=547, height=170, anchor="nw"
        )

    def _create_plot_canvas(self):
        self._plot_frame = tk.Frame(self.master, width=564, height=376, bg="white")
        self._plot_frame.place(relx=0.515, rely=0.48, anchor="nw")
        self._plot_img = tk.Label(self._plot_frame, image="")
        # Don't pack label, just show frame window

    ########################################################################################################
    # Clear input/output text fields

    def clear_data_select(self):
        self.tool_obj.data_folders.clear()
        self.tool_obj.data_queue = {}
        self.tool_obj.ref_queue = {}
        self.data_select_text.configure(state="normal")
        self.data_select_text.delete("1.0", "end")
        self.data_select_text.configure(state="disabled")

    def clear_new_slb(self):
        self.new_slb_text.configure(state="normal")
        self.new_slb_text.delete("1.0", "end")
        self.new_slb_text.configure(state="disabled")

    def clear_meas_list(self):
        for item in self.meas_list.get_children():
            self.meas_list.delete(item)

    def clear_warning_text(self):
        self._warning_text.configure(state="normal")
        self._warning_text.delete("1.0", "end")
        self._warning_text.configure(state="disabled")

    def clear_chosen_slb(self):
        self.tool_obj.slb_chosen.clear()
        self.slb_chosen_text.configure(state="normal")
        self.slb_chosen_text.delete("1.0", "end")
        self.slb_chosen_text.configure(state="disabled")

    def clear_status_text(self):
        self._status_text.configure(state="normal")
        self._status_text.delete("1.0", "end")
        self._status_text.configure(state="disabled")

    def clear_plot(self):
        self._plot_img.destroy()
        self._plot_img = tk.Label(self._plot_frame, image="")
        # Don't pack label, just show frame window

    ########################################################################################################
    # Append to input/output text fields

    def append_warning_text(self, warning):
        print(warning)
        self._warning_text.configure(state="normal")
        self._warning_text.insert("end", warning + "\n")
        self._warning_text.configure(state="disabled")
        self.master.update_idletasks()

    def append_status_text(self, message):
        print(message)
        self._status_text.configure(state="normal")
        self._status_text.insert("end", message + "\n")
        self._status_text.configure(state="disabled")
        self.master.update_idletasks()

    ########################################################################################################
    # Display info or plots on app window

    def show_data_select(self):
        self.data_select_text.configure(state="normal")
        for f in self.tool_obj.data_folders:
            self.data_select_text.insert("end", f + "\n")
        self.data_select_text.configure(state="disabled")

    def show_new_slb(self, slb_info_tuples):
        self.clear_new_slb()
        self.new_slb_text.configure(state="normal")
        for t in slb_info_tuples:
            self.new_slb_text.insert("end", t[0] + "\n")
        self.new_slb_text.configure(state="disabled")

    def show_meas_list(self, meas_info_tuples):
        self.clear_meas_list()
        for i, m in enumerate(meas_info_tuples):
            self.meas_list.insert(parent="", index="end", iid=i, text="", values=m)

    def show_chosen_slb(self, slb_info_tuples):
        self.slb_chosen_text.configure(state="normal")
        for t in slb_info_tuples:
            self.slb_chosen_text.insert("end", t[0] + "\n")
        self.slb_chosen_text.configure(state="disabled")

    def show_plot(self, img):
        if type(img) is str:
            img = Image.open(img)

        new_size = (
            int(0.99 * self._plot_frame.winfo_width()),
            int(0.99 * self._plot_frame.winfo_height()),
        )
        img = img.resize(new_size)
        img = ImageTk.PhotoImage(img)
        self._plot_img.config(image=img)
        self._plot_img.image = img
        self._plot_img.pack()
        self.master.update_idletasks()

    ########################################################################################################
    # Other call-backs

    def select_data(self):
        self.clear_data_select()
        self.clear_new_slb()
        self.clear_warning_text()
        self._list_frame.destroy()
        self._create_meas_list()
        if self._auto_slb.get():
            self.clear_chosen_slb()

        self.tool_obj.data_folders = list(
            tkfilebrowser.askopendirnames(initialdir=self._cur_sel_dir)
        )

        self.tool_obj.setup_tool()

        if len(self.tool_obj.data_folders) > 0:
            self.show_data_select()
            self._cur_sel_dir = path_split(self.tool_obj.data_folders[0])[0]
            info_disp = self.tool_obj.check_data_selection_dir()

        if self.tool_obj.ref_queue:
            self.show_new_slb(info_disp)
        elif self.tool_obj.data_queue:
            self.show_meas_list(info_disp)
            self.tool_obj.slb_chosen, disp_info = self.tool_obj.queue.get_slb_match(
                self.tool_obj.data_queue, self.tool_obj.slb_chosen
            )
            self.show_chosen_slb(disp_info)

    def autoqueue_callback(self, keep_warnings=False):
        if not keep_warnings:
            self.clear_warning_text()

        self.tool_obj.setup_tool()

        self.tool_obj.data_queue, info_disp = self.tool_obj.queue.get_data_queue(
            is_ref=False, path=None, no_msg=False
        )
        self.show_meas_list(info_disp)

        if self._auto_slb.get():
            self.clear_chosen_slb()

        self.tool_obj.slb_chosen, disp_info = self.tool_obj.queue.get_slb_match(
            self.tool_obj.data_queue, self.tool_obj.slb_chosen
        )
        self.show_chosen_slb(disp_info)

        if self.tool_obj.no_slb:
            return

        self.tool_obj.ref_queue, info_disp = self.tool_obj.queue.get_data_queue(
            is_ref=True, path=None, no_msg=False
        )
        self.show_new_slb(info_disp)

    def manual_select_slb(self):
        self.clear_chosen_slb()
        _folder = tkfilebrowser.askopendirname(initialdir=self.tool_obj.ref_src_dir)
        if _folder != "" and is_subdir(_folder, self.tool_obj.ref_src_dir):
            _folder = _folder.replace(self.tool_obj.ref_src_dir, "").strip(filesep)
            _slb, disp_info = self.tool_obj.queue.get_data_queue(
                is_ref=True, path=[_folder], no_msg=False
            )
            if len(disp_info) < 1:
                self.append_warning_text(
                    "Warning: No valid SLB measurement found in selection."
                )
            elif len(disp_info) > 1:
                self.append_warning_text(
                    "Warning: Multiple SLB measurements found. Only one allowed."
                )
            else:
                slb_meas_info = _slb[list(_slb.keys())[0]][0]
                self.tool_obj.slb_chosen[slb_meas_info.dut_id] = slb_meas_info
                _, _ = self.tool_obj.queue.get_slb_match(
                    self.tool_obj.data_queue, self.tool_obj.slb_chosen
                )
                self.show_chosen_slb(disp_info)
        else:
            self.append_warning_text("Warning: Invalid SLB folder selection.")

    def switch_manual_slb_state(self, *args):
        if self._auto_slb.get():
            self._manual_slb_button.configure(state="disabled")
            self.clear_chosen_slb()
            self.tool_obj.slb_chosen, disp_info = self.tool_obj.queue.get_slb_match(
                self.tool_obj.data_queue, {}
            )
            self.show_chosen_slb(disp_info)
        else:
            self._manual_slb_button.configure(state="normal")

    def start_state_zero_action(self):
        self._theme.configure("_button_style.TButton", foreground="red")
        self._start_button.configure(text="STOP", state="normal")
        self.move_load_database_state(steps=1 - self._load_database_state.get())
        self.master.update_idletasks()
        self.tool_obj.process.process_all(
            self.tool_obj.ref_queue,
            self.tool_obj.data_queue,
            self.tool_obj.slb_chosen,
        )
        self.tool_obj.stage.stage_analysis_output(
            self.tool_obj.ref_queue,
            self.tool_obj.data_queue,
        )

        if not self.tool_obj.disable_load_db:
            if self.tool_obj.pause_load_db:
                self.move_load_database_state()
            else:
                self.tool_obj.load_db(
                    self.tool_obj.func.prj.data_store_mapping, self.tool_obj.func
                )
        self.autoqueue_callback()
        self.move_start_button_state(steps=2)

    def start_state_one_action(self):
        self._theme.configure("_button_style.TButton", font=("Arial", 12, "bold"))
        self._start_button.configure(text="Stopping", state="disabled")

    def start_state_two_action(self):
        self._theme.configure(
            "_button_style.TButton",
            foreground="#00CD00",
            font=("Arial", 15, "bold"),
        )
        self._start_button.configure(text="START", state="normal")

    def move_start_button_state(self, *args, **kwargs):
        i = 1 if "steps" not in kwargs else kwargs["steps"]
        self._start_button_state.set((self._start_button_state.get() + i) % 3)

        actions = {
            0: self.start_state_zero_action,
            1: self.start_state_one_action,
            2: self.start_state_two_action,
        }

        actions[self._start_button_state.get()]()

    def move_load_database_state(self, *args, **kwargs):
        i = 1 if "steps" not in kwargs else kwargs["steps"]
        self._load_database_state.set((self._load_database_state.get() + i) % 3)

        if self._load_database_state.get() == 0:
            self._load_db_button.configure(
                text="  Loading\nDatabase...", state="disabled"
            )
            self.tool_obj.load_db(
                self.tool_obj.func.prj.data_store_mapping, self.tool_obj.func
            )
            self.move_load_database_state(steps=2)
        elif self._load_database_state.get() == 1:
            self._load_db_button.configure(text="   Load\nDatabase", state="disabled")
        else:
            self._load_db_button.configure(text="   Load\nDatabase", state="enabled")

    ########################################################################################################
    # Live update output parameters

    def update_tool_obj_var(self, *args):
        self.tool_obj.save_interm_data = self._save_interm_data.get()
        self.tool_obj.stage_sql_data = self._stage_sql_data.get()
        self.tool_obj.stage_interm_data = self._stage_interm_data.get()
        self.tool_obj.comment = self._comment.get()
        self.tool_obj.prj_select = self._prj_select.get()
        self.tool_obj.prj_config = self.tool_obj.prj_options[self.tool_obj.prj_select]
        self.tool_obj.id_word_keep = [
            i + 1 for i, check in enumerate(self._id_checks) if check.get()
        ]

    def update_tool_obj_project(self, *args):
        self.tool_obj.prj_select = self._prj_select.get()
        self.tool_obj.prj_config = self.tool_obj.prj_options[self.tool_obj.prj_select]
        self.tool_obj.setup_tool()
        self.autoqueue_callback(keep_warnings=False)

    # End of class pyautodp_app_window
