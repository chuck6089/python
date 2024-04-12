# -*- coding: utf-8 -*-

__authors__ = "Colin Chow (colinmec), Alden"
__version__ = ""

from collections import OrderedDict
from os.path import isdir, join as path_join, sep as filesep, split as path_split

from file_utils import is_subdir, listdir_dict, wgmet_parse_ini


class met_tool_class:
    def __init__(self, **kwargs):

        self.prnt_s, self.prnt_w = print, print
        self.show_plot, self.no_slb = None, False
        self.ref_src_dir, self.dat_src_dir = "", ""
        self.data_queue, self.ref_queue = {}, {}

        ini_params = wgmet_parse_ini(kwargs["ini_file"])
        for field in ini_params:
            self.__dict__[field] = ini_params[field]

        self.dat_src_dir = self.dat_src_dir.strip(filesep)
        self.ref_src_dir = self.ref_src_dir.strip(filesep)
        if self.ref_src_dir == "" or not isdir(self.ref_src_dir):
            self.auto_slb, self.no_slb = False, True

        if not hasattr(self, "slb_chosen"):
            self.slb_chosen = {}

        for a in ["id_word_keep", "data_folders"]:
            if not hasattr(self, a):
                self.__dict__[a] = []

        for a in ["prj_select", "comment", "pmengine_path"]:
            if not hasattr(self, a):
                self.__dict__[a] = ""

        for a in [
            "auto_slb",
            "save_interm_data",
            "stage_sql_data",
            "stage_interm_data",
        ]:
            if not hasattr(self, a):
                self.__dict__[a] = False

        if type(self.comment) is list:
            self.comment = ", ".join(self.comment)

        self.get_project_options()

    def get_project_options(self):
        if type(self.default_prj) is not list:
            self.default_prj = [self.default_prj]

        self.prj_options = OrderedDict()
        for p in self.default_prj:
            self.prj_options[p] = ""

        [folder, file] = path_split(self.tool_config)
        file = file.replace(".ini", "")
        ini_list = listdir_dict(path=folder, ext="ini")
        for f in ini_list["name"]:
            if f.startswith(file) and f != file + ".ini":
                prj_name = (
                    f.replace(file + "_", "").replace(".ini", "").replace("_", " ")
                )
                self.prj_options[prj_name] = path_join(folder, f)

    def check_prj_select(self):
        warning = (False, None)
        if hasattr(self, "prj_select"):
            if self.prj_select in self.prj_options:
                self.prj_config = self.prj_options[self.prj_select]
            else:
                warning = (
                    True,
                    "Warning: Project '" + self.prj_select + "' not available.",
                )
                self.prj_select, self.prj_config = "", ""
        return warning

    def check_data_selection_dir(self):
        info_disp = []
        if all(is_subdir(p, self.dat_src_dir) for p in self.data_folders):
            self.data_queue, info_disp = self.queue.get_data_queue(
                is_ref=False, path=self.data_folders, no_msg=False
            )
        elif not self.no_slb and all(
            is_subdir(p, self.ref_src_dir) for p in self.data_folders
        ):
            self.ref_queue, info_disp = self.queue.get_data_queue(
                is_ref=True, path=self.data_folders, no_msg=False
            )
        else:
            self.prnt_w(
                "Warning: Selection(s) must be sub-folder(s) of either SLB or data directory."
            )
        return info_disp

    def setup_tool(self):
        code_prfx = "met_tools." + self.module_code + "."
        module_prfx = self.module_code.split(".")[-1]
        func = module_prfx + "_shared_attribute"
        self.func = getattr(__import__(code_prfx + func, fromlist=[func]), func)(self)

        step_names = ["_get_queue", "_process_queue", "_stage_output"]
        self.queue, self.process, self.stage = [
            getattr(
                __import__(
                    code_prfx + module_prfx + s,
                    fromlist=[module_prfx + s],
                ),
                module_prfx + s,
            )(self.func)
            for s in step_names
        ]

    def get_load_db_module(self):
        if "load_db_module" not in self.__dict__:
            self.prnt_s("Load database module not found. This feature is disabled.")
            return 0

        self.load_db = getattr(
            __import__(self.load_db_module, fromlist=[self.load_db_module]),
            self.load_db_module,
        )

        return 1
