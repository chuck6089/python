# -*- coding: utf-8 -*-

__authors__ = "Colin Chow (colinmec), Alden"
__version__ = ""

from os.path import getmtime, join as path_join, split as path_split

from file_utils import listdir_dict, unixtime_to_datenum, wgmet_parse_ini


class shared_attribute_class:
    def __init__(self, tool_obj):

        self._ignore_prfx = [
            "intermediate data",
            "intermediate_data",
            "omet_reports",
            "omet plots",
            "omet_plots",
            "plots",
            "todb",
            "archived",
            "transferred",
            "old data",
            "old_data",
        ]

        for a in tool_obj.__dict__:
            self.__dict__[a] = tool_obj.__dict__[a]

        for a in [
            "ref_queue",
            "data_queue",
            "slb_chosen",
            "default_prj",
            "prj_options",
            "data_folders",
        ]:
            if hasattr(self, a):
                del self.__dict__[a]

        self.db_dir = f"ToDB_{self.module_code.split('.')[0].upper()}"
        self.plot_dir_prfx = "Plots"
        self.interm_dir_prfx = "Intermediate_Data"
        self.prj = tool_setup_param_class(self.tool_config, self.prj_config)
        self._ignore_prfx.extend(self.prj.ignore_prfx)

    def list_meas(self, root, exclude=None, is_ref=False, no_msg=False):

        datenum = unixtime_to_datenum(getmtime(root))
        end_dir = path_split(root.lower())[1]

        if any(end_dir.startswith(a.lower()) for a in self._ignore_prfx):
            return False, []

        if root in exclude and exclude[root] >= datenum:
            return True, []

        valid, files = self.is_valid_meas_dir(root, is_ref, True)

        if valid:
            return True, [
                meas_info_class(folder=root, datenum=datenum, data_files=files)
            ]
        else:
            dirs, meas_list, root_valid = (
                listdir_dict(path=root, dir_only=True),
                [],
                False,
            )
            for d in dirs["name"]:
                sub_dir_valid, sub_dir_list = self.list_meas(
                    path_join(root, d), exclude, is_ref, no_msg
                )
                root_valid = root_valid or sub_dir_valid
                meas_list.extend(sub_dir_list)

            if not no_msg and not root_valid:
                self.prnt_w(f'Warning: Skipped invalid measurement folder "{root}"')

        return root_valid, meas_list
        # End of list_meas

    def set_data_path(self, is_ref):
        # Based on whether is_ref is True (for SLB) or False, return data path accordingly

        return self.ref_src_dir if is_ref else self.dat_src_dir
        # End of set_data_path


class tool_setup_param_class:
    def __init__(self, tool_config, prj_config):
        self.colors = ["blue", "green", "red"]

        tool_param = wgmet_parse_ini(tool_config) if tool_config != "" else {}
        for a in tool_param:
            self.__dict__[a] = tool_param[a]

        prj_param = wgmet_parse_ini(prj_config) if prj_config != "" else {}
        for a in prj_param:
            self.__dict__[a] = prj_param[a]

        for a in ["colors", "ignore_prfx", "incl_metrics"]:
            if hasattr(self, a) and type(self.__dict__[a]) != list:
                self.__dict__[a] = [self.__dict__[a]]


class meas_info_class:
    def __init__(self, **kwargs):

        attrs = [
            "dut_id",
            "barcode",
            "eye_side",
            "meas_date",
            "project",
            "tool_id",
            "folder",
            "datenum",
            "operator",
            "comment",
            "calibration",
            "seq_file",
            "data_files",
        ]

        for a in attrs:
            if a in kwargs:
                self.__dict__[a] = kwargs[a]
            else:
                self.__dict__[a] = ""
                self.data_files = []
