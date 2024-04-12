# -*- coding: utf-8 -*-

__authors__ = "Colin Chow (colinmec), Alden"
__version__ = ""

from collections import OrderedDict
from os.path import (
    exists,
    isdir,
    join as path_join,
    sep as filesep,
    split as path_split,
)

from file_utils import listdir_dict, read_csv_as_dict_arrays


class get_queue_class:
    # This class defines:
    # 1. How new, un-processed datesets are identified
    # 2. How measurement information such as DUT ID, date, etc., are obtained
    # 3. How matching baseline reference is determined for each measurement
    # 4. Data structure for the queue

    def __init__(self, shared_attr):
        self.setup = shared_attr
        self.is_ref = False
        self.path_prfx = self.setup.set_data_path(self.is_ref)
        self._all_slbs = {}
        # End of __init__

    def _list_all_slb(self):
        # list all meas_info_class object associated with SLB

        self._all_slbs = {}
        slbs, info_disp = self.get_data_queue(is_ref=True, path=[""], no_msg=True)
        for m in slbs[""]:
            self._all_slbs[m.meas_date] = m

        # End of _list_all_slb

    def get_data_queue(self, is_ref=False, path=None, no_msg=False):
        # Inputs: (i) is_ref - True if for SLB
        #         (ii) path - Path to search. None for auto-queue
        #         (iii) no_msg - if True, suppress invalid measurement folder warning
        # First search for new measurements in sub-folders of data or SLB directory.
        # For each sub-folder containing new measurements, obtain the measurement queue and
        # return all such queues as a dict.
        # Note: A new measurement is created or modified later than corresponding record.

        self.is_ref, self.path_prfx = is_ref, self.setup.set_data_path(is_ref)

        if not exists(self.path_prfx) or not isdir(self.path_prfx):
            self.setup.prnt_s(f"Warning: {self.path_prfx} does not exist")
            return {}, []

        dir_q, info_disp = OrderedDict(), []
        dat_prcd = self.get_processed_list() if not path else {}
        dat_dirs = (
            listdir_dict(path=self.path_prfx, dir_only=True)
            if not path
            else {"name": [p.replace(self.path_prfx, "").strip(filesep) for p in path]}
        )

        for d in dat_dirs["name"]:
            valid, meas_list = self.setup.list_meas(
                path_join(self.path_prfx, d), dat_prcd, self.is_ref, no_msg
            )
            if meas_list:
                for m in meas_list:
                    m.tool_id, m.project, m.comment = (
                        self.setup.tool_id,
                        self.setup.prj_select,
                        self.setup.comment,
                    )
                    self.get_meas_info(m)
                    info_disp.append(get_disp_tuple(m, self.path_prfx))
                dir_q[d] = meas_list

        return dir_q, info_disp
        # End of get_data_queue

    def get_processed_list(self):
        # For auto-queue, read data/SLB processed tracking list
        # Returns dict[measurement folder] = timestamp

        processed = {}
        proc_list = (
            self.setup.ref_proc_list if self.is_ref else self.setup.dat_proc_list
        )
        col_key = "SLB_name" if self.is_ref else "Folder_name"
        dat_prcd = read_csv_as_dict_arrays(proc_list)
        dat_prcd[col_key] = [path_join(self.path_prfx, f) for f in dat_prcd[col_key]]
        dat_prcd["Timestamp"] = [float(t) for t in dat_prcd["Timestamp"]]

        for i, f in enumerate(dat_prcd[col_key]):
            if f not in processed or processed[f] < dat_prcd["Timestamp"][i]:
                processed[f] = dat_prcd["Timestamp"][i]

        return processed
        # End of get_processed_list

    def create_dut_id(self, meas_info):
        # Create DUT id from underscore separated <name> by keeping index-th word

        folder = path_split(meas_info.folder)[1]
        index = self.setup.id_word_keep

        if self.is_ref:
            index = [i + 1 for i in range(len(folder.split("_")))]

        str_arr = folder.split("_")
        new_name = ".".join([str_arr[i - 1] for i in index if i <= len(str_arr)])
        meas_info.dut_id = new_name.replace("-", ".")

        # End of create_dut_id

    def get_slb_match(self, data_queue, slb_chosen):
        # If SLB is manually selected (slb_chosen not empty), use it.
        # Otherwise, find SLB with closest meas_date prior to each measurement.
        # If such SLB does not exist (i.e., all SLBs are measured after measurement),
        # then choose one that is closest in date
        # Return all chosen SLBs as a dict

        if self.setup.no_slb:
            return

        self._list_all_slb()
        _slb_chosen, disp_info = {}, []
        for d in data_queue:
            for m in data_queue[d]:
                if slb_chosen:
                    m.callibration = list(slb_chosen.keys())[0]
                    continue

                order = [
                    (abs(m.meas_date - a), a)
                    for a in self._all_slbs
                    if m.meas_date - a >= 0
                ]
                if not order:
                    order = [(abs(m.meas_date - a), a) for a in self._all_slbs]
                order.sort(key=lambda a: a[0])
                chosen_key = self._all_slbs[order[0][1]].dut_id
                m.calibration = chosen_key

                if chosen_key not in _slb_chosen:
                    _slb_chosen[chosen_key] = self._all_slbs[order[0][1]]
                    disp_info.append(
                        get_disp_tuple(_slb_chosen[chosen_key], self.path_prfx)
                    )

        return _slb_chosen if not slb_chosen else slb_chosen, disp_info
        # End of get_slb_match


############################################################################################################
# Static function(s)


def get_eyeside_from_seq_name(meas_info):
    # If sequence name contains any keyword for eye-side, register it in meas_info

    seq_name = path_split(meas_info.seq_file)[-1]
    seq_name = seq_name.replace(" ", "_").replace("-", "_").replace(".", "_")
    name_parts = seq_name.split("_")

    for a in name_parts:
        if a.lower() in ["left", "lefteye", "l", "lefteye", "lef"]:
            meas_info.eye_side = "left"
        elif a.lower() in ["right", "righteye", "r", "rigth", "rigtheye", "righ"]:
            meas_info.eye_side = "right"

    # End of get_eyeside_from_seq_name


def get_disp_tuple(meas_info, path_prfx):
    # Simply returns DUT info to be displayed on interface

    return (
        meas_info.dut_id,
        meas_info.eye_side,
        meas_info.project,
        meas_info.folder.replace(path_prfx + filesep, ""),
    )
    # End of get_disp_tuple
