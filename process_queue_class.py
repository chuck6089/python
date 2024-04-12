# -*- coding: utf-8 -*-

__authors__ = "Colin Chow (colinmec), Alden"
__version__ = ""

from csv import writer as csv_writer
from datetime import datetime
from os import makedirs
from os.path import isdir, join as path_join, sep as filesep
from shutil import rmtree
from uuid import uuid4

from file_utils import datenum_to_timestring, unixtime_to_datenum


class process_queue_class:
    def __init__(self, shared_attr):

        self.setup = shared_attr
        self.is_ref = False
        self.path_prfx = self.setup.set_data_path(self.is_ref)
        self.slb_saved = {}
        self.wide_table = []

        # End of __init__

    def process_all(self, slb_q, data_q, slb_chosen):
        # First load all SLB measurements matched to DUT measurements
        # Then (load, if haven't), analyze and generate report for new SLBs
        # Finally load, analyze and save reports for DUT measurements

        if not self.setup.no_slb:
            self.is_ref, self.path_prfx = True, self.setup.set_data_path(True)

            # load matched SLBs
            for key in slb_chosen:
                self.slb_saved[key] = self.load_meas_data(slb_chosen[key])

            # Process new SLBs
            for key in slb_q:
                for meas in slb_q[key]:
                    if meas.dut_id not in self.slb_saved:
                        slb = self.load_meas_data(meas)
                    else:
                        slb = self.slb_saved[meas.dut_id]

                    self.setup.prnt_s(
                        f"Still need DEV: Pretend using var. slb: {slb.angle0}"
                    )

        self.is_ref, self.path_prfx = False, self.setup.set_data_path(False)

        # Process DUT measurements
        for key in data_q:
            # For each top measurement folder, save a wide-table format report
            self.wide_table.clear()

            for meas in data_q[key]:
                # Load, analyze and save
                data = self.load_meas_data(meas)
                res = self.get_analysis_results(meas, data)
                self.save_meas_data(meas, res)
                self.append_wide_table(res)
                self.update_processed_tracker(meas)

            self.save_wide_table(key)

        # End of process_all

    def get_analysis_results(self, meas_info, meas_data):
        # Import methods precribed in incl_metrics parameter and process one-by-one,
        # then append to variable 'res'

        res = [("is_bad", "False")]

        # First get header
        for a in meas_info.__dict__:
            if type(meas_info.__dict__[a]) is list:
                continue

            if a in ["meas_date", "datenum"]:
                meas_info.__dict__[a] = datenum_to_timestring(meas_info.__dict__[a])

            res.append((a, meas_info.__dict__[a]))

        # Now for all metric modules
        for m in self.setup.prj.incl_metrics:
            pkg_path = "metric_defs." + m
            module = pkg_path.split(".")[-1]
            method = getattr(__import__("metric_defs." + m, fromlist=[module]), module)(
                self.setup
            )
            res.extend(method.analyze(meas_info, meas_data))

        return res
        # End of analyze_all

    def save_meas_data(self, meas_info, results):
        # Create the 3-column table and save

        uuid = str(uuid4())
        res = [(uuid, a, b) for a, b in results]

        folder = path_join(self.setup.db_dir, meas_info.folder, self.setup.db_dir)

        # Remove all contents in existing ToDB folder
        if isdir(folder):
            rmtree(folder)

        makedirs(folder)

        filename = path_join(folder, uuid + ".csv")
        with open(filename, "w", newline="") as outfile:
            csv_out = csv_writer(outfile, delimiter=",")
            csv_out.writerow(["uuid", "testitem", "testvalue"])
            for row in res:
                csv_out.writerow(row)

        self.setup.prnt_s(f"Created CSV file: {filename}")

        # End of save_meas_data

    def append_wide_table(self, res):
        # Wide-table format contains fields in the first row, data for the rest
        # Don't include "is_bad" field

        if len(self.wide_table) == 0:
            self.wide_table.append(tuple([a for a, b in res[1:]]))

        self.wide_table.append(tuple([b for a, b in res[1:]]))

        # End of append_wide_table

    def save_wide_table(self, key):
        # Save a wide-table format report

        folder = path_join(self.path_prfx, key)

        if not isdir(folder):
            makedirs(folder)

        filename = f"{key}_{datetime.now().strftime('%Y%m%dT%H%M%S')}.csv"
        filename = path_join(folder, filename)
        with open(filename, "w", newline="") as outfile:
            csv_out = csv_writer(outfile, delimiter=",")
            for row in self.wide_table:
                csv_out.writerow(row)

        self.setup.prnt_s(f"Created report: {filename}")

        # End of save_wide_table

    def update_processed_tracker(self, meas_info):
        # Update tracker for processed SLB or DUT measurement

        tracker = self.setup.ref_proc_list if self.is_ref else self.setup.dat_proc_list
        row = [
            meas_info.folder.replace(self.path_prfx, "").strip(filesep),
            meas_info.calibration,
            unixtime_to_datenum(datetime.now().timestamp()),
        ]

        if self.is_ref:
            row.pop(1)

        with open(tracker, "a", newline="") as outfile:
            csv_out = csv_writer(outfile, delimiter=",")
            csv_out.writerow(row)

        # End of update_processed_tracker
