# -*- coding: utf-8 -*-

__authors__ = "Colin Chow (colinmec), Alden"
__version__ = ""

from os import makedirs
from os.path import isdir, join as path_join, sep as filesep
from shutil import copytree

from file_utils import listdir_dict


class stage_output_class:
    # Copy analysis results (DB CSVs, plots, intermediate output, etc.)
    # from individual measurement folder to staging directory for Manifold loading

    def __init__(self, shared_attr):

        self.setup = shared_attr
        self.is_ref = False
        self.path_prfx = self.setup.set_data_path(self.is_ref)

        # End of __init__

    def stage_analysis_output(self, slb_q, data_q):
        # Loop over SLB and data queues, and
        # copy corresponding analysis output to staging folder (in sub-function)

        if not self.setup.no_slb:
            self.is_ref, self.path_prfx = True, self.setup.set_data_path(True)
            stage_dir = self.setup.proc_ref_dir.strip(filesep)

            for key in slb_q:
                for meas in slb_q[key]:
                    self.copy_to_staging_folder(meas, stage_dir)

        self.is_ref, self.path_prfx = False, self.setup.set_data_path(False)
        stage_dir = self.setup.proc_dat_dir.strip(filesep)

        for key in data_q:
            for meas in data_q[key]:
                self.copy_to_staging_folder(meas, stage_dir)

        # End of stage_analysis_output

    def copy_to_staging_folder(self, meas, stage_dir):
        # Copy analysis output to staging folder
        # 3 analysis output folders supported here:
        # (i) 3-column DB CSVs, (ii) plots, and (iii) intermediate data
        # For DB CSVs, just copy files; for the rest, the root folder
        # "flags" variable indicates which folder and whether it will be copied

        dirs = listdir_dict(path=meas.folder, dir_only=True)
        for name in dirs["name"]:
            flags = [
                (name.startswith(self.setup.db_dir) and self.setup.stage_sql_data)
                or self.is_ref,
                (name.startswith(self.setup.plot_dir_prfx) and self.setup.save_plots)
                or self.is_ref,
                (
                    name.startswith(self.setup.interm_dir_prfx)
                    and self.setup.stage_interm_data
                )
                or self.is_ref,
            ]

            if not any(flags):
                continue

            if flags[0]:
                dest_dir = path_join(stage_dir, self.setup.db_dir)
            elif flags[1] or flags[2]:
                dest_dir = (
                    path_join(stage_dir, self.setup.plot_dir_prfx, name)
                    if flags[1]
                    else path_join(stage_dir, self.setup.interm_dir_prfx, name)
                )

            copytree(path_join(meas.folder, name), dest_dir, dirs_exist_ok=True)

        # End of copy_to_staging_folder

    # End of stage_output_class
