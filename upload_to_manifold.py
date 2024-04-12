# -*- coding: utf-8 -*-

__authors__ = "Colin Chow (colinmec), Alden"
__version__ = ""

import argparse

from os import listdir, makedirs, remove
from os.path import isdir, isfile, join as path_join
from shutil import copytree, move, rmtree
from subprocess import run
from sys import stdout as sys_stdout

from file_utils import read_csv_as_dict_arrays

############################################################################################################


class Logger(object):
    # Standard logger class that prints messages to both console and log file

    def __init__(self):

        self.terminal = sys_stdout
        self.flush = sys_stdout.flush

        # End of __init__

    def write(self, message):

        self.terminal.write(message)
        with open("upload_to_manifold.log", "a", encoding="utf-8") as log:
            log.write(message)

        # End of write


############################################################################################################

# Global sub-directory name for archived folder of transferred contents
archive_dir_name = "transferred"

############################################################################################################


def read_map_file(map_file, prnt_s):
    # Source-destination mapping is a CSV file with columns: source, destination

    mapping = read_csv_as_dict_arrays(map_file)
    if any(s not in mapping for s in ["source", "destination"]):
        msg = f"Error: Incorrectly formatted source-destination mapping in file {map_file}"
        prnt_s(msg)
        raise Exception(msg)

    return mapping
    # End of read_map_file


def copy_source_to_manifold(src, dest, prnt_s):
    # Copy contents of source directory <src> to Manifold

    for s in listdir(src):
        if s == archive_dir_name:
            continue

        if isdir(path_join(src, s)):
            msg = run(
                [
                    "manifold",
                    "-vip",
                    "putr",
                    "--overwrite",
                    path_join(src, s),
                    dest + "/" + s,
                ]
            )
        else:
            msg = run(
                [
                    "manifold",
                    "-vip",
                    "put",
                    "--overwrite",
                    path_join(src, s),
                    dest + "/" + s,
                ]
            )

        prnt_s(str(msg))

    # End of copy_source_to_manifold


def move_source_to_archive(src, prnt_s):
    # Move contents of source directory <src>, except <archive_dir_name> to the
    # sub-folder <archive_dir_name>

    archive_folder = path_join(src, archive_dir_name)

    for s in listdir(src):
        if s == archive_dir_name:
            continue

        source_path = path_join(src, s)
        archive_path = path_join(archive_folder, s)

        if isdir(archive_path):
            rmtree(archive_path)
        elif isfile(archive_path):
            remove(archive_path)

        move(source_path, archive_folder, copy_function=copytree)
        prnt_s(f"Moved {source_path} to transferred archive folder.")

    # End of move_source_to_archive


def upload_to_manifold(source_dest_mapping="", handles=None):
    # Read source-destination mapping info from a CSV file, then upload to manifold
    # After upload is done, move content to "transferred"
    # The optional input handle is an object with the attribute "prnt_s".
    # When this function is called from pyAutoDP, or other applications,
    # this can be use to route output messages to the interface of calling applications.

    prnt_s = handles.prnt_s if handles is not None else print
    if not isfile(source_dest_mapping):
        msg = f"Error: File {source_dest_mapping} not found."
        prnt_s(msg)
        raise Exception(msg)
    else:
        mapping = read_map_file(source_dest_mapping, prnt_s)

    for i, src in enumerate(mapping["source"]):
        dest = mapping["destination"][i]

        if src is None or src == "" or dest is None or dest == "":
            prnt_s(
                f"Warning: Empty source or destination in row {str(i)}: ({src}, {dest})"
            )
            continue

        if not isdir(src):
            prnt_s(f"Warning: Source directory {src} not found.")
            continue

        copy_source_to_manifold(src, dest, prnt_s)
        move_source_to_archive(src, prnt_s)

    prnt_s("Done loading Manifold.\n")
    # End of upload_to_manifold


############################################################################################################

if __name__ == "__main__":
    stdout = Logger()
    try:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "-m",
            "--source-dest-mapping",
            help="A CSV file that maps local source to manifold folders",
            required=True,
        )

        args = parser.parse_args()
        upload_to_manifold(**args.__dict__)

    except Exception as errmsg:
        stdout.write(str(errmsg) + "\n\n")

    input("Press enter to exit.")
