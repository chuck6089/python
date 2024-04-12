# Author: Colin M. Chow

from collections import defaultdict
from csv import DictReader
from datetime import datetime
from os import scandir, sep
from os.path import isdir, realpath
from time import timezone as time_offset


def read_csv_as_dict_arrays(file: str) -> dict:
    # Read CSV files into DICT of arrays

    d = defaultdict(list)
    with open(file) as inFile:
        reader = DictReader(inFile, skipinitialspace=True, delimiter=",")
        for row in reader:
            for (key, val) in row.items():
                d[key].append(val.rstrip())
    return d
    # End of read_csv_as_dict_arrays


def listdir_dict(path=".", ext="", dir_only=False):
    # List items in a folder in the style of MATLAB 'dir'

    d = {"name": [], "isdir": [], "byte": [], "datenum": []}
    with scandir(path) as itr:
        for f in itr:
            if (ext and not f.name.lower().endswith("." + ext.lower())) or (
                dir_only and not f.is_dir()
            ):
                continue
            d["name"].append(f.name)
            d["isdir"].append(f.is_dir())
            d["byte"].append(f.stat().st_size)
            d["datenum"].append(unixtime_to_datenum(f.stat().st_mtime))
    return d
    # End of listdir_dict


def is_subdir(dir1, dir2):
    # Return true of dir1 is a proper sub-directory of dir2

    if isdir(dir1) and isdir(dir2):
        dir1, dir2 = realpath(dir1), realpath(dir2)
        return dir1.startswith(dir2 + sep)
    else:
        return False
    # End of is_subdir


def wgmet_parse_ini(file: str) -> dict:
    # Parse *.ini file into dict 'd':
    # Text format: <key> = <value(s)>

    d = {}
    with open(file) as inFile:
        for line in inFile:
            # Remove comments, leading and trailing white spaces
            if "#" in line:
                line = line[: line.index("#")]  # Can have inline comments
            line = line.strip()

            if line:
                # Key and value(s) are separated by '='
                # Both can have leading and trailing white space
                # 'val' can have trailing ';'
                # 'val' can represent an array of values separated by ','
                key = line.split("=")[0].strip()
                val = line.split("=")[1].strip().rstrip(";")
                valList = [v.strip() for v in val.split(",")]

                # Check if the values are integers or floats
                remNeg = [v.lstrip("-") for v in valList]
                if all([v.isdigit() for v in remNeg]):
                    valList = [int(v) for v in valList]
                elif all([isfloat(a) for a in valList]):
                    valList = [float(v) for v in valList]
                elif all(
                    [v.lower() == "true" or v.lower() == "false" for v in valList]
                ):
                    valList = [v.lower() == "true" for v in valList]

                d[key] = valList[0] if len(valList) == 1 else valList
    return d
    # End of wgmet_parse_ini


def isfloat(s: str) -> bool:
    # Check if input string is in valid float format:
    # See Leetcode 65: Valid number

    s = s.strip()
    if not s:
        return False

    isDec, isExp, isNum = False, False, False
    for i, c in enumerate(s):
        if c in ["+", "-"]:
            if i != 0 and s[i - 1] != "e":
                return False
        elif c == ".":
            if isDec or isExp:
                return False
            else:
                isDec = True
        elif c == "e":
            if not isNum or isExp:
                return False
            else:
                isExp, isNum = True, False
        elif c.isdigit():
            isNum = True
        else:
            return False

    return isNum
    # End of isfloat


def unixtime_to_datenum(unixtime: float) -> float:
    # Convert unix time to MATLAB's datenum

    _pivot = 719529.0
    _day_in_seconds = 3600 * 24.0
    return (unixtime - time_offset) / _day_in_seconds + _pivot
    # End of unixtime_to_datenum


def datenum_to_timestring(datenum: float) -> str:
    # Convert from MATLAB's datenum to time string

    _pivot = 719529.0
    _day_in_seconds = 3600 * 24.0
    unixtime = (datenum - _pivot) * _day_in_seconds + time_offset

    return datetime.fromtimestamp(unixtime).strftime("%Y%m%dT%H%M%S")
    # End of datenum_to_timestring
