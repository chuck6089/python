# -*- coding: utf-8 -*-

__authors__ = "Colin Chow (colinmec), Alden"
__version__ = ""

from ctypes import memmove
from sys import path as sys_path

from clr import AddReference

# Read *.pmxm Prometric database file using PMEngine.dll as .NET single
# Convert from .NET single to numpy float32

from numpy import dtype as np_dtype, empty as np_empty

AddReference("System")
from System.Runtime.InteropServices import GCHandle, GCHandleType


class pmxm_reader_utils:
    # Helper for setting up PMEngine library path and basic functions
    # (i) get measurement list and (ii) get XYZ from a measurement

    def __init__(self, pmengine_path=""):
        # Allow using existing Prometric installation's library or pyAutoDP attached

        self.pmengine_path = (
            r".\external\prometric\lib" if pmengine_path == "" else pmengine_path
        )
        sys_path.append(self.pmengine_path)

        AddReference("PMEngine")
        AddReference("RadiantCommon")

        import RadiantCommon as RadiantCommon
        import RiPMEngine

        self.pmengine = RiPMEngine.PMEngine()
        self.rad = RadiantCommon

        # End of __init__

    def get_meas_list(self, pmxm_file):
        # Returns a list of measurements with IDs and descriptions

        self.pmengine.MeasurementDatabaseName = pmxm_file
        meaArr = self.pmengine.GetMeasurementList([self.rad.ListItem()], False)
        return [(a.ID, a.Description) for i, a in enumerate(meaArr)]

        # End of get_meas_list

    def get_xyz_arrays(self, meas_id=0, **kwargs):
        # Return measurement data as (X, Y, Z) tuple.
        # X = Z = None for monochromatic camera captures

        if meas_id <= 0 and ("meas_id" not in kwargs or kwargs["meas_id"] <= 0):
            return None
        else:
            if meas_id <= 0:
                meas_id = kwargs["meas_id"]

        if "pmxm_file" in kwargs:
            self.pmengine.MeasurementDatabaseName = kwargs["pmxm_file"]

        meas = self.pmengine.ReadMeasurementFfromDatabase(meas_id)
        x_ptr = meas.GetTristimulusArrayF(self.rad.MeasurementBase.TristimlusType.TrisX)
        y_ptr = meas.GetTristimulusArrayF(self.rad.MeasurementBase.TristimlusType.TrisY)
        z_ptr = meas.GetTristimulusArrayF(self.rad.MeasurementBase.TristimlusType.TrisZ)

        return [net_to_numpy(x_ptr), net_to_numpy(y_ptr), net_to_numpy(z_ptr)]

        # End of get_xyz_arrays

    # End of class pmxm_reader_utils


############################################################################################################


def net_to_numpy(net_arr):
    # A generic function for converting .NET data type to numpy array,
    # via faster in-memory transfer, rather than point-wise conversion

    if not net_arr:
        return None

    _net_to_numpy_map = {
        "Single": np_dtype("float32"),
        "Double": np_dtype("float64"),
        "SByte": np_dtype("int8"),
        "Int16": np_dtype("int16"),
        "Int32": np_dtype("int32"),
        "Int64": np_dtype("int64"),
        "Byte": np_dtype("uint8"),
        "UInt16": np_dtype("uint16"),
        "UInt32": np_dtype("uint32"),
        "UInt64": np_dtype("uint64"),
        "Boolean": np_dtype("bool"),
    }

    net_type = net_arr.GetType().GetElementType().Name
    dims = np_empty(net_arr.Rank, dtype=int)
    for i in range(net_arr.Rank):
        dims[i] = net_arr.GetLength(i)

    try:
        np_arr = np_empty(dims, order="C", dtype=_net_to_numpy_map[net_type])
    except KeyError:
        raise NotImplementedError(f"Unsupported System type: {net_type}")

    try:  # Memmove
        source_handle = GCHandle.Alloc(net_arr, GCHandleType.Pinned)
        source_ptr = source_handle.AddrOfPinnedObject().ToInt64()
        dest_ptr = np_arr.__array_interface__["data"][0]
        memmove(dest_ptr, source_ptr, np_arr.nbytes)
    finally:
        if source_handle.IsAllocated:
            source_handle.Free()

    return np_arr.transpose()
