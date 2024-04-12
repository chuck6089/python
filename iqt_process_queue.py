# -*- coding: utf-8 -*-

__authors__ = "Colin Chow (colinmec), Alden"
__version__ = "iqt.0.00"

from os.path import join as path_join

import numpy as np
from external.prometric.pmxm_reader_utils import pmxm_reader_utils
from image_utils import rotate_image
from met_tools.iqt.radiant_camera_corrections import radiant_camera_corrections
from PIL.Image import fromarray
from process_queue_class import process_queue_class
from scipy.ndimage import find_objects, label
from scipy.ndimage.morphology import binary_erosion

############################################################################################################


class data:
    # Data structure for IQT

    def __init__(self):

        self.ret_center = [-1, -1]
        self.angle0 = 0

        # These are images with tuple keys that identify (color, pupil)
        # Each contains a 3-element list of tristimulus XYZ
        self.align = {}
        self.clear = {}
        self.pos_checker = {}
        self.neg_checker = {}

        # These are arrays with string keys that identify color
        self.crop_range_x = {}
        self.crop_range_y = {}
        self.fov_axis_x = {}
        self.fov_axis_y = {}
        
        # This contains 2D masking arrays with tuple keys (color, name)
        self.masks = {}

        # End of __init__

    # End of class data


############################################################################################################


class iqt_process_queue(process_queue_class):
    # The bulk of codes here pertains to data loading sequence.
    # Analysis packages are organized separately in metric_defs.

    def __init__(self, shared_attr):

        super().__init__(shared_attr)

        self.cam_correct = radiant_camera_corrections(
            self.setup.depend_dir, self.setup.prj
        )
        self.pmxm_reader = pmxm_reader_utils(self.setup.pmengine_path)
        self.calibr = None

        # End of __init__

    def load_meas_data(self, meas):
        # Top-level data loading sequence for SLB and DUT measurements

        self.setup.prnt_s(f"Loading: {meas.dut_id}")

        # Initialize data object and load SLB for DUT measurement
        self.meas_data = data()
        self.calibr = self.slb_saved[meas.calibration] if not self.is_ref else None

        # Read from tester output files, then run image corrections
        self.read_data_from_tester_ouput(meas)
        self.get_alignment()
        self.run_image_calibration()

        # Crop FOV
        self.get_crop_range()
        self.crop_fov()

        if self.is_ref:
            self.resize_slb_clear()
        else:
            self.convert_to_efficiency()

            if (
                meas.eye_side.lower() == "left" and self.setup.prj.is_flipped_dut[0]
            ) or (
                meas.eye_side.lower() == "right" and self.setup.prj.is_flipped_dut[1]
            ):
                self.rotate_data()

        return self.meas_data
        # End of load_meas_data

    def read_data_from_tester_ouput(self, meas):
        # See external.prometric.pmxm_reader_utils

        pmxm_file = [f for f in meas.data_files if f.endswith(".pmxm")][0]
        pmxm_file = path_join(meas.folder, pmxm_file)

        for i, description in self.pmxm_reader.get_meas_list(pmxm_file):
            arr = description.split(";")
            pupil = get_meas_pupil(
                arr[-1], self.setup.prj.pupil_labels, self.setup.prj.pupil_in_descp
            )
            meas_type, color = get_meas_type(arr), get_meas_color(arr)
            xyz = self.pmxm_reader.get_xyz_arrays(pmxm_file=pmxm_file, meas_id=i)

            # Rotate 180-deg if camera is mounted facing "the other" way
            if self.setup.prj.is_flipped_cam:
                xyz = [m[::-1, ::-1] if m is not None else None for m in xyz]

            self.meas_data.__dict__[meas_type][(color, pupil)] = xyz

        # End of read_data_from_pmxm

    def get_alignment(self):
        # Get reticle center and rotation angle from alignment image (5 of 6 points)
        # First find coordinates of those points, then compared with reference from *ini file

        if self.is_ref and self.meas_data.align == {}:
            msg = "Error: SLB alignment (ghosting) measurement not found."
            self.setup.prnt_s(msg)
            raise Exception(msg)

        tol = self.setup.prj.five_pnts_ang_tolr / np.mean(
            np.diff(self.cam_correct.x_raw[None])
        )

        ref = np.array(
            list(zip(self.setup.prj.five_pnts_ref_y, self.setup.prj.five_pnts_ref_x))
        )
        found, angle1, angle2 = np.full(ref.shape, -1), np.nan, np.nan

        img = self.cam_correct.correct_distortion(
            self.meas_data.align[list(self.meas_data.align)[0]]
        )
        coord, weight = get_connected_comp(img[1], self.setup.prj.five_pnts_rel_thrshld)
        points = np.array([a for _, a in sorted(zip(weight, coord), reverse=True)])

        for p in points:
            if np.all(found >= 0):
                break

            for j, q in enumerate(ref):
                if np.all(found[j, :] != -1):
                    continue

                if np.sum((p - q) ** 2) < tol**2:
                    found[j] = p
                    break

        self.meas_data.ret_center = [found[4, 1], found[4, 0]]

        if self.is_ref:
            self.get_slb_rotation_angle(found)
        else:
            self.meas_data.angle0 = self.calibr.angle0

        # End of get_alignment

    def get_slb_rotation_angle(self, pnts_found):
        # Get FOV rotation angle for SLB measurements (to be used for DUT)

        if not np.all(pnts_found > 0):
            self.setup.prnt_s("Warning: Some SLB alignment points missing.")

        if any(a == -1 for a in self.meas_data.ret_center):
            self.setup.prnt_s(
                "Warning: SLB reticle center cannot be determined. Default used."
            )
            self.meas_data.ret_center = self.cam_correct.center_pos.copy()

        if np.all(pnts_found[0] > 0) and np.all(pnts_found[1] > 0):
            angle1 = -np.rad2deg(np.arctan2(*(pnts_found[1] - pnts_found[0])))
        if np.all(pnts_found[2] > 0) and np.all(pnts_found[3] > 0):
            angle2 = -np.rad2deg(np.arctan2(*(pnts_found[2] - pnts_found[3])))
        if not np.isnan(angle1) or not np.isnan(angle2):
            self.meas_data.angle0 = np.nanmean([angle1, angle2])

        # End of get_slb_rotation_angle

    def run_image_calibration(self):
        # Correct for:
        # CCD vertical streaking (SLB only), ND filter non-uniformity (SLB only),
        # lens distortion and image rotation

        p = self.meas_data.__dict__
        images = [
            p[a]
            for a in p
            if (type(p[a]) is dict and len(p[a]) > 0 and type(list(p[a])[0]) is tuple)
        ]

        for d in images:
            for key in d:
                if self.is_ref:
                    # Need y-pixel limit for CCD vertical streaking removal
                    self.cam_correct.get_y_pixel_limit(
                        self.meas_data.ret_center, self.meas_data.angle0
                    )
                    # Correct for CCD vertical streaking and ND-filter non-uniformity
                    d[key] = self.cam_correct.calibrate_nd_filter(
                        self.cam_correct.remove_vert_ccd_background(d[key]), key[0]
                    )

                # Lens distortion correction
                d[key] = self.cam_correct.correct_distortion(d[key], key[0])

                # Image rotation, if too off
                if abs(self.meas_data.angle0) > 0.25:
                    d[key] = [
                        rotate_image(
                            m, -self.meas_data.angle0, self.meas_data.ret_center
                        )
                        for m in d[key]
                    ]

        # End of run_image_calibration

    def get_crop_range(self):
        # For SLB, use edge detection to get crop range.
        # For DUT, use SLB range, but center accordingly.

        if self.is_ref:
            self.get_slb_crop_range()
        else:
            if all(d > 0 for d in self.meas_data.ret_center):
                for c in self.setup.prj.colors:
                    self.meas_data.crop_range_x[c] = (
                        self.calibr.crop_range_x[c]
                        + self.meas_data.ret_center[0]
                        - self.calibr.ret_center[0]
                    )
                    self.meas_data.crop_range_y[c] = (
                        self.calibr.crop_range_y[c]
                        + self.meas_data.ret_center[1]
                        - self.calibr.ret_center[1]
                    )
            else:
                self.setup.prnt_s(
                    "Warning: Auto-alignment unsuccessful. SLB crop range used."
                )
                for c in self.setup.prj.colors:
                    self.meas_data.crop_range_x[c] = self.calibr.crop_range_x[c]
                    self.meas_data.crop_range_y[c] = self.calibr.crop_range_y[c]

            self.check_range()

        # Designate x and y axis within FOV
        for c in self.setup.prj.colors:
            self.meas_data.fov_axis_x[c] = (
                self.cam_correct.x_raw[c][self.meas_data.crop_range_x[c]]
                - self.cam_correct.x_raw[c][self.meas_data.ret_center[0]]
            )
            self.meas_data.fov_axis_y[c] = (
                self.cam_correct.y_raw[c][self.meas_data.crop_range_y[c]]
                - self.cam_correct.y_raw[c][self.meas_data.ret_center[1]]
            )
            if not self.is_ref:
                self.meas_data.fov_axis_x[c] -= self.setup.prj.fov_offset[0]
                self.meas_data.fov_axis_y[c] -= self.setup.prj.fov_offset[1]

        # End of get_crop_range

    def check_range(self):
        # For DUT only. Make sure that the ranges aren't beyond supported reticle and camera FOV

        for c in self.setup.prj.colors:
            rngX = np.where(
                np.logical_and(
                    self.cam_correct.x_raw[c] - self.setup.prj.fov_offset[0]
                    >= -self.setup.prj.full_fov[0] / 2,
                    self.cam_correct.x_raw[c] - self.setup.prj.fov_offset[0]
                    <= self.setup.prj.full_fov[0] / 2,
                )
            )[0]
            rngY = np.where(
                np.logical_and(
                    self.cam_correct.y_raw[c] - self.setup.prj.fov_offset[1]
                    >= -self.setup.prj.full_fov[1] / 2,
                    self.cam_correct.y_raw[c] - self.setup.prj.fov_offset[1]
                    <= self.setup.prj.full_fov[1] / 2,
                )
            )[0]

            x1 = max([self.meas_data.crop_range_x[c][0], 0, rngX[0]])
            y1 = max([self.meas_data.crop_range_y[c][0], 0, rngY[0]])
            x2 = min(
                [
                    self.meas_data.crop_range_x[c][-1],
                    len(self.cam_correct.x_raw[c]),
                    rngX[-1],
                ]
            )
            y2 = min(
                [
                    self.meas_data.crop_range_y[c][-1],
                    len(self.cam_correct.y_raw[c]),
                    rngY[-1],
                ]
            )

            self.meas_data.crop_range_x[c] = list(range(x1, x2 + 1))
            self.meas_data.crop_range_y[c] = list(range(y1, y2 + 1))

        # End of check_range

    def get_slb_crop_range(self):
        # Use scipy.ndimage to find crop range for SLB clear images

        for c in self.meas_data.clear:
            img = self.meas_data.clear[c][1]
            m, n = img.shape
            mask = img / np.max(img) > self.setup.prj.crop_edge_threshold
            mask = binary_erosion(mask, structure=np.ones((3, 3)))

            labeled_img, _ = label(mask)
            slices = find_objects(labeled_img)
            slices = [
                ((s[0].stop - s[0].start) * (s[1].stop - s[1].start), s) for s in slices
            ]
            slices.sort(reverse=True)

            if len(slices) == 0 or slices[0][0] < 64:
                return [], []

            # +/- 1 to compensate for binary erosion
            y_min, y_max = max(slices[0][1][0].start - 1, 0), min(
                slices[0][1][0].stop + 1, m
            )
            x_min, x_max = max(slices[0][1][1].start - 1, 0), min(
                slices[0][1][1].stop + 1, n
            )

            self.meas_data.crop_range_y[c[0]] = list(
                range(
                    y_min + self.setup.prj.crop_margin,
                    y_max - self.setup.prj.crop_margin,
                )
            )
            self.meas_data.crop_range_x[c[0]] = list(
                range(
                    x_min + self.setup.prj.crop_margin,
                    x_max - self.setup.prj.crop_margin,
                )
            )

        # End of get_crop_range

    def crop_fov(self):
        # Crop FOV using pre-calculated ranges

        for key in self.meas_data.clear:
            rng_x = self.meas_data.crop_range_x[key[0]]
            rng_y = self.meas_data.crop_range_y[key[0]]

            for i, img in enumerate(self.meas_data.clear[key]):
                if img is None:
                    continue

                self.meas_data.clear[key][i] = img[np.array(rng_y)[:, None], rng_x]

        # End of crop_fov

    def resize_slb_clear(self):
        # Resize SLB clear pattern images to reduce memory and for later efficiency calibration

        for key in self.meas_data.clear:
            for i, img in enumerate(self.meas_data.clear[key]):
                if img is None:
                    continue

                new_size = np.flip(np.int32(np.ceil(np.array(img.shape) / 3)))
                self.meas_data.clear[key][i] = np.array(fromarray(img).resize(new_size))

        # End of resize_slb_clear

    def convert_to_efficiency(self):
        # Take DUT-clear-type to SLB-clear-type ratio, pixel-by-pixel to get efficiency
        # and save as clear-type

        slb_clear = {}
        for c in self.setup.prj.colors:
            key = [k for k in list(self.calibr.clear.keys()) if k[0] == c][0]
            img = self.calibr.clear[key][1]
            new_size = (
                len(self.meas_data.crop_range_x[c]),
                len(self.meas_data.crop_range_y[c]),
            )
            slb_clear[c] = np.array(fromarray(img).resize(new_size))

        for key in self.meas_data.clear:
            for i, img in enumerate(self.meas_data.clear[key]):
                if img is None:
                    continue
                self.meas_data.clear[key][i] = img / slb_clear[key[0]]

        # End of convert_to_efficiency

    def rotate_data(self):
        # First setup axis accordingly, then rotation all images by 180-deg
        
        for c in self.setup.prj.colors:
            self.meas_data.fov_axis_x[c] = (
                -np.flip(self.meas_data.fov_axis_x[c])
                - 2 * self.setup.prj.fov_offset[0]
            )
            self.meas_data.fov_axis_y[c] = (
                -np.flip(self.meas_data.fov_axis_y[c])
                - 2 * self.setup.prj.fov_offset[1]
            )

        p = self.meas_data.__dict__
        images = [
            p[a]
            for a in p
            if (type(p[a]) is dict and len(p[a]) > 0 and type(list(p[a])[0]) is tuple)
        ]

        for d in images:
            for key in d:
                d[key] = [m[::-1, ::-1] if m is not None else None for m in d[key]]

        # End of rotate_data

    # End of class iqt_process_queue


############################################################################################################


def get_meas_type(info):
    # Get reticle pattern type from measurement info

    pattern_map = {
        "ghosting": "align",
        "clear": "clear",
        "negchecker": "neg_checker",
        "checker": "pos_checker",
    }

    if info[1].lower() in pattern_map:
        return pattern_map[info[1].lower()]

    if info[0].lower() in pattern_map:
        return pattern_map[info[0].lower()]

    return None
    # End of get_meas_type


def get_meas_color(info):
    # Get input color from measurement info

    color_map = {"B": "blue", "G": "green", "R": "red"}

    if info[-2][-1] in color_map:
        return color_map[info[-2][-1]]

    if info[-1][-1] in color_map:
        return color_map[info[-1][-1]]

    return None
    # End of get_meas_color


def get_meas_pupil(info, labels, label_pos=-1):
    # Get pupil location name from measurement info

    info = info.split("_")[label_pos]
    labels = [str(a) if type(a) is int else a for a in labels]

    for a in labels:
        label_arr = a.split("/")
        for s in label_arr:
            if info.lower() == s.lower():
                return label_arr[0]

    return ""
    # End of get_meas_pupil


def get_connected_comp(img, thrshld):
    # Find connected components in input image that signifies alignment dots
    # Note that size of slice is width x height, as opposed to number of pixels in Matlab

    min_dot_size, max_dot_size = 9, 121
    coord, weight = [], []

    img_mod = img - np.nanmedian(img)
    img_mod = img_mod / np.nanmax(img_mod)
    img_bw = (img_mod > thrshld).astype(int)

    labeled_data, _ = label(img_bw)
    object_slices = find_objects(labeled_data)

    for slc in object_slices:
        cut = img_mod[slc]
        if cut.size < min_dot_size or cut.size > max_dot_size:
            continue

        denom = np.sum(cut)
        weighted_x = np.sum(cut * np.arange(slc[1].start, slc[1].stop)) / denom
        weighted_y = np.sum(cut.T * np.arange(slc[0].start, slc[0].stop)) / denom

        coord.append((round(weighted_y), round(weighted_x)))
        weight.append(denom)

    return coord, weight
    # End of get_connected_comp
