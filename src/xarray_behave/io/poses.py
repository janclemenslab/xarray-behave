"""Poses loaders

should return:
    poses: np.array[frames, flies, body_parts, x/y] - in egocentric coordinates
    poses_allo: np.array[frames, flies, body_parts, x/y] - in allocentric (frame) coordinates
    body_parts: List[str]
    first_pose_frame: int
    last_pose_frame: int
"""
import h5py
import numpy as np
import xarray as xr
import pandas as pd
import zarr
import os
from .. import io
import math
from typing import Optional
import logging
import json


def rotate_point(pos, degrees, origin=(0, 0)):
    """Rotate point.

    Args:
        pos (tuple): (x.y) position
        degrees ([type]): degree by which to rotate
        origin (tuple, optional): point (x,y) around which to rotate point. Defaults to (0, 0).

    Returns:
        tuple: (x, y) rotated
    """
    x, y = pos
    radians = degrees / 180 * np.pi
    offset_x, offset_y = origin
    adjusted_x = (x - offset_x)
    adjusted_y = (y - offset_y)
    cos_rad = math.cos(radians)
    sin_rad = math.sin(radians)
    qx = offset_x + cos_rad * adjusted_x + sin_rad * adjusted_y
    qy = offset_y + -sin_rad * adjusted_x + cos_rad * adjusted_y
    return qx, qy


def rotate_pose(positions, degree, origin=(0, 0)):
    """Rotate point set.

    Args:
        pos (np.ndarray): [points, x/y]
        degrees (float): degree by which to rotate
        origin (Tuple[float, float], optional): point (x,y) around which to rotate point. Defaults to (0, 0).

    Returns:
        tuple: (x, y) rotated
    """
    radians = degree / 180 * np.pi
    cos_rad = np.cos(radians)
    sin_rad = np.sin(radians)
    rot = np.array([[cos_rad, sin_rad], [-sin_rad, cos_rad]])
    positions_centered = positions - origin
    return rot.dot(positions_centered.T).T + origin


def load_skeleton(filename, body_parts):
    skeleton_file = os.path.splitext(filename)[0] + '_skeleton.csv'
    if os.path.exists(skeleton_file):
        logging.info(f'Loading skeleton from {skeleton_file}.')
        df = pd.read_csv(skeleton_file)
        skeleton = df.values
    elif len(body_parts) == 11:  # probably a fly - use default skeleton
        logging.info(f'No skeleton file ({skeleton_file}) found but detected 11 - assume this is a fly.')
        skeleton = np.array([[0, 1], [1, 8], [8, 11],  # body axis
                        [8, 2], [8, 3], [8, 4], [8, 5], [8, 6], [8, 7],  # legs
                        [8, 9], [8, 10]])  # wings
    else:
        logging.info(f'No skeleton file ({skeleton_file}) found and poses do not have 11 body parts - falling back to empty skeleton.')
        skeleton = np.zeros((0, 2), dtype=np.uint)

    return skeleton


class Poses():
    def make(self, filename: Optional[str] = None):
        pixel_size_mm = None
        if filename is None:
            filename = self.path
        poses, poses_allo, body_parts, first_pose_frame, last_pose_frame = self.load(filename)

        # load skeleton from file
        skeleton = load_skeleton(filename, body_parts)

        xr_poses = xr.DataArray(data=poses,
                                 dims=['frame_number', 'flies', 'poseparts', 'coords'],
                                 coords={'frame_number': np.arange(first_pose_frame, last_pose_frame),
                                         'poseparts': body_parts,
                                         'coords': ['y', 'x']},
                                 attrs={'description': 'coords are "allocentric" - rel. to the full frame',
                                        'type': 'poses',
                                        'spatial_units': 'pixels',
                                        'pixel_size_mm': pixel_size_mm,
                                        'loader': self.NAME,
                                        'kind': self.KIND,
                                        'path': filename,
                                        'skeleton': skeleton})

        xr_poses_allo = xr.DataArray(data=poses_allo,
                                 dims=['frame_number', 'flies', 'poseparts', 'coords'],
                                 coords={'frame_number': np.arange(first_pose_frame, last_pose_frame),
                                         'poseparts': body_parts,
                                         'coords': ['y', 'x']},
                                 attrs={'description': 'coords are "allocentric" - rel. to the full frame',
                                        'type': 'poses',
                                        'spatial_units': 'pixels',
                                        'pixel_size_mm': pixel_size_mm,
                                        'loader': self.NAME,
                                        'kind': self.KIND,
                                        'path': filename,
                                        'skeleton': skeleton})
        return xr_poses, xr_poses_allo


@io.register_provider
class Leap(Poses, io.BaseProvider):

    KIND = 'poses'
    NAME = 'leap'
    SUFFIXES = ['_poses.h5', '_poses_leap.h5']

    def load(self, filename: Optional[str] = None):
        """Load pose tracks estimated using LEAP.

        Args:
            filepath ([type]): [description]

        Returns:
            poses, poses_allo, body_parts, first_pose_frame, last_pose_frame
        """
        if filename is None:
            filename = self.path

        with h5py.File(filename, 'r') as f:
            box_offset = f['box_centers'][:]
            box_angle = f['fixed_angles'][:].astype(np.float64)
            box_size = (f['box_size'].attrs['i0'], f['box_size'].attrs['i1'])
            poses = f['positions'][:]
            nb_flies = int(np.max(f['fly_id']) + 1)
        body_parts = ['head', 'neck', 'front_left_leg', 'middle_left_leg', 'back_left_leg', 'front_right_leg',
                      'middle_right_leg', 'back_right_leg', 'thorax', 'left_wing', 'right_wing', 'tail']
        # need to offset and rotate
        last_pose_index = np.argmin(poses[:, 0, 0] > 0)
        first_pose_index = np.argmin(poses[:, 0, 0] == 0)
        box_offset = box_offset[first_pose_index:last_pose_index, ...]
        box_angle = box_angle[first_pose_index:last_pose_index, ...]
        poses = poses[first_pose_index:last_pose_index, ...]
        poses = poses.astype(np.float64)

        # transform poses back to frame coordinates
        origin = [b / 2 for b in box_size]
        poses_allo = np.zeros_like(poses)
        for cnt, ((x, y), ang, p_ego) in enumerate(zip(box_offset, box_angle, poses)):
            # tmp = [rotate_point(pt, a, origin) for pt in p_ego]  # rotate
            tmp = rotate_pose(p_ego, float(ang), origin)
            p_allo = np.stack(tmp) + np.array((x, y)) - origin  # translate
            poses_allo[cnt, ...] = p_allo

        # center poses around thorax
        thorax_idx = 8
        thorax_pos = poses[:, thorax_idx, :]
        poses = poses - thorax_pos[:, np.newaxis, :]  # subtract remaining thorax position
        origin = (0, 0)

        # rotate poses such that the head is at 0 degrees
        head_idx = 0
        head_pos = poses[:,  head_idx, :]
        head_thorax_angle = 90 + np.arctan2(head_pos[:, 0], head_pos[:, 1]) * 180 / np.pi
        for cnt, (a, p_ego) in enumerate(zip(head_thorax_angle, poses)):
            # poses[cnt, ...] = [rotate_point(pt, -a, origin) for pt in p_ego]
            poses[cnt, ...] = rotate_pose(p_ego, -a, origin)

        box_offset = box_offset.reshape((-1, nb_flies, *box_offset.shape[1:]))  # "unfold" fly dimension
        box_angle = box_angle.reshape((-1, nb_flies, *box_angle.shape[1:]))  # "unfold" fly dimension
        poses = poses.reshape((-1, nb_flies, *poses.shape[1:]))  # "unfold" fly dimension
        poses_allo = poses_allo.reshape((-1, nb_flies, *poses_allo.shape[1:]))  # "unfold" fly dimension

        first_pose_frame = int(first_pose_index / nb_flies)
        last_pose_frame = int(last_pose_index / nb_flies)

        return poses, poses_allo, body_parts, first_pose_frame, last_pose_frame


@io.register_provider
class DeepPoseKit(Poses, io.BaseProvider):

    KIND = 'poses'
    NAME = 'deepposekit'
    SUFFIXES = ['_poses_dpk.zarr']

    def load(self, filename: Optional[str] = None):
        """Load pose tracks estimated using DeepPoseKit.

        Args:
            filepath (str): name of the file produced by DeepPoseKit

        Returns:
            poses_ego (xarray.DataArray): poses in EGOcentric (centered around thorax, head aligned straight upwards), [frames, flies, bodypart, x/y]
            poses_allo (xarray.DataArray): poses in ALLOcentric (frame) coordinates, [frames, flies, bodypart, x/y]
            partnames (List[str]): list of names for each body part (is already part of the poses_ego/allo xrs)
            first_pose_frame, last_pose_frame (int): frame corresponding to first and last item in the poses_ego/allo arrays (could attach this info as xr dim)
        """
        if filename is None:
            filename = self.path

        with zarr.ZipStore(filename, mode='r') as zarr_store:
            ds = xr.open_zarr(zarr_store).load()  # the final `load()` prevents
        nb_flies = len(ds.flies)
        box_size = np.array(ds.attrs['box_size'])

        poses_allo = ds.poses + ds.box_centers - box_size/2

        first_pose_frame = int(np.argmin(np.isnan(ds.poses.data[:, 0, 0, 0]).data))
        last_pose_frame = int(np.argmin(~np.isnan(np.array(ds.poses.data[first_pose_frame:, 0, 0, 0]).data)) + first_pose_frame)
        if last_pose_frame == first_pose_frame or last_pose_frame == 0:
            last_pose_frame = ds.poses.shape[0]

        # CUT to first/last frame with poses
        poses_ego = ds.poses[first_pose_frame:last_pose_frame, ...]
        poses_allo = poses_allo[first_pose_frame:last_pose_frame, ...]

        poses_ego = poses_ego - poses_ego.sel(poseparts='thorax')  # CENTER egocentric poses around thorax

        # ROTATE egocentric poses such that the angle between head and thorax is 0 degrees (straight upwards)
        head_thorax_angle = 270 + np.arctan2(poses_ego.sel(poseparts='head', coords='y'),
                                             poses_ego.sel(poseparts='head', coords='x')) * 180 / np.pi
        for cnt, (a, p_ego) in enumerate(zip(head_thorax_angle.data, poses_ego.data)):
            for fly in range(nb_flies):
                poses_ego.data[cnt, fly, ...] = rotate_pose(p_ego[fly], -a[fly])

        return poses_ego, poses_allo, ds.poseparts, first_pose_frame, last_pose_frame


@io.register_provider
class Sleap(Poses, io.BaseProvider):

    KIND = 'poses'
    NAME = 'sleap'
    SUFFIXES = ['_poses_sleap.h5', '_sleap.h5.slp']

    def load(self, filename: Optional[str] = None):
        if filename is None:
            filename = self.path

        with h5py.File(filename, 'r') as f:

            if 'node_names' not in f:
                meta = json.loads(f['metadata'].attrs['json'])
                pose_parts = []
                for node in meta['nodes']:
                    pose_parts.append(node['name'])
            else:
                pose_parts = f['node_names'][:]

            # track_names = f['track_names']
            # track_occupancy = f['track_occupancy'][:]
            tracks = f['tracks'][:]  # flies, (x/y), bodypart, frame

        poses_allo = tracks.transpose([3, 0, 2, 1])   # -> [frames, flies, bodypart, x/y]
        poses_allo = poses_allo[..., ::-1]  # swap x/y
        poses_ego = poses_allo.copy()
        pose_parts = [pose_part.decode() for pose_part in pose_parts]  # from byte to str

        first_pose_frame = 0
        last_pose_frame = poses_allo.shape[0]

        return poses_ego, poses_allo, pose_parts, first_pose_frame, last_pose_frame



# @io.register_provider
# class CSV_poses(Poses, io.BaseProvider):

#     KIND = 'tracks'
#     NAME = 'generic csv'
#     SUFFIXES = ['_tracks.csv']

#     def load(self, filename: Optional[str] = None):
#         """Load tracker data from CSV file.

#         Head of the CSV file should look like this:
#         track	track1	track1	track1	track1	track1	track1
#         part	a	    a	    b	    b  	    c	    c
#         coord	x	    y	    x	    y	    x	    y
#         0       0.09    0.09    0.09    0.09    0.09    0.09
#         1       0.09    0.09    0.09    0.09    0.09    0.09
#         2       0.09    0.09    0.09    0.09    0.09    0.09
#         ...

#         First column contains the framenumber (does not need to start at 0),
#         remaining columns contain coordinate date for different tracks/trackparts.

#         Args:
#             filename (Optional[str], optional): Path to the CSV file. Defaults to None.

#         Returns:
#             x: np.array([frames, tracks, parts, coords (y/x)]), track_names: List[], track_parts: List[], frame_numbers: np.array[np.intp]
#         """

#         if filename is None:
#             filename = self.path
#         logging.warning('Loading tracks from CSV.')
#         df = pd.read_csv(filename, header=[0, 1, 2], index_col=0)
#         track_names = df.columns.levels[df.columns.names.index('track')].to_list()
#         track_parts = df.columns.levels[df.columns.names.index('part')].to_list()
#         track_coord = df.columns.levels[df.columns.names.index('coord')].to_list()
#         coord_order = [track_coord.index('y'), track_coord.index('x')]
#         x = np.reshape(df.values, (-1, len(track_names), len(track_parts), len(track_coord)))
#         x = x[..., coord_order]
#         frame_numbers = df.index.to_numpy().astype(np.intp)
#         # return x, track_names, track_parts, frame_numbers
#         return poses_ego, poses_allo, ds.poseparts, first_pose_frame, last_pose_frame

