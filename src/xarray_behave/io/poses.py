"""Poses loaders

should return:
    poses: np.array[frames, flies, body_parts, x/y] - in egocentric coordinates
    poses_allo: np.array[frames, flies, body_parts, x/y] - in allocentric (frame) coordinates
    body_parts: List[str]
    first_pose_frame: int
    last_pose_frame: int
"""
import h5py
import flammkuchen
import numpy as np
import pandas as pd
import xarray as xr
import zarr
from .. import io
import math
from typing import Optional

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


@io.register_provider
class Leap(io.BaseProvider):

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
class DeepPoseKit(io.BaseProvider):

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
                # poses_ego.data[cnt, fly, ...] = [rotate_point(pt, -a[fly]) for pt in p_ego[fly]]
                poses_ego.data[cnt, fly, ...] = rotate_pose(p_ego[fly], -a[fly])

        return poses_ego, poses_allo, ds.poseparts, first_pose_frame, last_pose_frame


@io.register_provider
class Sleap(io.BaseProvider):

    KIND = 'poses'
    NAME = 'sleap'
    SUFFIXES = ['_poses_sleap.h5']

    def load(self, filename: Optional[str] = None):
        if filename is None:
            filename = self.path
        # with h5py.File(filepath, 'r') as f:
        #     pose_parts = f['node_names']
        #     track_names = f['track_names']
        #     track_occupancy = f['track_occupancy'][:]
        #     tracks = f['tracks'][:]

        # poses_allo = tracks.transpose([3, 0, 2, 1])
        # return poses_ego, poses_allo, ds.poseparts, first_pose_frame, last_pose_frame
        logging.warning('Loading SLEAP poses not implemented yet.')
