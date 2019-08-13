import numpy as np
import itertools
import xarray as xr


def distance(positions):
    """ arg:
            positions: positions of the thorax. [time, flies, y/x]
        returns:
            dists: distances between flies. [time, flies, flies]
    """
    nagents = positions.shape[1]

    dis = np.empty((positions.shape[0], nagents, nagents))
    dis.fill(np.nan)

    for i, j in itertools.combinations(range(nagents), r=2):
        diff_ij = np.squeeze(np.diff(positions[:, [i, j], :], axis=1))
        distance = np.sqrt(np.einsum('ij,ij->i', diff_ij, diff_ij))
        dis[:, i, j] = distance
        dis[:, j, i] = distance

    return dis


def velocity(pos1, pos2: np.array = None, timestep: float = 1, ref: str = 'self'):
    """
    arg:
        pos1: position of vector's base, center of agent. [time, agent, y/x]
        pos2: position of vector's head, head of agent. [time, agent, y/x]
        timestep: time difference between data points (default = 1)
        ref: type of output velocity. 'self' for forward/lateral velocity components, 'chamber' for y/x components.
    returns:
        vels: change of variable with respect to time. [time,agent,y/x]
    """

    # velocity in reference to chamber
    vels_yx = np.gradient(pos1, timestep, axis=0)

    if ref == 'self':
        vels_mags = np.linalg.norm(vels_yx, axis=2)    # velocity magnitude
        angle_diff = np.arctan2(vels_yx[..., 0], vels_yx[..., 1]) - np.arctan2(pos2[..., 0] - pos1[..., 0], pos2[..., 1] - pos1[..., 1])    # angle difference between velocity vector and orientation vector
        vels = np.empty_like(vels_yx)    # velocity components with reference to self orientation
        vels[:, :, 0] = vels_mags*np.cos(angle_diff)    # forward
        vels[:, :, 1] = -vels_mags*np.sin(angle_diff)    # lateral
    elif ref == 'chamber':
        vels = vels_yx

    return vels


def acceleration(pos1, pos2: np.array = None, timestep: float = 1, ref: str = 'self'):
    """
    arg:
        pos1: position of vector's base, center of agent. [time, flies, y/x]
        pos2: position of vector's head, head of agent. [time, flies, y/x]
        timestep: time difference between data points (default = 1)
        ref: type of output components. 'self' for forward/lateral components, 'chamber' for y/x components.
    returns:
        accs: second order change of variable with respect to time. [time,agent,y/x]
    """

    # acceleration in reference to chamber
    accs_yx = np.gradient(np.gradient(pos1, timestep, axis=0), timestep, axis=0)

    if ref == 'self':
        accs_mags = np.linalg.norm(accs_yx, axis=2)    # acceleration magnitude
        angle_diff = np.arctan2(accs_yx[..., 0], accs_yx[..., 1]) - np.arctan2(pos2[..., 0] - pos1[..., 0], pos2[..., 1] - pos1[..., 1])    # angle difference between acceleration vector and orientation vector
        accs = np.empty_like(accs_yx)    # acceleration components with reference to self orientation
        accs[:, :, 0] = accs_mags*np.cos(angle_diff)    # forward
        accs[:, :, 1] = -accs_mags*np.sin(angle_diff)    # lateral
    elif ref == 'chamber':
        accs = accs_yx

    return accs


def angle(pos1, pos2):
    """
    arg:
        pos1: position of vector's base, center of agent. [time, agent, y/x]
        pos2: position of vector's head, head of agent. [time, agent, y/x]
    returns:
        angles: orientation of flies with respect to chamber. 0 degrees when flies look towards positive x axis, 90 degrees when vertical and looking upwards. [time, flies]
    """
    return np.arctan2(pos2[..., 0] - pos1[..., 0], pos2[..., 1] - pos1[..., 1]) * 180 / np.pi


def relative_angle(pos1, pos2):
    """ Angle between agents. An element (k,i,j) from the output is the angle at kth sample between ith (reference head) and jth (target base).
    arg:
        pos1: positions of the thoraxes for all flies. [time, flies, y/x]
        pos2: positions of the heads for all flies. [time, flies, y/x]
    returns:
        rel_angles: orientation of flies with respect to chamber. [time, flies, flies]
    """

    nagents = pos1.shape[1]
    rel_angles = np.empty((pos1.shape[0], nagents, nagents), dtype=np.float32)
    rel_angles.fill(np.nan)

    for i, j in itertools.combinations(range(nagents), r=2):
        x1 = pos2[:, i, 1]-pos1[:, i, 1]
        y1 = pos2[:, i, 0]-pos1[:, i, 0]
        x2 = pos1[:, j, 1]-pos2[:, i, 1]
        y2 = pos1[:, j, 0]-pos2[:, i, 0]
        dot = x1*x2 + y1*y2
        det = x1*y2 - y1*x2
        rel_angles[:, i, j] = np.arctan2(det, dot)

        x1 = pos2[:, j, 1]-pos1[:, j, 1]
        y1 = pos2[:, j, 0]-pos1[:, j, 0]
        x2 = pos1[:, i, 1]-pos2[:, j, 1]
        y2 = pos1[:, i, 0]-pos2[:, j, 0]
        dot = x1*x2 + y1*y2
        det = x1*y2 - y1*x2
        rel_angles[:, j, i] = np.arctan2(det, dot)

    return rel_angles*180/np.pi


def rot_speed(pos1, pos2, timestep: float = 1):
    """
    arg:
        pos1: position of vector's base, center of agent. [time, flies, y/x]
        pos2: position of vector's head, head of agent. [time, flies, y/x]
    returns:
        rot_speed: rotational speed. [time, flies]
    """

    # orientations
    angles = np.arctan2(pos2[..., 0] - pos1[..., 0], pos2[..., 1] - pos1[..., 1])

    # unwrap orientations
    unwrapped_angles = np.unwrap(angles, axis=0)*180/np.pi

    # rotational speed
    rot_speed = np.gradient(unwrapped_angles, timestep, axis=0)

    return rot_speed


def rot_acceleration(pos1, pos2, timestep: float = 1):
    """
    arg:
        pos1: position of vector's base, center of agent. [time, flies, y/x]
        pos2: position of vector's head, head of agent. [time, flies, y/x]
    returns:
        rot_accs: rotational acceleration. [time, flies]
    """

    # orientations
    angles = np.arctan2(pos2[..., 0] - pos1[..., 0], pos2[..., 1] - pos1[..., 1])

    # unwrap orientations
    unwrapped_angles = np.unwrap(angles, axis=0)*180/np.pi

    # rotational speed
    rot_accs = np.gradient(np.gradient(unwrapped_angles, timestep, axis=0), timestep, axis=0)

    return rot_accs


def assemble_metrics(dataset):
    """ arg:
            dataset: xarray.Dataset from datastructure module, which takes experiment name as input.
        return:
            feature_dataset: xarray.Dataset with collection of calculated features.
                features:
                    angles [time,flies]
                    vels [time,flies,forward/lateral]
                    chamber_vels [time,flies,y/x]
                    rotational_speed [time,flies]
                    accelerations [time,flies,forward/lateral]
                    chamber_acc [time,flies,y/x]
                    rotational_acc [time,flies]
                    wing_angle_left [time,flies]
                    wing_angle_right [time,flies]
                    wing_angle_sum [time,flies]
                    relative angle [time,flies,flies]
                    (relative orientation) [time,flies,flies]
                    (relative velocities) [time,flies,flies,y/x]
    """
    time = dataset.time
    nearest_frame_time = dataset.nearest_frame

    sampling_rate = 10000
    step = int(sampling_rate / 1000)  # ms - will resample song annotations and tracking data to 1000Hz

    abs_feature_names = [
        'angles',
        'rotational_speed',
        'rotational_acceleration',
        'velocity_magnitude',
        'velocity_x',
        'velocity_y',
        'velocity_forward',
        'velocity_lateral',
        'acceleration_mag',
        'acceleration_x',
        'acceleration_y',
        'acceleration_forward',
        'acceleration_lateral',
        'wing_angle_left',
        'wing_angle_right',
        'wing_angle_sum'
    ]

    rel_feature_names = [
        'distance',
        'relative_angle',
        'relative_orientation',
        'relative_velocity_mag',
        'relative_velocity_forward',
        'relative_velocity_lateral'
    ]

    thoraces = dataset.pose_positions_allo.loc[:, :, 'thorax', :].astype(np.float32)
    heads = dataset.pose_positions_allo.loc[:, :, 'head', :].astype(np.float32)
    wing_left = dataset.pose_positions_allo.loc[:, :, 'left_wing', :].astype(np.float32)
    wing_right = dataset.pose_positions_allo.loc[:, :, 'right_wing', :].astype(np.float32)

    # ABSOLUTE FEATURES #
    angles = angle(thoraces, heads)
    rotational_speed = rot_speed(thoraces, heads)
    rotational_acc = rot_acceleration(thoraces, heads)

    vels = velocity(thoraces, heads)
    chamber_vels = velocity(thoraces, ref='chamber')
    accelerations = acceleration(thoraces, heads)
    chamber_acc = acceleration(thoraces, ref='chamber')

    vels_x = chamber_vels[..., 1]
    vels_y = chamber_vels[..., 0]
    vels_forward = vels[..., 0]
    vels_lateral = vels[..., 1]
    vels_mag = np.linalg.norm(vels, axis=2)
    accs_x = chamber_acc[..., 1]
    accs_y = chamber_acc[..., 0]
    accs_forward = accelerations[..., 0]
    accs_lateral = accelerations[..., 1]
    accs_mag = np.linalg.norm(accelerations, axis=2)

    wing_angle_left = angle(heads, thoraces) - angle(thoraces, wing_left)
    wing_angle_right = -(angle(heads, thoraces) - angle(thoraces, wing_right))
    wing_angle_sum = wing_angle_left + wing_angle_right

    list_absolute = [
        angles,
        rotational_speed,
        rotational_acc,
        vels_mag,
        vels_x,
        vels_y,
        vels_forward,
        vels_lateral,
        accs_mag,
        accs_x,
        accs_y,
        accs_forward,
        accs_lateral,
        wing_angle_left,
        wing_angle_right,
        wing_angle_sum
    ]

    absolute = np.stack(list_absolute, axis=2)

    # RELATIVE FEATURES #
    dis = distance(thoraces)
    rel_angles = relative_angle(thoraces, heads)
    rel_orientation = np.repeat(np.swapaxes(angles.values[:, :, np.newaxis], 1, 2), angles.shape[1], axis=1)-np.repeat(angles.values[:, :, np.newaxis], angles.shape[1], axis=2)
    rel_velocities_forward = np.repeat(np.swapaxes(chamber_vels[..., 0][:, :, np.newaxis], 1, 2), chamber_vels.shape[1], axis=1)-np.repeat(chamber_vels[..., 0][:, :, np.newaxis], chamber_vels.shape[1], axis=2)
    rel_velocities_lateral = np.repeat(np.swapaxes(chamber_vels[..., 1][:, :, np.newaxis], 1, 2), chamber_vels.shape[1], axis=1)-np.repeat(chamber_vels[..., 1][:, :, np.newaxis], chamber_vels.shape[1], axis=2)
    rel_velocities_mag = np.linalg.norm(np.concatenate((rel_velocities_forward[..., np.newaxis], rel_velocities_lateral[..., np.newaxis]), axis=3), axis=3)

    list_relative = [
        dis,
        rel_angles,
        rel_orientation,
        rel_velocities_mag,
        rel_velocities_forward,
        rel_velocities_lateral
    ]

    relative = np.stack(list_relative, axis=3)

    # make DataArray
    abs_features = xr.DataArray(data=absolute,
                                     dims=['time', 'flies', 'absolute_features'],
                                     coords={'time': time,
                                             'absolute_features': abs_feature_names,
                                             'nearest_frame': (('time'), nearest_frame_time)},
                                     attrs={'description': 'coords are "egocentric" - rel. to box',
                                            'sampling_rate_Hz': sampling_rate / step,
                                            'time_units': 'seconds',
                                            'spatial_units': 'pixels'})

    rel_features = xr.DataArray(data=relative,
                                     dims=['time', 'flies', 'relative_flies', 'relative_features'],
                                     coords={'time': time,
                                             'relative_features': rel_feature_names,
                                             'nearest_frame': (('time'), nearest_frame_time)},
                                     attrs={'description': 'coords are "egocentric" - rel. to box',
                                            'sampling_rate_Hz': sampling_rate / step,
                                            'time_units': 'seconds',
                                            'spatial_units': 'pixels'})

    # MAKE ONE DATASET
    feature_dataset = xr.Dataset({'abs_features': abs_features, 'rel_features': rel_features},
                                 attrs={})
    return feature_dataset
