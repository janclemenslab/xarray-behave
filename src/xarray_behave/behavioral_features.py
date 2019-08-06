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

    for i,j in itertools.combinations(range(nagents), r=2):
        diff_ij = np.squeeze(np.diff(positions[:,[i,j],:],axis=1))
        distance = np.sqrt(np.einsum('ij,ij->i', diff_ij, diff_ij))
        dis[:,i,j] = distance
        dis[:,j,i] = distance

    return dis


def velocity(pos1,pos2: np.array=None,timestep: float=1,ref: str='self'):
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
        angle_diff = np.arctan2(vels_yx[...,0], vels_yx[...,1]) - np.arctan2(pos2[...,0] - pos1[...,0], pos2[...,1] - pos1[...,1])    # angle difference between velocity vector and orientation vector
        vels = np.empty_like(vels_yx)    # velocity components with reference to self orientation
        vels[:,:,0] = vels_mags*np.cos(angle_diff)    # forward
        vels[:,:,1] = -vels_mags*np.sin(angle_diff)    # lateral
    elif ref == 'chamber':
        vels = vels_yx

    return vels


def acceleration(pos1,pos2: np.array=None,timestep: float=1,ref: str='self'):
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
        angle_diff = np.arctan2(accs_yx[...,0], accs_yx[...,1]) - np.arctan2(pos2[...,0] - pos1[...,0], pos2[...,1] - pos1[...,1])    # angle difference between acceleration vector and orientation vector
        accs = np.empty_like(accs_yx)    # acceleration components with reference to self orientation
        accs[:,:,0] = accs_mags*np.cos(angle_diff)    # forward
        accs[:,:,1] = -accs_mags*np.sin(angle_diff)    # lateral
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
    return np.arctan2(pos2[...,0] - pos1[...,0], pos2[...,1] - pos1[...,1]) * 180 / np.pi


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

    for i,j in itertools.combinations(range(nagents), r=2):
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


def rot_speed(pos1, pos2, timestep: float=1):
    """
    arg:
        pos1: position of vector's base, center of agent. [time, flies, y/x]
        pos2: position of vector's head, head of agent. [time, flies, y/x]
    returns:
        rot_speed: rotational speed. [time, flies]
    """

    # orientations
    angles = np.arctan2(pos2[...,0] - pos1[...,0], pos2[...,1] - pos1[...,1])

    # unwrap orientations
    unwrapped_angles = np.unwrap(angles,axis=0)*180/np.pi

    # rotational speed
    rot_speed = np.gradient(unwrapped_angles, timestep, axis=0)

    return rot_speed


def rot_acceleration(pos1, pos2, timestep: float=1):
    """
    arg:
        pos1: position of vector's base, center of agent. [time, flies, y/x]
        pos2: position of vector's head, head of agent. [time, flies, y/x]
    returns:
        rot_accs: rotational acceleration. [time, flies]
    """

    # orientations
    angles = np.arctan2(pos2[...,0] - pos1[...,0], pos2[...,1] - pos1[...,1])

    # unwrap orientations
    unwrapped_angles = np.unwrap(angles,axis=0)*180/np.pi

    # rotational speed
    rot_accs = np.gradient(np.gradient(unwrapped_angles, timestep, axis=0),timestep, axis=0)

    return rot_accs


def assemble_metrics(dataset):
    """ arg:
            dataset: xarray.Dataset from datastructure module, which takes experiment name as input.
        return:
            feature_dataset: xarray.Dataset with collection of calculated features.
                features:
                    angles [time,flies,y/x]
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

    first_sample = 0
    last_sample = time.shape[0]
    sampling_rate = 10_000
    step = int(sampling_rate / 1000)  # ms - will resample song annotations and tracking data to 1000Hz

    abs_feature_names = ['angles','forward_velocity', 'lateral_velocity']
                    #, 'chamber_velocity','rotational_speed', 'forward_acceleration','lateral_acceleration','chamber_acceleration',
                    # 'rotational_acceleration','wing_angle_left','wing_angle_right','wing_angle_sum']

    rel_feature_names = ['distance', 'relative_angles']
                    # 'relative_velocities', 'relative_orientation']

    HEAD = 0
    THORAX = 8 #all other positions are relative to thorax
    WING_L = 9
    WING_R = 10
    TAIL = 11

    pose_positions_allo=dataset.pose_positions_allo

    thoraces = pose_positions_allo[...,THORAX,:].values
    heads = pose_positions_allo[...,HEAD,:].values
    wing_left = pose_positions_allo[...,WING_L,:].values
    wing_right = pose_positions_allo[...,WING_R,:].values


    # ABSOLUTE FEATURES #
    angles = angle(thoraces,heads)
    vels = velocity(thoraces,heads)
    chamber_vels = velocity(thoraces,ref='chamber')
    rotational_speed = rot_speed(thoraces,heads)
    accelerations = acceleration(thoraces,heads)
    chamber_acc = acceleration(thoraces,ref='chamber')
    rotational_acc = rot_acceleration(thoraces,heads)
    wing_angle_left = angle(heads,thoraces) - angle(thoraces, wing_left)
    wing_angle_right = -(angle(heads,thoraces) - angle(thoraces, wing_right))
    wing_angle_sum = wing_angle_left + wing_angle_right

    absolute = np.concatenate((angles[...,np.newaxis], vels), axis=2) # , chamber_vels[...,np.newaxis], rotational_speed[...,np.newaxis],
                               # accelerations, chamber_acc[...,np.newaxis], rotational_acc[...,np.newaxis],
                               # wing_angle_left[...,np.newaxis], wing_angle_right[...,np.newaxis], wing_angle_sum[...,np.newaxis]), axis=2)

    # RELATIVE FEATURES #
    dis = distance(thoraces)
    rel_angles = relative_angle(thoraces, heads)
    rel_orientation = np.repeat(np.swapaxes(angles[:,:,np.newaxis],1,2),angles.shape[1],axis=1)-np.repeat(angles[:,:,np.newaxis],angles.shape[1],axis=2)
    # rel_velocities =

    relative = np.concatenate((dis[...,np.newaxis], rel_angles[...,np.newaxis]), axis=3)
                                # , rel_velocities, rel_orientation), axis=3)


    # make DataArray
    absolute_features = xr.DataArray(data=absolute,
                         dims=['time', 'flies', 'movement_features'],
                         coords={'time': time,
                                 'movement_features': abs_feature_names,
                                 'nearest_frame': (('time'), nearest_frame_time)},
                         attrs={'description': 'coords are "egocentric" - rel. to box',
                                'sampling_rate_Hz': sampling_rate / step,
                                'time_units': 'seconds',
                                'spatial_units': 'pixels'})

    relative_features = xr.DataArray(data=relative,
                         dims=['time', 'flies', 'relative_flies', 'movement_features'],
                         coords={'time': time,
                                 'movement_features': rel_feature_names,
                                 'nearest_frame': (('time'), nearest_frame_time)},
                         attrs={'description': 'coords are "egocentric" - rel. to box',
                                'sampling_rate_Hz': sampling_rate / step,
                                'time_units': 'seconds',
                                'spatial_units': 'pixels'})


    # MAKE ONE DATASET
    feature_dataset = xr.Dataset({'absolute_features': absolute_features, 'relative_features': relative_features},
                         attrs={})
    return feature_dataset
