"""Calculate metrics from behavioral data."""
import numpy as np
import scipy.signal.windows


def smooth(x, winlen: int = 201):
    """Smooth data along time dimension (axis=0) with Gaussian window.

    Args:
        x ([type]): [description]
        winlen (int, optional): duration of gaussian window in samples. std will be winlen/8. Defaults to 201 (std=25 samples).

    Returns:
        [type]: [description]
    """
    win = scipy.signal.windows.gaussian(winlen, winlen / 8)
    win /= sum(win)
    x = scipy.ndimage.convolve1d(x, win, axis=0, mode='nearest')
    return x


def remove_nan(x):
    """Replace any nan value with nearest valid value."""
    mask = np.isnan(x)
    x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    return x


def remove_multi_nan(x):
    """Replace any nan value with nearest valid value."""
    if x.ndim < 2:
        mask = np.isnan(x)
        x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask),
                            x[~mask])
    elif x.ndim == 2:
        for x_feature in x.T:
            mask = np.isnan(x_feature)
            x_feature[mask] = np.interp(np.flatnonzero(mask),
                                        np.flatnonzero(~mask),
                                        x_feature[~mask])
    else:
        raise ValueError(
            f'Invalid arguments: x can only be one or two dimensional. Has {x.ndim} dimensions.'
        )
    return x


def distance(pos1,
             pos2=None,
             set_self_to_nan: bool = False,
             exclude_cross_terms: bool = False):
    """Compute pairwise euclidean distances.

    Args:
        pos1 ([type]): positions of the thorax. [time, flies1, y/x]
        pos2 ([type], optional): positions of the thorax. [time, flies2, y/x], Defaults to None
        set_self_to_nan (bool, optional): set self-distances (diagonal along 2nd and 3rd dim) to nan. Defaults to False. Will be ignored if pos2 is not None.
        exclude_cross_terms (boolean, optional):, Defaults to False.

    Returns:
        [type]: distances between flies. [time, flies1, flies2]
    """
    if pos2 is not None:
        set_self_to_nan = False

    if pos2 is None:
        pos2 = pos1

    if exclude_cross_terms:
        dis = np.sqrt(np.sum((pos1 - pos2)**2, axis=-1))
    else:
        dis = np.sqrt(
            np.sum((pos1[:, np.newaxis, :, :] - pos2[:, :, np.newaxis, :])**2,
                   axis=-1))

    if set_self_to_nan:
        nb_flies = pos1.shape[1]
        dis[..., range(nb_flies),
            range(nb_flies)] = np.nan  # set diag to np.nan
    return dis


def yx2fwdlat(pos1, pos2, dyx):
    """[summary]

    Args:
        pos1 ([type]): [description]
        pos2 ([type]): [description]
        dyx ([type]): [description]

    Returns:
        [type]: [description]
    """
    mags = np.linalg.norm(dyx, axis=2)  # acceleration magnitude
    angle_diff = angle(dyx, degrees=False) - angle(
        pos1, pos2,
        degrees=False)  # angle difference between dyx and orientation
    fwdlat = np.empty_like(
        dyx)  # acceleration components with reference to self orientation
    fwdlat[:, :, 0] = mags * np.cos(angle_diff)  # forward
    fwdlat[:, :, 1] = -mags * np.sin(angle_diff)  # lateral
    return fwdlat


def derivative(x, dt: float = 1, degree: int = 1, axis: int = 0):
    """Calculate derivative of x with respect to axis.

    Args:
        x ([type]): Data. At last 1D.
        dt (float, optional): Timestep. Defaults to 1.
        degree (int, optional): Number of time the gradient is taken. Defaults to 1 (velocity). degree=2 return acceleration.
        axis (int, optional): Axis along which to take the gradient. Defaults to 0.

    Returns:
        [type]: [description]
    """
    for _ in range(degree):
        x = np.gradient(x, dt, axis=axis)
    return x


def velocity(pos1,
             pos2: np.array = None,
             timestep: float = 1,
             ref: str = 'self'):
    """
    arg:
        pos1: position of vector's base, center of agent. [time, agent, y/x]
        pos2: position of vector's head, head of agent. [time, agent, y/x]
        timestep: time difference between data points (default = 1)
        ref: type of output velocity. 'self' for forward/lateral velocity components, 'chamber' for y/x components.
    returns:
        vels: change of variable with respect to time. [time,agent,y/x]
    """
    vels = derivative(pos1, timestep, degree=1)
    if ref == 'self':
        vels = yx2fwdlat(pos1, pos2, vels)
    return vels


def acceleration(pos1,
                 pos2: np.array = None,
                 timestep: float = 1,
                 ref: str = 'self'):
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
    accs = derivative(pos1, timestep, degree=2)
    if ref == 'self':
        accs = yx2fwdlat(pos1, pos2, accs)
    return accs


def angle(pos1, pos2=None, degrees: bool = True, unwrap: bool = False):
    """Compute angle.

    Args:
        pos1 ([type]): position of vector's base, center of agent. [time, agent, y/x]
        pos2 ([type], optional): position of vector's head, head of agent. [time, agent, y/x]. If provided will compute angle between pos1 and pos2. Defaults to None.
        degrees (bool, optional): [description]. Defaults to True.
        unwrap (bool, optional): [description]. Defaults to False.

    Returns:
        [type]: orientation of flies with respect to chamber. 0 degrees when flies look towards positive x axis, 90 degrees when vertical and looking upwards. [time, flies]
    """
    if pos2 is None:
        ang = np.arctan2(pos1[..., 0], pos1[..., 1])
    else:
        ang = np.arctan2(pos2[..., 0] - pos1[..., 0],
                         pos2[..., 1] - pos1[..., 1])

    if unwrap:
        ang = np.unwrap(ang)

    if degrees:
        ang = ang * 180.0 / np.pi
    return ang


def relative_angle(pos1, pos2):
    """ Angle between agents. An element (k,i,j) from the output is the angle at kth sample between ith (reference head) and jth (target base).
    arg:
        pos1: positions of the thoraces for all flies. [time, flies, y/x]
        pos2: positions of the heads for all flies. [time, flies, y/x]
    returns:
        rel_angles: orientation of flies with respect to chamber. [time, flies, flies]
    """
    d0 = pos2 - pos1
    d1 = pos1[:,
              np.newaxis, :, :] - pos2[:, :, np.
                                       newaxis, :]  # all pairwise "distances"

    dot = d0[:, :, np.newaxis, 1] * d1[:, :, :, 1] + d0[:, :, np.newaxis,
                                                        0] * d1[:, :, :, 0]
    det = d0[:, :, np.newaxis, 1] * d1[:, :, :, 0] - d0[:, :, np.newaxis,
                                                        0] * d1[:, :, :, 1]

    rel_angles = np.arctan2(det, dot)

    return rel_angles * 180.0 / np.pi


def rot_speed(pos1, pos2, timestep: float = 1):
    """
    arg:
        pos1: position of vector's base, center of agent. [time, flies, y/x]
        pos2: position of vector's head, head of agent. [time, flies, y/x]
    returns:
        rot_speed: rotational speed. [time, flies]
    """
    unwrapped_angles = angle(pos1, pos2, unwrap=True)
    rot_speed = derivative(unwrapped_angles, timestep, degree=1)
    return rot_speed


def rot_acceleration(pos1, pos2, timestep: float = 1):
    """[summary]

    Args:
        pos1 ([type]): position of vector's base, center of agent. [time, flies, y/x]
        pos2 ([type]): position of vector's head, head of agent. [time, flies, y/x]
        timestep (float, optional): [description]. Defaults to 1.

    Returns:
        [type]: rotational acceleration. [time, flies]
    """
    unwrapped_angles = angle(pos1, pos2, unwrap=True)
    rot_accs = derivative(unwrapped_angles, timestep, degree=2)
    return rot_accs


def project_velocity(vx, vy, radians):
    """Rotate vectors around the origin (0, 0).

    Args:
        vx ([type]): vector x component.
        vy ([type]): vector y component.
        radians ([type]): angles of rotation in radians.

    Returns:
        rot_vx ([type]): vector lateral component. Left negative.
        rot_vy ([type]): vector forward component. Backwards negative
    """

    if radians.shape != vx.shape:
        radians = np.repeat(radians[..., np.newaxis], radians.shape[1], axis=2)

    rot_vx = -vx * np.cos(radians) + vy * np.sin(radians)
    rot_vy = vx * np.sin(radians) + vy * np.cos(radians)

    return rot_vx, rot_vy


def internal_angle(A: np.array,
                   B: np.array,
                   C: np.array,
                   deg: bool = True,
                   array_logic: str = 'tfc'):
    """Calculates internal angle (∠ABC) between three points (aka, angle between lines AB and BC).
    If A,B,C are lists or arrays, calculation happens element-wise.

    Args:
        A, B, C (np.array): position of points between which the angle is calculated.
        deg (bool): Return angle in degrees if True, radians if False (default).

    Returns:
        angles (np.array): internal angle between lines AB and BC.
    """

    v1s = B - A  # vector from point B to A
    v2s = B - C  # vector from point B to C

    # reshape vector arrays to be [time, coordinates, ...]
    if A.ndim == 3:
        if array_logic == 'tfc':
            v1s = np.swapaxes(v1s, 1, 2)
            v2s = np.swapaxes(v2s, 1, 2)
        elif array_logic == 'ftc':
            v1s = np.swapaxes(v1s, 1, 2)
            v1s = np.swapaxes(v1s, 0, 2)
            v2s = np.swapaxes(v2s, 1, 2)
            v2s = np.swapaxes(v2s, 0, 2)
        elif array_logic == 'fct':
            v1s = np.swapaxes(v1s, 0, 2)
            v2s = np.swapaxes(v2s, 0, 2)
    elif A.ndim > 3:
        print('Result might not be correct, only tested for arrays with 2 or 3 dimensions. Contact Adrian for help, if required.')

    dot_v1_v2 = np.einsum('ij...,ij...->i...', v1s, v2s)
    dot_v1_v1 = np.einsum('ij...,ij...->i...', v1s, v1s)
    dot_v2_v2 = np.einsum('ij...,ij...->i...', v2s, v2s)

    # small values for dot products which happen to be zero (shouldn't happen)
    dot_v1_v1[dot_v1_v1 == 0] = 0.0001
    dot_v2_v2[dot_v2_v2 ==0] = 0.0001

    angles = np.arccos(dot_v1_v2 / np.sqrt(dot_v1_v1 * dot_v2_v2))

    if deg:
        angles *= 180 / np.pi

    return angles