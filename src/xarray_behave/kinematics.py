"""
These metrics rel to each fly's center/thorax (could extend to include all body parts)
forward vel/acc m/f/fm
lateral spd/acc m/f/fm
angle fm/mf
distance fm
rotational spd/acc m/f (Adam does not use rot acc and fm rot spd/acc)

All relative (pairwise) metrics should be computed for all flies in the movie as a matrix

song:
pfast, pslow, sine, vibration, aggression

Optionally delay-embed

general interface:
low-level functions:
xr.DataArray([time, fly, acc/vel]) = get_abs_kinematic([time, fly, (x,y)])
xr.DataArray([time, fly1, fly2, distance]) = get_rel_kinematic([time, fly, (x,y)])

"""

import xarray as xr
import scipy.signal
import numpy as np
from numpy.lib.stride_tricks import as_strided as ast



def chunk_data(data, window_size, overlap_size=0, flatten_inside_window=True):
    """Chunk data with overlap as a view.
    
    Args:
        data (numpy array): [description]
        window_size (int?): [description]
        overlap_size (int, optional): [description]. Defaults to 0.
        flatten_inside_window (bool, optional): [description]. Defaults to True.
    
    Returns:
        [type]: [description]
    """
    assert data.ndim == 1 or data.ndim == 2
    if data.ndim == 1:
        data = data.reshape((-1, 1))

    # get the number of overlapping windows that fit into the data
    num_windows = (data.shape[0] -
                   window_size) // (window_size - overlap_size) + 1
    overhang = data.shape[0] - \
        (num_windows*window_size - (num_windows-1)*overlap_size)

    # if there's overhang, need an extra window and a zero pad on the data
    # (numpy 1.7 has a nice pad function I'm not using here)
    # TODO: use np.pad here!!
    if overhang != 0:
        num_windows += 1
        newdata = np.zeros(
            (num_windows*window_size - (num_windows-1)*overlap_size, data.shape[1]))
        newdata[:data.shape[0]] = data
        data = newdata

    sz = data.dtype.itemsize
    ret = ast(
        data,
        shape=(num_windows, window_size*data.shape[1]),
        strides=((window_size-overlap_size)*data.shape[1]*sz, sz)
    )

    if flatten_inside_window:
        return ret
    else:
        return ret.reshape((num_windows, -1, data.shape[1]))


def fix_nan(x):
    """Replace all nan and inf values
    
    Args:
        x (np.ndarray): 1D array
    
    Returns:
        np.ndarray: x with all nan and inf values replaced with nearest numeric value
    """
    mask = ~np.isfinite(x)
    x[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), x[~mask])
    return x


def smooth(x, winlen: int = 201):
    """Smooth 1D array with Gaussian window.
    
    Args:
        x ([type]): [description]
        winlen (int, optional): [description]. Defaults to 201.
    
    Returns:
        [type]: [description]
    """
    win = scipy.signal.windows.gaussian(winlen, winlen/8)
    win /= sum(win)
    x = scipy.signal.convolve(np.pad(x,(winlen,winlen), mode='edge'), win, mode='same')[winlen:-winlen]
    return x


def get_abs_kinematics(thx: xr.DataArray, fix: bool = True, smooth: bool = True):
    """Get per-fly kinematics.
    
    Args:
        thx (xarray.DataArray): [time, flies, x/y]
        fix (bool, optional): fix nan/inf. Defaults to True.
        smooth (bool, optional): smooth positions. Defaults to True.

    Returns:
        [xarray.DataArray]: [time, flies, speed/acceleration]
    """
    speed = np.sqrt(np.sum(np.gradient(thx, thx.time, axis=0)**2, axis=-1))
    acceleration = np.gradient(speed, thx.time, axis=0)

    abs_metrics = xr.DataArray(np.stack([speed, acceleration], -1),
                dims=['time', 'flies', 'metric'],
                coords={'time': thx.time, 
                        'nearest_frame': (('time'), thx.nearest_frame),
                        'metric': ['speed', 'acceleration']},
                attrs={'speed_units': 'pixels/second',
                    'acceleration_units': 'pixels/second^2'})
    return abs_metrics


def get_rel_kinematics(thx):
    """Get rel metrics (that involve pairs of flies).
    
    Args:
        thx (xarray.DataArray): [time, flies, x/y]
        
    Returns:
        [xarray.DataArray]: [time, flies, speed/acceleration]
    """
    # TODO: take care of symm (e.g. distance) vs. assymmetrical (angle) metrics
    import itertools
    nb_flies = np.max(thx.flies).values+1
    distances = []
    speeds_rel = []
    pairs = []

    speed = np.sqrt(np.sum(np.gradient(thx, thx.time, axis=0)**2, axis=-1))
    
    for cnt, (fly1, fly2) in enumerate(itertools.product(range(nb_flies), range(nb_flies))):
        if fly2<=fly1:
            pass
        else:
            distances.append(np.sqrt(np.sum(np.diff(thx[:,[fly1, fly2],:], axis=1)**2, axis=-1)))
            speeds_rel.append(np.diff(speed[:,[fly1, fly2]], axis=1))
            pairs.append((fly1, fly2))

    pairs = np.stack(pairs, -1)
    distances = np.concatenate(distances, axis=-1)
    speeds_rel = np.concatenate(speeds_rel, axis=-1)

    rel_metrics = xr.DataArray(np.stack((distances, speeds_rel), axis=2),
                dims=['time', 'fly_pairs', 'metric'],
                coords={'time': thx.time,
                        'nearest_frame': (('time'), thx.nearest_frame),
                        'flies1': (('fly_pairs'), pairs[0,:]),
                        'flies2': (('fly_pairs'), pairs[1,:]),
                    'metric': ['distance', 'speed_rel']})
    return rel_metrics