import numpy as np
import itertools


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


def velocity(pos1, pos2):
    """ arg:
            pos1: positions of the body reference (thorax). Will be considered as the center. [time, flies, y/x]
            pos2: positions of the other body reference for orientation (head). [time, flies, y/x]

        note: vector from pos1 to pos2.

        returns:
            vels: velocity of flies. [time, flies, forward/lateral]
    """
    x = 1
    y = 0

    dir_vector_y = pos2[...,y] - pos1[...,y]
    dir_vector_x = pos2[...,x] - pos1[...,x]
    angle = np.squeeze(np.arctan2(dir_vector_y, dir_vector_x) * 180 / np.pi)

    mov_vector_x = np.gradient(pos1[...,x],axis=0) # gradient substracts past from present
    mov_vector_y = np.gradient(pos1[...,y],axis=0)
    mov_angle = np.arctan2(mov_vector_y, mov_vector_x) * 180 / np.pi

    theta = mov_angle-angle

    distance_of_movement = np.empty((pos1.shape[:2]))
    for iagent in range(pos1.shape[1]):
        present_min_past = np.gradient(pos1[:,iagent,:])[0]
        distance_of_movement[:,iagent] = np.sqrt(np.einsum('ij,ij->i', present_min_past, present_min_past))

    forward_vel=distance_of_movement*np.cos(theta*np.pi/180)
    lateral_vel=distance_of_movement*np.sin(theta*np.pi/180)
    vels = np.concatenate((forward_vel[...,np.newaxis],lateral_vel[...,np.newaxis]),axis=2)

    return vels


def acceleration(pos1, pos2):
    """ arg:
            pos1: positions of the body reference (thorax). Will be considered as the center. [time, flies, y/x]
            pos2: positions of the other body reference for orientation (head). [time, flies, y/x]

        note: vector from pos1 to pos2.

        returns:
            accs: acceleration of flies. [time, flies, forward/lateral]
    """
    pass


def chamber_acceleration(pos):
    """ arg:
            pos: positions of the body reference (thorax). Will be considered as the center. [time, flies, y/x]

        returns:
            accs: acceleration of flies relative to chamber axis. [time, flies, y/x]
    """
    pass


def chamber_velocity(pos):
    """ arg:
            pos: positions of the body reference (thorax). Will be considered as the center. [time, flies, y/x]

        returns:
            vels: velocity of flies relative to chamber axis. [time, flies, y/x]
    """
    pass


def angle(pos1, pos2):
    """ arg:
            pos1: positions of the body reference (thorax). Will be considered as the center. [time, flies, y/x]
            pos2: positions of the other body reference for orientation (head). [time, flies, y/x]

        note: vector from pos1 to pos2.

        returns:
            angles: orientation of flies with respect to chamber. [time, flies]
    """
    x = 1
    y = 0

    dir_vector_y = pos2[...,y] - pos1[...,y]
    dir_vector_x = pos2[...,x] - pos1[...,x]
    angles = np.squeeze(np.arctan2(dir_vector_y, dir_vector_x) * 180 / np.pi)

    return angles


def relative_angle(pos1, pos2):
    """ arg:
            pos1: positions of the thoraxes for all flies. [time, flies, y/x]
            pos2: positions of the heads for all flies. [time, flies, y/x]

        note: vector from pos1 to pos2.

        returns:
            rel_angles: orientation of flies with respect to chamber. [time, flies]
    """
    x = 1
    y = 0
    nagents = pos1.shape[1]

    dir_vector_y = pos2[...,y] - pos1[...,y]
    dir_vector_x = pos2[...,x] - pos1[...,x]
    angle = np.squeeze(np.arctan2(dir_vector_y, dir_vector_x) * 180 / np.pi)

    rel_angles = np.empty((pos1.shape[0], nagents, nagents))
    for i,j in itertools.product(range(nagents),range(nagents)):
        if i != j:
            axis_x_between=np.diff(pos1[:,[i,j],x])
            axis_y_between=np.diff(pos1[:,[i,j],y])
            angle_between = np.squeeze(np.arctan2(axis_y_between, axis_x_between) * 180 / np.pi)
            uncontinuous_angle = angle_between - angle[:,i]
            rel_angles[:,i,j] = uncontinuous_angle - ((uncontinuous_angle + 180) // 360) * 360 # make angles continious between -180 and 180
    return rel_angles


def rot_speed(pos1, pos2):
    """ arg:
            pos1: positions of the body reference (thorax). Will be considered as the center. [time, flies, y/x]
            pos2: positions of the other body reference for orientation (head). [time, flies, y/x]

        note: vector from pos1 to pos2.

        returns:
            rot_speed: rotational speed of flies. [time, flies]
    """
    pass


def rot_acceleration(pos1, pos2):
    """ arg:
            pos1: positions of the body reference (thorax). Will be considered as the center. [time, flies, y/x]
            pos2: positions of the other body reference for orientation (head). [time, flies, y/x]

        note: vector from pos1 to pos2.

        returns:
            rot_accs: rotational acceleration of flies. [time, flies]
    """
    pass


def assemble_metrics(dataset):
    """ arg:
            dataset: xarray.Dataset from datastructure module, which takes experiment name as input.
        return:
            feature_dataset: xarray.Dataset with collection of calculated features.

        notes: wing angle left and right and sum; relative velocities [time,flies,flies,y/x]; relative orientation
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
    chamber_velocity = chamber_velocity(thoraces)
    rotational_speed = rot_speed(thoraces,heads)
    accelerations = acceleration(thoraces,heads)
    chamber_acceleration = chamber_acceleration(thoraces)
    rotational_acceleration = rot_acceleration(thoraces,heads)
    wing_angle_left = angles - angle(thoraces, wing_left)
    wing_angle_right = angles - angle(thoraces, wing_right)
    wing_angle_sum = np.abs(wing_angle_left) + np.abs(wing_angle_right)

    absolute = np.concatenate((angles[...,np.newaxis], vels), axis=2) # , chamber_velocity[...,np.newaxis], rotational_speed[...,np.newaxis],
                               # accelerations, chamber_acceleration[...,np.newaxis], rotational_acceleration[...,np.newaxis],
                               # wing_angle_left[...,np.newaxis], wing_angle_right[...,np.newaxis], wing_angle_sum[...,np.newaxis]), axis=2)


    # RELATIVE FEATURES #
    dis = distance(thoraces)
    rel_angles = relative_angle(thoraces, heads)
    # rel_velocities =
    # rel_orientation =

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
