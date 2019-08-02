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
    for i,j in itertools.product(range(nagents),range(nagents)):
        diff_ij = np.squeeze(np.diff(positions[:,[i,j],:],axis=1))
        distance = np.sqrt(np.einsum('ij,ij->i', diff_ij, diff_ij))
        dis[:,i,j] = distance

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
    pass


def relative_angle(heads_pos, tail_pos):
    """ arg:
            heads_pos: positions of the heads for all flies. [time, flies, y/x]
            tails_pos: positions of the tails for all flies. [time, flies, y/x]

        note: vector from pos1 to pos2.

        returns:
            rel_angles: orientation of flies with respect to chamber. [time, flies]
    """
    pass


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


def assemble_metrics(xarray):
    """ arg:
            xarray: data structure from datastructure module, which takes experiment name as input.
        return:
            feature_xarray: data structure with collection of calculated features.

        notes: wing angle left and right and sum; relative velocities [time,flies,flies,y/x]; relative orientation
    """
    pass
