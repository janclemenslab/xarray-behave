import numpy as np


def distance(positions):
    """ arg:
            positions: positions of the thorax. [time, flies, y/x]
        returns:
            dists: distances between flies. [time, flies, flies]
    """
    pass


def velocity(pos1, pos2):
    """ arg:
            pos1: positions of the body reference (thorax). Will be considered as the center. [time, flies, y/x]
            pos2: positions of the other body reference for orientation (head). [time, flies, y/x]

        note: vector from pos1 to pos2.

        returns:
            vels: velocity of flies. [time, flies, forward/lateral]
    """
    pass


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
