"""Load FicTrac ball tracking data

should return:
    x: pd.DataFrame[frames, (variables)]
"""
import numpy as np
import pandas as pd
from .. import io
from typing import Optional
import logging
import xarray as xr

class BallTracks():

    def make(self, filename: Optional[str] = None):
        if filename is None:
            filename = self.path

        logging.info(f'Loading ball tracker:')
        balltracks, first_balltracked_frame, last_balltracked_frame = self.load(filename)

        xr_balltracks = xr.DataArray(data=balltracks.values,
                                    dims=['frame_number_ball', 'data_ball'],
                                    coords={'frame_number_ball': balltracks['frame counter (starts at 1)']-1,
                                            'data_ball': list(balltracks.columns)},
                                    attrs={'loader': self.NAME,
                                           'kind': self.KIND,
                                           'path': filename,})
        return xr_balltracks


@io.register_provider
class FicTrac_balltracks(io.BaseProvider, BallTracks):

    KIND = 'balltracks'
    NAME = 'fictrac csv'
    SUFFIXES = ['_ball.dat']

    def load(self, filename: Optional[str] = None):
        """Load tracker data"""
        if filename is None:
            filename = self.path

        # from https://github.com/rjdmoore/fictrac/blob/master/doc/data_header.txt
        #
        #  COL     PARAMETER                       DESCRIPTION
        #     1       frame counter                   Corresponding video frame (starts at #1).
        #     2-4     delta rotation vector (cam)     Change in orientation since last frame,
        #                                             represented as rotation angle/axis (radians)
        #                                             in camera coordinates (x right, y down, z
        #                                             forward).
        #     5       delta rotation error score      Error score associated with rotation
        #                                             estimate.
        #     6-8     delta rotation vector (lab)     Change in orientation since last frame,
        #                                             represented as rotation angle/axis (radians)
        #                                             in laboratory coordinates (see
        #                                             *configImg.jpg).
        #     9-11    absolute rotation vector (cam)  Absolute orientation of the sphere
        #                                             represented as rotation angle/axis (radians)
        #                                             in camera coordinates.
        #     12-14   absolute rotation vector (lab)  Absolute orientation of the sphere
        #                                             represented as rotation angle/axis (radians)
        #                                             in laboratory coordinates.
        #     15-16   integrated x/y position (lab)   Integrated x/y position (radians) in
        #                                             laboratory coordinates. Scale by sphere
        #                                             radius for true position.
        #     17      integrated animal heading (lab) Integrated heading orientation (radians) of
        #                                             the animal in laboratory coordinates. This
        #                                             is the direction the animal is facing.
        #     18      animal movement direction (lab) Instantaneous running direction (radians) of
        #                                             the animal in laboratory coordinates. This is
        #                                             the direction the animal is moving in the lab
        #                                             frame (add to animal heading to get direction
        #                                             in world).
        #     19      animal movement speed           Instantaneous running speed (radians/frame)
        #                                             of the animal. Scale by sphere radius for
        #                                             true speed.
        #     20-21   integrated forward/side motion  Integrated x/y position (radians) of the
        #                                             sphere in laboratory coordinates neglecting
        #                                             heading. Equivalent to the output from two
        #                                             optic mice.
        #     22      timestamp                       Either position in video file (ms) or frame
        #                                             capture time (ms since epoch).
        #     23      sequence counter                Position in current frame sequence. Usually
        #                                             corresponds directly to frame counter, but
        #                                             can reset to 1 if tracking is reset.
        #     24      delta timestamp                 Time (ms) since last frame.
        #     25      alt. timestamp                  Frame capture time (ms since midnight).
        column_names = ['frame counter (starts at 1)',
                        'delta rotation vector (cam) x right [rad]',
                        'delta rotation vector (cam) y down [rad]',
                        'delta rotation vector (cam) z forward [rad]',
                        'delta rotation error score',
                        'delta rotation vector (lab) x right [rad]',
                        'delta rotation vector (lab) y down [rad]',
                        'delta rotation vector (lab) z forward [rad]',
                        'absolute rotation vector (cam) x right [rad]',
                        'absolute rotation vector (cam) y down [rad]',
                        'absolute rotation vector (cam) z forward [rad]',
                        'absolute rotation vector (lab) x right [rad]',
                        'absolute rotation vector (lab) y down [rad]',
                        'absolute rotation vector (lab) z forward [rad]',
                        'integrated x position [sphere radius]',
                        'integrated y position [sphere radius]',
                        'integrated animal heading (lab) [rad]',
                        'animal movement direction (lab) [rad]',
                        'animal movement speed [rad/frame/sphere radius]',
                        'integrated forward motion [rad]',
                        'integrated side motion [rad]',
                        'timestamp [ms]',
                        'sequence counter [frame relative to last reset]',
                        'delta timestamp [ms since last frame]',
                        # 'alt. timestamp [ms since midnight ]'
                        ]
        balltracks = pd.read_csv(filename)
        balltracks.columns = column_names
        first_balltracked_frame = balltracks['frame counter (starts at 1)'].values[0]
        last_balltracked_frame = balltracks['frame counter (starts at 1)'].values[-1]
        return balltracks, first_balltracked_frame, last_balltracked_frame
