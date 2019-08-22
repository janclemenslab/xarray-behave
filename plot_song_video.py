"""PLOT SONG AND PLAY VIDEO IN SYNC

`python plot_song_video.py datename root`

datename: experiment name (e.g. localhost-20181120_144618)
root: defaults to the current directory - this will work if you're in #Common/chainingmic

Keys (may need to "activate" plot by clicking on song trace first to make this work):
W/A - move left/right
S/D - zoom in/out
K/L - jump to previous/next
M - toggle mark  position of each fly with colored dot
C - toggle crop video around position fly
F - next fly for cropping
X - swap first with second fly for all frames following the current frame 
O - save swap indices to file
SPACE - play/stop video/song trace in
"""
# TODO: capture mouse events to select swap flies in case there are more than 2 flies (fly1 could be focal fly from crop, fly2 determined by mouse click position) 
#       see http://www.pyqtgraph.org/documentation/graphicsscene/mouseclickevent.html
# TODO: use defopt for catching/processing cmdline args
# FIXME: remove horrible global vars and global scope!


import os
import sys
import logging
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import xarray_behave as xb
import numpy as np
from videoreader import VideoReader
from pathlib import Path
import cv2

logging.basicConfig(level=logging.INFO)


def make_colors(nb_flies):
    colors = np.zeros((1, nb_flies, 3), np.uint8)
    colors[0, :, 1:] = 220  # set saturation and brightness to 220
    colors[0, :, 0] = np.arange(0, 180, 180.0 / nb_flies)  # set range of hues
    colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)[0].astype(np.uint8)[..., ::-1]
    return colors


class VideoReaderNP(VideoReader):
    """VideoReader posing as numpy array."""

    def __getitem__(self, index):
        return self.read(index)[1]

    @property
    def dtype(self):
        return np.uint8

    @property
    def shape(self):
        return (self.number_of_frames, *self.frame_shape)

    @property
    def ndim(self):
        return len(self.shape)

    @property
    def size(self):
        return np.product(self.shape)

    def min(self):
        return 0

    def max(self):
        return 255

    def transpose(self, *args):
        return self


class ImageViewVR(pg.ImageView):

    def quickMinMax(self, data):
        """Dummy min/max for numpy videoreader. The Original function tries to read the full video!"""
        return 0, 255


class KeyPressWidget(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)


def keyPressed(evt):
    global t0, span, crop, fly, STOP, dot, cue_index, span
    if evt.key() == QtCore.Qt.Key_Left:  # go one frame back
        t0 -= frame_interval  # TODO get val from fps
    elif evt.key() == QtCore.Qt.Key_Right:  # advance by one frame
        t0 += frame_interval  # TODO get val from fps
    if evt.key() == QtCore.Qt.Key_A:
        t0 -= span / 2
    elif evt.key() == QtCore.Qt.Key_D:
        t0 += span / 2
    elif evt.key() == QtCore.Qt.Key_W:
        span /= 2
    elif evt.key() == QtCore.Qt.Key_S:
        span *= 2
    elif evt.key() == QtCore.Qt.Key_M:
        dot = not dot
    elif evt.key() == QtCore.Qt.Key_K:
        cue_index = max(0, cue_index-1)
        logging.debug(f'cue val at cue_index {cue_index} is {cue_points[cue_index]}')
        # t0 = dataset.time[cue_points[cue_index]].values  # jump to PREV cue point
        t0 = cue_points[cue_index]  # jump to PREV cue point
    elif evt.key() == QtCore.Qt.Key_L:
        cue_index = min(cue_index+1, len(cue_points)-1)
        logging.debug(f'cue val at cue_index {cue_index} is {cue_points[cue_index]}')
        t0 = cue_points[cue_index]  # jump to PREV cue point
    elif evt.key() == QtCore.Qt.Key_X:
        swap_flies(t0)
        logging.info(f'   Swapping flies 1 & 2 at time {t0}.')
    elif evt.key() == QtCore.Qt.Key_O:
        savefilename = Path(root, 'res', datename, f'{datename}_idswaps.txt')
        save_swap_events(savefilename, swap_events)
        logging.info(f'   Saving list of swap indices to {savefilename}.')
    elif evt.key() == QtCore.Qt.Key_C:
        crop = not crop
        logging.info(f'   Cropping = {crop}.')
    elif evt.key() == QtCore.Qt.Key_F:
        fly = (fly+1) % nb_flies
        logging.info(f'   Cropping around fly {fly}.')
    elif evt.key() == QtCore.Qt.Key_Space:
        if STOP:
            logging.info(f'   Starting playback.')
            STOP = False
            play()
        else:
            logging.info(f'   Stopping playback.')
            STOP = True

    span = int(max(500, span))
    logging.debug(t0)
    t0 = int(np.clip(t0, span/2, tmax - span/2))
    logging.debug(t0)
    update(t0, span, crop, fly)
    app.processEvents()


def update(index, span, crop=False, fly=0):
    if hasattr(dataset, 'song'):
        fs_song = dataset.song.attrs['sampling_rate_Hz']
        fs_other = dataset.song_events.attrs['sampling_rate_Hz']
        index_other = int(index * fs_other / fs_song)

        # clear trace plot and update with new trace
        t0 = int(index - span / 2)
        t1 = int(index + span / 2)

        x = dataset.sampletime[t0:t1].values
        y = dataset.song[t0:t1].values
        step = int(np.ceil(len(x) / fs_song / 2))
        slice_view.clear()
        slice_view.plot(x[::step], y[::step])

        # # mark song events in trace
        vibration_indices = np.where(dataset.song_events[int(
            index_other - span / 20):int(index_other + span / 20), 0].values)[0] * int(1 / fs_other * fs_song)
        vibrations = x[vibration_indices]
        for vib in vibrations:
            slice_view.addItem(pg.InfiniteLine(movable=False, angle=90, pos=vib))

        # indicate time point of displayed frame in trace
        slice_view.addItem(pg.InfiniteLine(movable=False, angle=90,
                                           pos=x[int(span / 2)], pen=pg.mkPen(color='r', width=1)))
    else:
        fs_other = dataset.body_positions.attrs['sampling_rate_Hz']
        index_other = int(index * fs_other / 10_000)
    fn = dataset.body_positions.nearest_frame[index_other]
    frame = vr[fn]
    thorax_index = 8
    if dot:
        dot_size = 2
        for dot_fly in range(nb_flies):
            fly_pos = dataset.pose_positions_allo[index_other, dot_fly, thorax_index]
            x_dot = np.clip((fly_pos[0]-dot_size, fly_pos[0]+dot_size), 0, vr.frame_width-1).astype(np.uintp)
            y_dot = np.clip((fly_pos[1]-dot_size, fly_pos[1]+dot_size), 0, vr.frame_height-1).astype(np.uintp)
            frame[slice(*x_dot), slice(*y_dot), :] = fly_colors[dot_fly]  # now crop frame around one of the flies

    if crop:
        # breakpoint()
        box_size = 200
        fly_pos = dataset.pose_positions_allo[index_other, fly, thorax_index]
        # makes sure crop does not exceed frame bounds
        x_range = np.clip((fly_pos[0]-box_size, fly_pos[0]+box_size), 0, vr.frame_width-1).astype(np.uintp)
        y_range = np.clip((fly_pos[1]-box_size, fly_pos[1]+box_size), 0, vr.frame_height-1).astype(np.uintp)
        # now crop frame around the focal fly
        frame = frame[slice(*x_range), slice(*y_range), :]
    image_view.clear()
    image_view.setImage(frame)


def play(rate=100):  # TODO: get rate from ds (video fps attr)
    global t0, span, crop, fly, STOP
    RUN = True
    while RUN:
        t0 += rate
        update(t0, span, crop, fly)
        app.processEvents()
        if STOP:
            RUN = False


def swap_flies(index, fly1=0, fly2=1):
    # use swap_dims to swap flies in all dataarrays in the data set?
    fs_song = dataset.song.attrs['sampling_rate_Hz']
    fs_other = dataset.pose_positions_allo.attrs['sampling_rate_Hz']
    index_other = int(index * fs_other / fs_song)
    if [index_other, fly1, fly2] in swap_events:  # if already in there remove - swapping a second time negates first swap
        swap_events.remove([index_other, fly1, fly2])
    else:
        swap_events.append([index_other, fly1, fly2])
    # THE FOLLOWING DOES NOT UPDATE THE DATASET:
    # for var_name, var in dataset.data_vars.items():
    #     if 'flies' in var.dims:
    #         var.sel(time=slice(index,None), flies=fly1).values, var.sel(time=slice(index,None), flies=fly2).values = var.sel(time=slice(index,None), flies=fly2).values, var.sel(time=slice(index,None), flies=fly1).values
    #         dataset.data_vars[var_name].values = var.values
    dataset.pose_positions_allo.values[index_other:, [
        fly2, fly1], ...] = dataset.pose_positions_allo.values[index_other:, [fly1, fly2], ...]
    dataset.pose_positions.values[index_other:, [
        fly2, fly1], ...] = dataset.pose_positions.values[index_other:, [fly1, fly2], ...]
    dataset.body_positions.values[index_other:, [
        fly2, fly1], ...] = dataset.body_positions.values[index_other:, [fly1, fly2], ...]


def save_swap_events(savefilename, lst):
    np.savetxt(savefilename, lst, fmt='%d', header='index fly1 fly2')


pg.setConfigOptions(useOpenGL=False)   # appears to be faster that way
logging.basicConfig(level=logging.INFO)

datename = 'localhost-20181120_144618'
root = ''
cue_points = []
if len(sys.argv) >= 2:
    datename = sys.argv[1]
if len(sys.argv) >= 3:
    root = sys.argv[2]
if len(sys.argv) >= 4:
    cue_points = eval(sys.argv[3])


# if os.path.exists(datename + '.zarr'):
#     logging.info(f'Loading dataset from {datename}.zarr.')
#     dataset = xb.load(datename + '.zarr')
# else:
logging.info(f'Assembling dataset for {datename}.')
dataset = xb.assemble(datename, root=root)
# logging.info(f'Saving dataset to {datename}.zarr.')
# xb.save(datename + '.zarr', dataset)
logging.info(dataset)
filepath = dataset.attrs['video_filename']
vr = VideoReaderNP(filepath[:-3] + 'avi')

# indices
t0 = 2_800_000  # 1_100_000
span = 100_000
if hasattr(dataset, 'song'):
    tmax = len(dataset.song)
else:
    tmax = len(dataset.body_positions) * 10  # TODO: get factor from dataset

crop = False
dot = True
fly = 0
cue_index = 0
nb_flies = np.max(dataset.flies).values+1
fly_colors = make_colors(nb_flies)
STOP = True
swap_events = []
frame_interval = 100  # TODO: get from dataset

app = pg.QtGui.QApplication([])
win = pg.QtGui.QMainWindow()
win.resize(800, 800)
win.setWindowTitle("psv")

image_view = ImageViewVR(name="img_view")
image_view.setImage(vr)

slice_view = pg.PlotWidget(name="song")

cw = KeyPressWidget()
cw.sigKeyPress.connect(keyPressed)

ly = pg.QtGui.QVBoxLayout()
ly.addWidget(image_view, stretch=4)
ly.addWidget(slice_view, stretch=1)
cw.setLayout(ly)
win.setCentralWidget(cw)
win.show()

image_view.ui.histogram.hide()
image_view.ui.roiBtn.hide()
image_view.ui.menuBtn.hide()

update(t0, span, crop, fly)

# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
