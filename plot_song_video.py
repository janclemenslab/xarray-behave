"""PLOT SONG AND PLAY VIDEO IN SYNC

`python plot_song_video.py datename root`

datename: experiment name (e.g. localhost-20181120_144618)
root: defaults to the current directory - this will work if you're in #Common/chainingmic

Keys (may need to "activate" plot by clicking on song trace first to make this work):
W - move left 
A - move right
S - zoom in 
D - zoom out
C - crop video around position fly
F - next fly for cropping
X - swap first with second fly for all frames following the current frame 
O - 
SPACE - play/stop video/song trace in
"""
import os
import sys
import logging
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import xarray_behave as xb
import numpy as np
from videoreader import VideoReader
from pathlib import Path

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
        """Dummy min/max for numpy videoreader."""
        return 0, 255


class KeyPressWidget(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)


def keyPressed(evt):
    global t0, span, crop, fly, STOP
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
    t0 = int(np.clip(t0, span/2, tmax - span/2))
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
        vibration_indices = np.where(dataset.song_events[int(index_other - span / 20):int(index_other + span / 20), 0].values)[0] * int(1 / fs_other * fs_song)
        vibrations = x[vibration_indices]
        for vib in vibrations:
            slice_view.addItem(pg.InfiniteLine(movable=False, angle=90, pos=vib))
        
        # indicate time point of displayed frame in trace
        slice_view.addItem(pg.InfiniteLine(movable=False, angle=90, pos=x[int(span / 2)], pen=pg.mkPen(color='r', width=1)))
    else:
        fs_other = dataset.body_positions.attrs['sampling_rate_Hz']
        index_other = int(index * fs_other / 10_000)
    fn = dataset.body_positions.nearest_frame[index_other]
    frame = vr[fn]
    if crop:
        box_size = 200
        thorax_index = 8
        fly_pos = dataset.pose_positions_allo[index_other, fly, thorax_index]
        # makes sure crop does not exceed frame bounds
        x_range = np.clip((fly_pos[0]-box_size, fly_pos[0]+box_size), 0, vr.frame_width-1).astype(np.uintp)
        y_range = np.clip((fly_pos[1]-box_size, fly_pos[1]+box_size), 0, vr.frame_height-1).astype(np.uintp)
        frame = frame[slice(*x_range), slice(*y_range), :]  # now crop frame around one of the flies
    image_view.clear()
    image_view.setImage(frame)


def play(rate=100):    
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
    ## THE FOLLOWING DOES NOT UPDATE THE DATASET:
    # for var_name, var in dataset.data_vars.items():
    #     if 'flies' in var.dims:
    #         var.sel(time=slice(index,None), flies=fly1).values, var.sel(time=slice(index,None), flies=fly2).values = var.sel(time=slice(index,None), flies=fly2).values, var.sel(time=slice(index,None), flies=fly1).values
    #         dataset.data_vars[var_name].values = var.values
    dataset.pose_positions_allo.values[index_other:, [fly2, fly1], ...] = dataset.pose_positions_allo.values[index_other:, [fly1, fly2], ...]
    dataset.pose_positions.values[index_other:, [fly2, fly1], ...] = dataset.pose_positions.values[index_other:, [fly1, fly2], ...]
    dataset.body_positions.values[index_other:, [fly2, fly1], ...] = dataset.body_positions.values[index_other:, [fly1, fly2], ...]


def save_swap_events(savefilename, lst):    
    np.savetxt(savefilename, lst, fmt='%d', header='index fly1 fly2')


pg.setConfigOptions(useOpenGL=False)   # appears to be faster that way   
logging.basicConfig(level=logging.INFO)

datename = 'localhost-20181120_144618'
root = ''
if len(sys.argv) >= 2:
    datename = sys.argv[1]
if len(sys.argv) >= 3:
    root = sys.argv[2]

if os.path.exists(datename + '.zarr'):
    logging.info(f'Loading dataset from {datename}.zarr.')
    dataset = xb.load(datename + '.zarr')
else:
    logging.info(f'Assembling dataset for {datename}.')
    dataset = xb.assemble(datename, root=root)
    logging.info(f'Saving dataset to {datename}.zarr.')
    xb.save(datename + '.zarr', dataset)
logging.info(dataset)
filepath = dataset.attrs['video_filename']
vr = VideoReaderNP(filepath[:-3] + 'avi')

# indices
t0 = 2_800_000  #1_100_000
span = 100_000
if hasattr(dataset, 'song'):
    tmax = len(dataset.song)
else:
    tmax = len(dataset.body_positions) * 10  # TODO: get factor from dataset

crop = False
fly = 0
nb_flies = np.max(dataset.flies).values+1
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

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
