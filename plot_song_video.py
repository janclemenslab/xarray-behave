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
SPACE - play/stop video/song trace in
"""
import os
import sys
import logging
import time
from pyqtgraph.Qt import QtGui, QtCore
import pyqtgraph as pg
import xarray_behave as dst
import numpy as np
from videoreader import VideoReader


class KeyPressWidget(pg.GraphicsLayoutWidget):
    sigKeyPress = QtCore.pyqtSignal(object)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def keyPressEvent(self, ev):
        self.scene().keyPressEvent(ev)
        self.sigKeyPress.emit(ev)


def keyPressed(evt):
    global t0, span, crop, fly, STOP
    if evt.key() == QtCore.Qt.Key_A:
        t0 -= span/2
    elif evt.key() == QtCore.Qt.Key_D:
        t0 += span/2
    elif evt.key() == QtCore.Qt.Key_W:
        span /= 2
    elif evt.key() == QtCore.Qt.Key_S:
        span *= 2
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
    t0 = int(np.clip(t0, 0, tmax))
    update(t0, span, crop, fly)
    app.processEvents()


def update(index, span, crop=False, fly=0):
    if hasattr(dataset, 'song'):
        fs_song = dataset.song.attrs['sampling_rate_Hz']
        fs_other = dataset.song_events.attrs['sampling_rate_Hz']
        index_other = int(index * fs_other / fs_song)

        # clear trace plot and update with new trace
        t0 = int(index - span / 2)
        t0 = max(0, t0)
        t1 = int(index + span / 2)
        t1 = max(t0 + span, t1)
        
        x = dataset.sampletime[t0:t1].values
        y = dataset.song[t0:t1].values
        step = int(np.ceil(len(x) / fs_song / 2))
        step = max(step, 1)  # make sure step is at least 1
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


def play(rate=100):    
    global t0, span, crop, fly, STOP
    RUN = True
    cnt = 0
    start_time = 0
    while RUN:
        cnt += 1
        if cnt % 10 == 0:
            logging.info(f'{time.time()-start_time},{len(slice_view.items())}')
            start_time = time.time()
        t0 += rate
        update(t0, span, crop, fly)
        app.processEvents()
        if STOP:
            RUN = False
        

pg.setConfigOptions(useOpenGL=False)   # appears to be faster that way   
logging.basicConfig(level=logging.INFO)

datename = 'localhost-20181120_144618'
root =''
if len(sys.argv)>=2:
    datename = sys.argv[1]
if len(sys.argv)>=3:
    root = sys.argv[2]

if os.path.exists(datename + '.zarr'):
    logging.info(f'Loading dataset from {datename}.zarr.')
    dataset = dst.load(datename + '.zarr')
else:
    logging.info(f'Assembling dataset for {datename}.')
    dataset = dst.assemble(datename, root=root)
    logging.info(f'Saving dataset to {datename}.zarr.')
    dst.save(datename + '.zarr', dataset)
logging.info(dataset)
filepath = dataset.attrs['video_filename']
vr = VideoReaderNP(filepath[:-3] + 'avi')

# indices
t0 = 1_100_000
span = 100_000
if hasattr(dataset, 'song'):
    tmax = len(dataset.song)
else:
    tmax = len(dataset.body_positions)*10

crop = False
fly = 0
nb_flies = np.max(dataset.flies).values+1
STOP = True

app = pg.QtGui.QApplication([])
win = pg.QtGui.QMainWindow()
win.resize(800, 800)
win.setWindowTitle("psv")

image_view = ImageViewVR(name="img_view")
image_view.setImage(vr)

slice_view = pg.PlotWidget(name="song")

cw = KeyPressWidget()
cw.sigKeyPress.connect(keyPressed)

l = pg.QtGui.QVBoxLayout()
l.addWidget(image_view, stretch=4)
l.addWidget(slice_view, stretch=1)
cw.setLayout(l)
win.setCentralWidget(cw)
win.show()

image_view.ui.histogram.hide()
image_view.ui.roiBtn.hide()
image_view.ui.menuBtn.hide()

update(t0,span, crop, fly)

# Start Qt event loop unless running in interactive mode or using pyside.
if __name__ == '__main__':

    import sys
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        QtGui.QApplication.instance().exec_()
