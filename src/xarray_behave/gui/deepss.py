import zarr
import numpy as np
import logging

import xarray_behave as xb
from .formbuilder import YamlFormWidget, YamlDialog

import sys
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
import pyqtgraph as pg

try:
    import dss.make_dataset
    import dss.npy_dir
    import dss.train
    import dss.predict
except ImportError as e:
    print(e)
    print('you may need to install DeepSS: link_to_pypi')

package_dir = xb.__path__[0]


def predict(ds):
    dialog = YamlDialog(yaml_file=package_dir + "/gui/forms/dss_predict.yaml",
                        title='Predict labels using DeepSS')
    dialog.show()
    result = dialog.exec_()

    if result == QtGui.QDialog.Accepted:
        form = dialog.form.get_form_data()
        model_path = form['model_path']
        model_path = model_path.rsplit('_',1)[0]  # split off suffix

        logging.info('   running inference.')
        events, segments, _ = dss.predict.predict(ds.song_raw.compute(), model_path, verbose=1, batch_size=96,
                                                  event_thres=form['event_thres'], event_dist=form['event_dist'],
                                                  event_dist_min=form['event_dist_min'], event_dist_max=form['event_dist_max'],
                                                  segment_thres=form['event_thres'], segment_fillgap=form['segment_fillgap'],
                                                  segment_minlen=form['segment_minlen'],
                                                  )
        return events, segments
    else:
        logging.info('   aborting.')
        return None

def update_predictions(ds):
    from xarray_behave.gui.formbuilder import YamlFormWidget

    import sys
    from pyqtgraph.Qt import QtGui, QtCore, QtWidgets
    import pyqtgraph as pg

    def save_as():
        print('saved')

    app = pg.QtGui.QApplication([])
    win = pg.QtGui.QMainWindow()

    win.setWindowTitle("Configure training")
    widget = YamlFormWidget(yaml_file="/Users/janc/Dropbox/code.py/xarray_behave/src/xarray_behave/gui/forms/update_labels.yaml")
    widget.mainAction.connect(save_as)
    win.setCentralWidget(widget)
    win.show()


def export_for_dss(ds):
    assert ds.song_raw.attrs['sampling_rate_Hz'] == ds.song_events.attrs['sampling_rate_Hz']  # make sure audio data and the annotations are all on the same sampling rate

    event_type = 'sine_manual'
    start_index = 0
    end_index = None
    splits = [0.6, 0.8]
    class_names=['noise', 'song']
    class_types=['segment', 'segment']

    x = ds.song_raw.data
    yy = ds.song_events.loc[:, event_type].data

    if end_index is None or start_index is None:
        indices = np.argwhere(yy == 1)[:, 0]
    if start_index is None:
        start_index = np.max((0, indices[0] - 10_000))

    if end_index is None:
        end_index = np.min((len(yy), indices[-1] + 10_000))

    x = x[start_index:end_index]
    yy = yy[start_index:end_index]

    print(x.shape, yy.shape)
    y = np.zeros((yy.shape[0], len(class_names)), dtype=np.uint8)
    y[:,0] = 1-yy
    y[:,1] = yy
    plt.plot(y[::10, :])
    samplerate = ds.attrs['samplingrate']  # this is the sample rate of your data and the pulse times

    root = dss.make_dataset.init_store(
            nb_channels=x.shape[1],  # number of channels/microphones in the recording
            nb_classes=y.shape[1],  # number of classes to predict - [noise, pulse]
            samplerate=ds.song_raw.attrs['sampling_rate_Hz'],
            class_names=class_names,
            class_types=class_types,
            store_type=zarr.DictStore,  # use zarr.DirectoryStore for big data
            store_name='mouse_test.zarr', # only used with DirectoryStore - this is the path to the directory created
            )
    nb_samples = x.shape[0]

    train_val_test_split = (np.array(splits)*nb_samples).astype(np.int)
    x_splits = np.split(x, train_val_test_split)
    y_splits = np.split(y, train_val_test_split)

    for x_split, y_split, name in zip(x_splits, y_splits, ['train', 'val','test']):
        print(x_split.shape, y_split.shape, name)
        root[name]['x'].append(x_split)
        root[name]['y'].append(y_split)

    dss.npy_dir.save('mouse_test.npy', root)

    dss.train.train(data_dir='mouse_test.npy', model_name='tcn', nb_pre_conv=4, verbose=1)
