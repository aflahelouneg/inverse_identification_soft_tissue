"""
Obtain displacement measurements from digital image correlation results.
"""

import os
import time
import scipy
import numpy as np
import matplotlib.pyplot as plt

from examples.utility import apply_mean_filter


CURRENT_DIRECTORY = os.path.dirname(os.path.relpath(__file__))
CURRENT_DIRECTORY_NAME = os.path.basename(CURRENT_DIRECTORY)

PARENT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
PARENT_DIRECTORY_NAME = os.path.basename(PARENT_DIRECTORY)

SUBDIRECTORY_INPUT_DATA = os.path.join("datafiles_unprocessed", "displacement")
DIRECTORY_INPUT_DATA = os.path.join(CURRENT_DIRECTORY, SUBDIRECTORY_INPUT_DATA)

SUBDIRECTORY_OUTPUT_DATA = os.path.join("datafiles_processed", "displacement")
DIRECTORY_OUTPUT_DATA = os.path.join(CURRENT_DIRECTORY, SUBDIRECTORY_OUTPUT_DATA)
DIRECTORY_OUTPUT_FIGURES = os.path.join(CURRENT_DIRECTORY, "results")


PLOT_DATA = True
SAVE_PLOTS = False
WRITE_DATA = False

EXTENSIOMETER_WINDOW_WIDTH = 24.0 # Independent of experiment
# EXTENSIOMETER_PAD_DISTANCE = 36.5 # Depends on experiment (NOTE: Not used)

DISCARD_SPURIOUS_POINTS = True

SPURIOUS_POINTS = [
    (449.333, 328.994),
    (419.220, 288.840),
    (419.908, 349.060),
    (499.142, 408.897),
    (429.379, 368.600),
    (478.913, 398.797),
    (429.434, 298.771)]

APPLY_TEMPORAL_FILTER = True
TEMPORAL_FILTERING_TIMES = 10
TEMPORAL_FILTER_KERNELS = [
    np.ones((5,),float),
    np.ones((3,),float)] # Flat-top filters


def read_data_files(file_xt, file_yt, delimiter=None):

    xt = np.loadtxt(file_xt, dtype=float, delimiter=delimiter, ndmin=2)
    yt = np.loadtxt(file_yt, dtype=float, delimiter=delimiter, ndmin=2)

    if xt.shape != yt.shape:
        raise TypeError('Different shapes of arrays `xt` and `yt`.')

    nt = xt.shape[1]

    xt = np.split(xt, nt, axis=1)
    yt = np.split(yt, nt, axis=1)

    xt = [np.concatenate(xt_i, axis=1)
          for xt_i in zip(xt, yt)]

    return xt


def write_data_files(dct):

    if not os.path.isdir(DIRECTORY_OUTPUT_DATA):
        os.makedirs(DIRECTORY_OUTPUT_DATA)

    for key, val in dct.items():
        if isinstance(val, (list, tuple)):
            if not all(isinstance(val_i, np.ndarray) for val_i in val):
                raise RuntimeError('Expected the sequence to contain arrays.')
            for i, val_i in enumerate(val):
                np.savetxt(os.path.join(DIRECTORY_OUTPUT_DATA, key+f'_{i:04d}.out'), val_i)
        else:
            np.savetxt(os.path.join(DIRECTORY_OUTPUT_DATA, key+'.out'), val)


### Read measurements

x0_dic = read_data_files(
    os.path.join(DIRECTORY_INPUT_DATA, 'x0.dat'),
    os.path.join(DIRECTORY_INPUT_DATA, 'y0.dat'))
xt_dic = read_data_files(
    os.path.join(DIRECTORY_INPUT_DATA, 'xt.dat'),
    os.path.join(DIRECTORY_INPUT_DATA, 'yt.dat'))

x0_ref_1 = read_data_files(
    os.path.join(DIRECTORY_INPUT_DATA, 'x0_ref_1.dat'),
    os.path.join(DIRECTORY_INPUT_DATA, 'y0_ref_1.dat'))
xt_ref_1 = read_data_files(
    os.path.join(DIRECTORY_INPUT_DATA, 'xt_ref_1.dat'),
    os.path.join(DIRECTORY_INPUT_DATA, 'yt_ref_1.dat'))

x0_ref_2 = read_data_files(
    os.path.join(DIRECTORY_INPUT_DATA, 'x0_ref_2.dat'),
    os.path.join(DIRECTORY_INPUT_DATA, 'y0_ref_2.dat'))
xt_ref_2 = read_data_files(
    os.path.join(DIRECTORY_INPUT_DATA, 'xt_ref_2.dat'),
    os.path.join(DIRECTORY_INPUT_DATA, 'yt_ref_2.dat'))

if x0_dic[0].shape != xt_dic[0].shape:
    raise RuntimeError

if x0_ref_1[0].shape != xt_ref_1[0].shape:
    raise RuntimeError

if x0_ref_2[0].shape != xt_ref_2[0].shape:
    raise RuntimeError

xt_dic.insert(0, x0_dic[0])
xt_ref_1.insert(0, x0_ref_1[0])
xt_ref_2.insert(0, x0_ref_2[0])

del x0_dic
del x0_ref_1
del x0_ref_2

number_of_measurements = len(xt_dic)

if len(xt_ref_1) != number_of_measurements:
    raise RuntimeError('Inconsistent numbers of measurements.')

if len(xt_ref_2) != number_of_measurements:
    raise RuntimeError('Inconsistent numbers of measurements.')


### Remove spurious points

if DISCARD_SPURIOUS_POINTS:

    xs_spr = np.asarray(SPURIOUS_POINTS)
    x0_dic = xt_dic[0]

    assert xs_spr.ndim == 2
    assert xs_spr.shape[1] == x0_dic.shape[1]

    # Indices into `x0_dic` s.t. `x0_dic[ind]` are nearest to `xs_spr`
    ind = [np.argmin(((x0_dic-xi)**2).sum(axis=1)) for xi in xs_spr]

    xs_spr = x0_dic[ind, :].copy()

    msk = np.ones((len(x0_dic),), bool)
    msk[ind]=False; ind=np.flatnonzero(msk)

    for i, xt_dic_i in enumerate(xt_dic):
        xt_dic[i] = xt_dic_i[ind,:].copy()

    assert np.allclose(xs_spr, SPURIOUS_POINTS, rtol=1e-2)


### Temporal filtering

assert isinstance(xt_dic, list)
assert isinstance(xt_ref_1, list)
assert isinstance(xt_ref_2, list)

if APPLY_TEMPORAL_FILTER:
    for w_i in TEMPORAL_FILTER_KERNELS:
        for _ in range(TEMPORAL_FILTERING_TIMES):
            apply_mean_filter(w_i, xt_dic)
            apply_mean_filter(w_i, xt_ref_1)
            apply_mean_filter(w_i, xt_ref_2)


### Scale measurements from px to mm

# NOTE: Using the vertical scale rather than the horizontal scale is more robust
# because the extensometer width is known whereas the pad distance is arbitrary.

_dy_ref_1 = xt_ref_1[0][1,1] - xt_ref_1[0][0,1]
_dy_ref_2 = xt_ref_2[0][1,1] - xt_ref_2[0][0,1]

# _dy = _dy_ref_1
# _dy = _dy_ref_2
_dy = (_dy_ref_1 + _dy_ref_2) * 0.5

scale = EXTENSIOMETER_WINDOW_WIDTH / _dy

for _xs in xt_dic:
    _xs *= scale

for _xs in xt_ref_1:
    _xs *= scale

for _xs in xt_ref_2:
    _xs *= scale


### Compute pad positions

xt_pad_mov = [(xs[0,:] + xs[1,:])*0.5 for xs in xt_ref_1]
xt_pad_fix = [(xs[0,:] + xs[1,:])*0.5 for xs in xt_ref_2]

del xt_ref_1 # Nolonger useful (superseded by `xt_pad_mov`)
del xt_ref_2 # Nolonger useful (superseded by `xt_pad_fix`)


### Make measurements relative to fixed pad

for _xs, _x in zip(xt_dic, xt_pad_fix):
    _xs -= _x

for _xs, _x in zip(xt_pad_mov, xt_pad_fix):
    _xs -= _x

for _x in xt_pad_fix:
    _x[:] = 0.0


### Offset coordinates relative to moving pad

_x = xt_pad_mov[0].copy()

for _xs in xt_dic:
    _xs -= _x

for _xs in xt_pad_mov:
    _xs -= _x

for _xs in xt_pad_fix:
    _xs -= _x


### Displacements

x0_dic     = xt_dic[0].copy()
x0_pad_mov = xt_pad_mov[0].copy()
x0_pad_fix = xt_pad_fix[0].copy()

ut_dic     = [xs - x0_dic for xs in xt_dic]
ut_pad_mov = [xs - x0_pad_mov for xs in xt_pad_mov]
ut_pad_fix = [xs - x0_pad_fix for xs in xt_pad_fix]


### Extensometer dimensions

W = EXTENSIOMETER_WINDOW_WIDTH
L = x0_pad_fix[0] - x0_pad_mov[0]


### Export these data

measurements = {
    'x_dic':     x0_dic,
    'u_dic':     ut_dic,
    'x_pad_mov': np.array(x0_pad_mov, ndmin=2),
    'u_pad_mov': np.array(ut_pad_mov, ndmin=2),
    'x_pad_fix': np.array(x0_pad_fix, ndmin=2),
    'u_pad_fix': np.array(ut_pad_fix, ndmin=2),
    }


### Plotting

def plot_data_frame(index=-1, title=None, ax=None,
                    exclude_registration_points=False):

    if ax is None:
        fh = plt.figure(); fh.clear()
        ax = fh.add_subplot(1,1,1)

    elif not isinstance(ax, plt.Axes):
        raise TypeError('Parameter `ax` must be of type `plt.Axes`.')

    if exclude_registration_points:
        x0, xi = xt_dic[0], xt_dic[index]

    else:

        x0 = np.concatenate([x0_pad_mov[None,:], # 1D -> 2D
                             x0_pad_fix[None,:],
                             x0_dic], axis=0)

        xi = np.concatenate([xt_pad_mov[index][None,:],
                             xt_pad_fix[index][None,:],
                             xt_dic[index]], axis=0)

    ax.scatter(x0[:,0], x0[:,1], c='w', s=5,  marker='o', edgecolor='k')
    ax.scatter(xi[:,0], xi[:,1], c='r', s=10, marker='.', edgecolor='r')

    ax.legend(['undeformed', 'deformed'])

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    ax.axis('equal')

    return ax


def plot_data_frames(title=None, duration=0.05, margin=0.05,
                     exclude_registration_points=False):

    fh = plt.figure(title)
    fh.clear()
    ax = fh.add_subplot(1,1,1)

    if exclude_registration_points:
        xt_ends = [xt_dic[0], xt_dic[-1]]
    else:
        xt_ends = [np.concatenate([a[None,:], b[None,:], c], axis=0)
                   for a, b, c in zip((xt_pad_mov[0], xt_pad_mov[-1]),
                                      (xt_pad_fix[0], xt_pad_fix[-1]),
                                      (xt_dic[0], xt_dic[-1]))]

    xmin = min(xi[:,0].min() for xi in xt_ends)
    ymin = min(xi[:,1].min() for xi in xt_ends)
    xmax = max(xi[:,0].max() for xi in xt_ends)
    ymax = max(xi[:,1].max() for xi in xt_ends)

    m = max(xmax-xmin, ymax-ymin) * margin
    axis_limits = xmin-m, xmax+m, ymin-m, ymax+m

    for index in range(number_of_measurements):

        ax.clear()

        plot_data_frame(index, title, ax, exclude_registration_points)

        ax.axis(axis_limits)

        fh.canvas.draw()
        fh.canvas.flush_events()

        time.sleep(duration)

    return ax


def save_data_frame(fh, name="Untitled"):

    if not isinstance(fh, plt.Figure):
        raise TypeError('Parameter `fh` must be of type `plt.Figure`.')

    if not os.path.isdir(DIRECTORY_OUTPUT_FIGURES):
        os.makedirs(DIRECTORY_OUTPUT_FIGURES)

    savepath = os.path.join(DIRECTORY_OUTPUT_FIGURES, name)

    plt.savefig(savepath+'.png', dpi=300)
    plt.savefig(savepath+'.svg')
    plt.savefig(savepath+'.pdf')


if __name__ == "__main__":

    plt.interactive(True)
    plt.close('all')
    plt.show()

    print(f'Pad separation distance, L: {L:g}')
    print(f'Extensometer window width, W: {W:g}')

    FRAME_INDEX = -1

    if PLOT_DATA or SAVE_PLOTS:

        title = "Displacement Field Measurment (Last Snapshot)"

        ax = plot_data_frame(FRAME_INDEX, title,
            exclude_registration_points=True)

        fh = ax.get_figure()

        if SAVE_PLOTS:

                file_name = title.lower().strip('()')
                for c in (' (', ') ', '(', ')', ' '):
                    file_name = file_name.replace(c, '_')

                save_data_frame(fh, file_name)

        if not PLOT_DATA:
            fh.close()

    if WRITE_DATA:
        write_data_files({
            'xt_dic':     xt_dic,
            'xt_pad_mov': np.array(xt_pad_mov, ndmin=2),
            'xt_pad_fix': np.array(xt_pad_fix, ndmin=2),
            })
