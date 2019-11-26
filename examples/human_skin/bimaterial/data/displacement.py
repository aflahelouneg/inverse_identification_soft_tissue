"""
Obtain the displacement measurements from digital image correlation results.
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

# SUBDIRECTORY_INPUT_DATA = os.path.join("datafiles_unprocessed", "displacement", "dic_1")
SUBDIRECTORY_INPUT_DATA = os.path.join("datafiles_unprocessed", "displacement", "dic_2")
DIRECTORY_INPUT_DATA = os.path.join(CURRENT_DIRECTORY, SUBDIRECTORY_INPUT_DATA)

# SUBDIRECTORY_OUTPUT_DATA = os.path.join("datafiles_processed", "displacement", "dic_1")
SUBDIRECTORY_OUTPUT_DATA = os.path.join("datafiles_processed", "displacement", "dic_2")
DIRECTORY_OUTPUT_DATA = os.path.join(CURRENT_DIRECTORY, SUBDIRECTORY_OUTPUT_DATA)
DIRECTORY_OUTPUT_FIGURES = os.path.join(CURRENT_DIRECTORY, "results")


PLOT_DATA = True
SAVE_PLOTS = False
WRITE_DATA = False

EXTENSIOMETER_PAD_WIDTH = 8.0 # Independent of experiment
EXTENSIOMETER_WINDOW_WIDTH = 24.0 # Independent of experiment
# EXTENSIOMETER_PAD_DISTANCE = 36.5 # Depends on experiment (NOTE: Not used)

SCALE_DATA_FROM_PIXELS_TO_MM = True
SET_DISPLACEMENTS_RELATIVE_TO_FIXED_PAD = True
SET_COORDINATES_RELATIVE_TO_MOVING_PAD = True

DISCARD_SPURIOUS_POINTS = True
INDICES_SPURIOUS_POINTS = [1, 18, 23, 24, 30, 42, 55, 95, 99, 136, 141, 146,
    161, 168, 169, 171, 180, 181, 182, 198, 204, 209, 216, 243, 245, 252, 268,
    269, 279, 281, 285, 295, 307, 365, 388, 404, 433, 517, 518, 545, 557, 608,
    627, 646, 647, 654, 655, 656, 659, 660, 664, 665, 666, 678]

APPLY_TEMPORAL_FILTER = True
TEMPORAL_FILTERING_TIMES = 15
TEMPORAL_FILTER_KERNELS = [
    # np.ones((9,),float),
    # np.ones((7,),float),
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

# x0_ref_1 = read_data_files(
#     os.path.join(DIRECTORY_INPUT_DATA, 'x0_ref_1.dat'),
#     os.path.join(DIRECTORY_INPUT_DATA, 'y0_ref_1.dat'))
# xt_ref_1 = read_data_files(
#     os.path.join(DIRECTORY_INPUT_DATA, 'xt_ref_1.dat'),
#     os.path.join(DIRECTORY_INPUT_DATA, 'yt_ref_1.dat'))

# x0_ref_2 = read_data_files(
#     os.path.join(DIRECTORY_INPUT_DATA, 'x0_ref_2.dat'),
#     os.path.join(DIRECTORY_INPUT_DATA, 'y0_ref_2.dat'))
# xt_ref_2 = read_data_files(
#     os.path.join(DIRECTORY_INPUT_DATA, 'xt_ref_2.dat'),
#     os.path.join(DIRECTORY_INPUT_DATA, 'yt_ref_2.dat'))

point_spacing_x_axis = 10

indices_points_left_edge  = np.flatnonzero(x0_dic[0][:,0]
    < x0_dic[0][:,0].min() + point_spacing_x_axis)

indices_points_right_edge = np.flatnonzero(x0_dic[0][:,0]
    > x0_dic[0][:,0].max() - point_spacing_x_axis)

x0_ref_1 = [xs[indices_points_left_edge].copy() for xs in x0_dic]
xt_ref_1 = [xs[indices_points_left_edge].copy() for xs in xt_dic]

x0_ref_2 = [xs[indices_points_right_edge].copy() for xs in x0_dic]
xt_ref_2 = [xs[indices_points_right_edge].copy() for xs in xt_dic]

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

xt_dic = xt_dic[:number_of_measurements]
xt_ref_1 = xt_ref_1[:number_of_measurements]
xt_ref_2 = xt_ref_2[:number_of_measurements]


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

if SCALE_DATA_FROM_PIXELS_TO_MM:

    # NOTE: Using the vertical scale rather than the horizontal scale is more robust
    # because the extensometer width is known whereas the pad distance is arbitrary.

    _dy_ref_1 = xt_ref_1[0][:,1].max() - xt_ref_1[0][:,1].min()
    _dy_ref_2 = xt_ref_2[0][:,1].max() - xt_ref_2[0][:,1].min()

    _dy = (_dy_ref_1 + _dy_ref_2) * 0.5
    scale = EXTENSIOMETER_PAD_WIDTH / _dy

    for _xs in xt_dic:
        _xs *= scale

    for _xs in xt_ref_1:
        _xs *= scale

    for _xs in xt_ref_2:
        _xs *= scale


### Compute pad positions

index_ymax = np.argmax(xt_ref_1[0][:,1])
index_ymin = np.argmin(xt_ref_1[0][:,1])

xt_pad_mov = [(xs[index_ymax] + xs[index_ymin]) * 0.5 for xs in xt_ref_1]

index_ymax = np.argmax(xt_ref_2[0][:,1])
index_ymin = np.argmin(xt_ref_2[0][:,1])

xt_pad_fix = [(xs[index_ymax] + xs[index_ymin]) * 0.5 for xs in xt_ref_2]


### Set measurements relative to fixed pad

if SET_DISPLACEMENTS_RELATIVE_TO_FIXED_PAD:

    for _xs_dic, _xi_pad_fix in zip(xt_dic, xt_pad_fix):
        _xs_dic -= _xi_pad_fix

    for _xs_pad_mov, _xi_pad_fix in zip(xt_pad_mov, xt_pad_fix):
        _xs_pad_mov -= _xi_pad_fix

    for _xi in xt_pad_fix:
        _xi[:] = 0.0


### Set initial coordinates relative to moving pad

if SET_COORDINATES_RELATIVE_TO_MOVING_PAD:

    _xi_pad_mov = xt_pad_mov[0].copy()

    for _xs in xt_dic:
        _xs -= _xi_pad_mov

    for _xs in xt_pad_mov:
        _xs -= _xi_pad_mov

    for _xs in xt_pad_fix:
        _xs -= _xi_pad_mov


### Remove spurious points

def get_indices_compliment(indices, maxsize):
    mask = np.ones((maxsize,), bool)
    mask[indices] = False
    return np.flatnonzero(mask)

indices_points_genuine = list(range(len(xt_dic[0])))
indices_points_spurious = INDICES_SPURIOUS_POINTS

if DISCARD_SPURIOUS_POINTS:
    indices_points_genuine = get_indices_compliment(
        indices_points_spurious, len(xt_dic[0]))

xt_dic_spr = []

for xi_dic in xt_dic:
    xt_dic_spr.append(xi_dic[indices_points_spurious,:].copy())

if DISCARD_SPURIOUS_POINTS:
    for i, xi_dic in enumerate(xt_dic):
        xt_dic[i] = xi_dic[indices_points_genuine,:].copy()


### Displacements

x0_dic     = xt_dic[0].copy()
x0_pad_mov = xt_pad_mov[0].copy()
x0_pad_fix = xt_pad_fix[0].copy()

ut_dic     = [xs - x0_dic for xs in xt_dic]
ut_pad_mov = [xs - x0_pad_mov for xs in xt_pad_mov]
ut_pad_fix = [xs - x0_pad_fix for xs in xt_pad_fix]


### Extensometer dimensions

extensiometer_window_width = EXTENSIOMETER_WINDOW_WIDTH
extensiometer_pad_distance = x0_pad_fix[0] - x0_pad_mov[0]


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

def plot_data_frame(index=-1, title=None, ax=None, annotate=False):

    if ax is None:
        fh = plt.figure(); fh.clear()
        ax = fh.add_subplot(1,1,1)

    elif not isinstance(ax, plt.Axes):
        raise TypeError('Parameter `ax` must be of type `plt.Axes`.')

    ax.scatter(xt_dic[0][:,0],
               xt_dic[0][:,1],
               c="k", s=10, marker='.')

    ax.scatter(xt_dic[index][:,0],
               xt_dic[index][:,1],
               c='r', s=10, marker='o', alpha=0.5)

    if annotate:

        ax.scatter(xt_dic_spr[0][:,0],
            xt_dic_spr[0][:,1],
            c='b', s=20, marker='+')

        ax.scatter(xt_dic_spr[index][:,0],
                xt_dic_spr[index][:,1],
                c='b', s=20, marker='x')

        for i, xi_i in zip(indices_points_genuine, xt_dic[index]):
            ax.annotate(i, xi_i, fontsize='small')

        for i, xi_i in zip(indices_points_spurious, xt_dic_spr[index]):
            ax.annotate(i, xi_i, fontsize='small')

    ax.legend(['undeformed', 'deformed'])

    if title is not None:
        ax.set_title(title)

    ax.set_xlabel('x (mm)')
    ax.set_ylabel('y (mm)')

    ax.axis('equal')

    return ax


def plot_data_frames(title=None, duration=0.05, margin=0.05,
                     exclude_registration_points=False, annotate=False):

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

        plot_data_frame(index, title, ax, annotate)

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

    FRAME_INDEX = -1

    if PLOT_DATA or SAVE_PLOTS:

        title = "Displacement Field Measurment (Last Snapshot)"

        ax = plot_data_frame(FRAME_INDEX, title)

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
