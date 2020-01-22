'''Obtain measurements of the reaction forces and displacements.
'''

import os
import scipy
import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt

from examples.utility import apply_mean_filter


CURRENT_DIRECTORY = os.path.dirname(os.path.relpath(__file__))
CURRENT_DIRECTORY_NAME = os.path.basename(CURRENT_DIRECTORY)

PARENT_DIRECTORY = os.path.dirname(CURRENT_DIRECTORY)
PARENT_DIRECTORY_NAME = os.path.basename(PARENT_DIRECTORY)

SUBDIRECTORY_INPUT_DATA = os.path.join("datafiles_unprocessed", "reactionforce")
DIRECTORY_INPUT_DATA = os.path.join(CURRENT_DIRECTORY, SUBDIRECTORY_INPUT_DATA)

SUBDIRECTORY_OUTPUT_DATA = os.path.join("datafiles_processed", "reactionforce")
DIRECTORY_OUTPUT_DATA = os.path.join(CURRENT_DIRECTORY, SUBDIRECTORY_OUTPUT_DATA)
DIRECTORY_OUTPUT_FIGURES = os.path.join(CURRENT_DIRECTORY, "results")


PLOT_DATA = True
SAVE_PLOTS = False
WRITE_DATA = False

DISCARD_OUTLIER_POINTS = True
OUTLIER_POINTS = [0, 9, 98, 181, 182, 188, 247, 302, 338,
                  362, 371, 501, 510, 522, 553, 737, 865]

APPLY_TEMPORAL_FILTER = True
TEMPORAL_FILTERING_TIMES = 5
TEMPORAL_FILTER_KERNELS = [
    np.ones((33,), float),
    np.ones((17,), float),
    np.ones(( 9,), float),
    np.ones(( 5,), float),
    np.ones(( 3,), float)] # Flat-top filters

# USE_DISPLACEMENT_RANGE = [-np.inf, np.inf]
USE_DISPLACEMENT_RANGE = [0.275, np.inf]

REVERSE_MEASUREMENT_SIGN_FINALLY = True
SET_MEASUREMENTS_AT_ORIGIN = True


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


### Measurement data files

ut_msr = np.loadtxt(os.path.join(DIRECTORY_INPUT_DATA, 'u_moving_pad.dat'), ndmin=1)
ft_msr = np.loadtxt(os.path.join(DIRECTORY_INPUT_DATA, 'f_moving_pad.dat'), ndmin=1)

if ut_msr.ndim > 1: ut_msr = ut_msr.squeeze()
if ft_msr.ndim > 1: ft_msr = ft_msr.squeeze()

if not (ut_msr.ndim == ft_msr.ndim == 1):
    raise RuntimeError

if ut_msr.shape != ft_msr.shape:
    raise RuntimeError


### Discard outlier points

index_bad_data = OUTLIER_POINTS

index_good_data = list(set(range(len(ut_msr))).difference(index_bad_data))

ut_msr_outlier = ut_msr[OUTLIER_POINTS]
ft_msr_outlier = ft_msr[OUTLIER_POINTS]

if DISCARD_OUTLIER_POINTS:
    ut_msr = np.ascontiguousarray(ut_msr[index_good_data])
    ft_msr = np.ascontiguousarray(ft_msr[index_good_data])


### Filter measurements

if APPLY_TEMPORAL_FILTER:

    ut_msr_flt = ut_msr.tolist()
    ft_msr_flt = ft_msr.tolist()

    for w_i in TEMPORAL_FILTER_KERNELS:
        for _ in range(TEMPORAL_FILTERING_TIMES):
            apply_mean_filter(w_i, ut_msr_flt)
            apply_mean_filter(w_i, ft_msr_flt)

    # i0_msr = len(w_i)+1
    #
    # ut_msr_flt = np.array(ut_msr_flt[i0_msr:-i0_msr+1])
    # ft_msr_flt = np.array(ft_msr_flt[i0_msr:-i0_msr+1])

    ut_msr_flt = np.array(ut_msr_flt)
    ft_msr_flt = np.array(ft_msr_flt)

else:

    ut_msr_flt = ut_msr.copy()
    ft_msr_flt = ft_msr.copy()

### Trim displacement range

mask = (USE_DISPLACEMENT_RANGE[0] < ut_msr_flt) \
     * (ut_msr_flt < USE_DISPLACEMENT_RANGE[1])

ut_msr_flt = ut_msr_flt[mask].copy()
ft_msr_flt = ft_msr_flt[mask].copy()


### Reset measurement at origin

if SET_MEASUREMENTS_AT_ORIGIN:

    offset_value_ut = ut_msr_flt[0]
    offset_value_ft = ft_msr_flt[0]

    ut_msr -= offset_value_ut
    ft_msr -= offset_value_ft

    ut_msr_outlier -= offset_value_ut
    ft_msr_outlier -= offset_value_ft

    ut_msr_flt -= offset_value_ut
    ft_msr_flt -= offset_value_ft


### Export these variables

ux_pad_mov = -ut_msr_flt if REVERSE_MEASUREMENT_SIGN_FINALLY else ut_msr_flt
fx_pad_mov = -ft_msr_flt if REVERSE_MEASUREMENT_SIGN_FINALLY else ft_msr_flt

measurements = {
    'ux_pad_mov': ux_pad_mov,
    'fx_pad_mov': fx_pad_mov,
    }


if __name__ == "__main__":

    plt.interactive(True)
    plt.close('all')
    plt.show()

    if PLOT_DATA or SAVE_PLOTS:

        all_figure_names = []
        all_figure_handles = []


        figname = 'Pad Displacement Measurements'

        fh = plt.figure(figname)
        ah = fh.add_subplot(111)

        ah.plot(index_good_data, ut_msr, 'k:')
        ah.plot(index_bad_data, ut_msr_outlier, 'rx', markersize=4.0)


        # ah.set_title(figname)
        ah.set_xlabel('Measurement snapshot (#)')
        ah.set_ylabel('Pad displacement, $|u|$ (mm)')
        ah.legend(['ignored data', 'used data'])

        all_figure_names.append(figname)
        all_figure_handles.append(fh)


        figname = 'Pad Reaction Force Measurements'

        fh = plt.figure(figname)
        ah = fh.add_subplot(111)

        ah.plot(index_bad_data, ft_msr_outlier, 'rx', markersize=4.0)
        ah.plot(index_good_data, ft_msr, 'ko', markersize=1.0)

        # ah.set_title(figname)
        ah.set_xlabel('Measurement snapshot (#)')
        ah.set_ylabel('Pad reaction force, $|f|$ (N)')
        ah.legend(['ignored data', 'used data'])

        all_figure_names.append(figname)
        all_figure_handles.append(fh)


        figname = 'Pad Reaction Force vs Displacement Curve'
        fh = plt.figure(figname)
        ah = fh.add_subplot(111)

        ah.plot(ut_msr, ft_msr, 'ko', markersize=1.0)
        ah.plot(ut_msr_flt, ft_msr_flt, 'r-')
        ah.plot(ut_msr_flt[0], ft_msr_flt[0], 'ws', markersize=4.0, markeredgecolor='r')

        # ah.set_title(figname)
        ah.set_xlabel('Pad displacement, $|u|$ (mm)')
        ah.set_ylabel('Pad reaction force, $|f|$ (N)')
        ah.legend(['raw data', 'filtered data', 'assumed origin'])

        all_figure_names.append(figname)
        all_figure_handles.append(fh)


        if SAVE_PLOTS:

            if not os.path.isdir(DIRECTORY_OUTPUT_FIGURES):
                os.makedirs(DIRECTORY_OUTPUT_FIGURES)

            for handle_i, name_i in zip(all_figure_handles, all_figure_names):

                savename = name_i.lower().strip().replace(' ', '_')
                savepath = os.path.join(DIRECTORY_OUTPUT_FIGURES, savename)

                handle_i.savefig(savepath+'.png', dpi=300)
                handle_i.savefig(savepath+'.svg')
                handle_i.savefig(savepath+'.pdf')

            if not PLOT_DATA:
                plt.close('all')

    if WRITE_DATA:
        write_data_files({
            'ux_pad_mov': ux_pad_mov,
            'fx_pad_mov': fx_pad_mov,
            })
