'''
problems/healthy_skin_fixed_pads/data.py

TODO:
    - rotate axis

'''

import os
import sys
import scipy
import numpy as np
import scipy.optimize
import scipy.interpolate
import matplotlib.pyplot as plt

from invsolve import prepare
from invsolve import project


### Problem Parameters

EXTENSIOMETER_WIDTH = 24.0
EXTENSIOMETER_PAD_PERIMETER = 36.0


DIC_DISCARD_OUTLIERS = False
DIC_FILTER_ROTATIONS = False

DIC_FILTER_DISPLACEMENTS_TEMPORAL = True
DIC_FILTER_DISPLACEMENTS_SPATIAL = True

ASSUME_FIRST_SNAPSHOT = 20
IGNORE_RESIDUAL_REACTON_FORCE = True
IGNORE_RESIDUAL_DISPLACEMENT = True

USE_FILTERED_MEASUREMENTS = True
INTERPOLATE_MEASUREMENTS = True
CURVE_FIT_MEASUREMENTS = False

# PLOT_DATA = __name__ == '__main__'
# PLOT_DATA = PLOT_DATA and True
# PLOT_DATA_DIC = PLOT_DATA and False
PLOT_DATA = True
PLOT_DATA_DIC = True
### Measurement Data Files

DATADIR = './data/healthy_skin'

SUBDIR_DIC = 'dic_displacements'
SUBDIR_REG = 'dic_displacements_reg'

FILENAME_GRID_X   = 'grid_x.dat'
FILENAME_GRID_Y   = 'grid_y.dat'
FILENAME_VALUES_X = 'validx.dat'
FILENAME_VALUES_Y = 'validy.dat'

FILENAME_GRID_X_REG_1   = 'grid_xleft.dat'
FILENAME_GRID_Y_REG_1   = 'grid_yleft.dat'
FILENAME_VALUES_X_REG_1 = 'validxleft.dat'
FILENAME_VALUES_Y_REG_1 = 'validyleft.dat'

FILENAME_GRID_X_REG_2   = 'grid_xright.dat'
FILENAME_GRID_Y_REG_2   = 'grid_yright.dat'
FILENAME_VALUES_X_REG_2 = 'validxright.dat'
FILENAME_VALUES_Y_REG_2 = 'validyright.dat'

FILENAME_PAD_DISPLACEMENTS  = 'pad_displacement_values.dat'
FILENAME_PAD_REACTION_FORCE = 'pad_reaction_force_values.dat'

data_dic_window = prepare.DisplacementMeasurement.load_from_files(
    filepath_xk=os.path.join(DATADIR, SUBDIR_DIC, FILENAME_GRID_X),
    filepath_yk=os.path.join(DATADIR, SUBDIR_DIC, FILENAME_GRID_Y),
    filepath_uk=os.path.join(DATADIR, SUBDIR_DIC, FILENAME_VALUES_X),
    filepath_vk=os.path.join(DATADIR, SUBDIR_DIC, FILENAME_VALUES_Y))
	
# print(data_dic_window)
# input("press to continue ...")

data_dic_pad_one = prepare.DisplacementMeasurement.load_from_files(
    filepath_xk=os.path.join(DATADIR, SUBDIR_REG, FILENAME_GRID_X_REG_1),
    filepath_yk=os.path.join(DATADIR, SUBDIR_REG, FILENAME_GRID_Y_REG_1),
    filepath_uk=os.path.join(DATADIR, SUBDIR_REG, FILENAME_VALUES_X_REG_1),
    filepath_vk=os.path.join(DATADIR, SUBDIR_REG, FILENAME_VALUES_Y_REG_1))

data_dic_pad_two = prepare.DisplacementMeasurement.load_from_files(
    filepath_xk=os.path.join(DATADIR, SUBDIR_REG, FILENAME_GRID_X_REG_2),
    filepath_yk=os.path.join(DATADIR, SUBDIR_REG, FILENAME_GRID_Y_REG_2),
    filepath_uk=os.path.join(DATADIR, SUBDIR_REG, FILENAME_VALUES_X_REG_2),
    filepath_vk=os.path.join(DATADIR, SUBDIR_REG, FILENAME_VALUES_Y_REG_2))


# data_dic_window.uk.insert(0, data_dic_window.view_coords())
# data_dic_pad_one.uk.insert(0, data_dic_pad_one.view_coords())
# data_dic_pad_two.uk.insert(0, data_dic_pad_two.view_coords())

data_dic_window.set_values_relative()
data_dic_pad_one.set_values_relative()
data_dic_pad_two.set_values_relative()


### Rotate axis (OPTIONAL)

# if DIC_FILTER_ROTATIONS:
#
#     # TODO
#
#     dx, dy = np.diff(data_dic_pad_two.view_coords(), axis=0)[0]
#     angle = np.arctan2(dy,dx)
#
#     print(dx)
#     print(dy)
#     print(angle)
#
#     # (data_dic_window + data_dic_pad_one + data_dic_pad_two) \
#     #     .plot_data('DIC: Raw Input Data')


### Discard outlier (artefact) values

if DIC_DISCARD_OUTLIERS:

    points_to_remove_from_dic = [
        (449.333, 328.994),
        (419.220, 288.840),
        (419.908, 349.060),
        (499.142, 408.897),
        (429.379, 368.600),
        (478.913, 398.797),
        (429.434, 298.771)] # units px

    xk_removed, uk_removed, idc_removed = data_dic_window.\
        remove_data_points(points_to_remove_from_dic)

    if PLOT_DATA_DIC:
        (data_dic_window + data_dic_pad_one + data_dic_pad_two) \
            .plot_data('DIC: Removed Outliers')
        input('plot DIC: Removed Outliers ...')


### Apply Temporal Filtering

if DIC_FILTER_DISPLACEMENTS_TEMPORAL:

    filter_weights = [
        np.ones((5,),float),
        np.ones((3,),float)
        ] # TODO: is filtering required?

    for w in filter_weights:
        data_dic_window.filter_values(w, count=10)
        data_dic_pad_one.filter_values(w, count=10)
        data_dic_pad_two.filter_values(w, count=10)

    if PLOT_DATA_DIC:
        (data_dic_window + data_dic_pad_one + data_dic_pad_two) \
            .plot_data('DIC: Values Filtered in Time')
        input('plot DIC: Values Filtered in Time ...')


### Apply Spatial Filtering (meshless interpolation)

if DIC_FILTER_DISPLACEMENTS_SPATIAL:

    xk = data_dic_window.view_coords()
    uk = data_dic_window.view_displacements()

    ui = project.project_pointvalues_on_points(xk, uk, xi=xk,
        meshless_degree=0, num_neighbors=None, distance_norm=2)

    data_dic_window.uk = ui

    if PLOT_DATA_DIC:
        (data_dic_window + data_dic_pad_one + data_dic_pad_two) \
            .plot_data('DIC: Values Filtered in Space')
        input('plot DIC: Values Filtered in Space ...')

### Scale DIC Measurements from px to mm

xk_dic_pad_one = data_dic_pad_one.view_coords()
xk_dic_pad_two = data_dic_pad_two.view_coords()

_, dy_pad_one = xk_dic_pad_one[1,:] - xk_dic_pad_one[0,:]
_, dy_pad_two = xk_dic_pad_two[1,:] - xk_dic_pad_two[0,:]

extensiometer_width_from_image = (dy_pad_one + dy_pad_two) * 0.5
scale = EXTENSIOMETER_WIDTH / extensiometer_width_from_image

data_dic_window.rescale(scale)
data_dic_pad_one.rescale(scale)
data_dic_pad_two.rescale(scale)


### Set DIC Displacements Relative to Pad Two

uk_mean_dic_pad_two = data_dic_pad_two.compute_mean_values()

data_dic_window.offset_values(-uk_mean_dic_pad_two)
data_dic_pad_one.offset_values(-uk_mean_dic_pad_two)
data_dic_pad_two.offset_values(-uk_mean_dic_pad_two)


### Set DIC Coordiantes Relative to Pad One

xk_mean_dic_pad_one = data_dic_pad_one.compute_mean_coords()

data_dic_window.offset_coords(-xk_mean_dic_pad_one)
data_dic_pad_one.offset_coords(-xk_mean_dic_pad_one)
data_dic_pad_two.offset_coords(-xk_mean_dic_pad_one)

#CENTER_MESH = np.array([50., 20.])
INTER_PADS = 36.
WIDTH_PAD = 24.
xk_mean_dic_pad_one = np.array([50. - 0.5*INTER_PADS, 20.])

data_dic_window.offset_coords(+xk_mean_dic_pad_one)
data_dic_pad_one.offset_coords(+xk_mean_dic_pad_one)
data_dic_pad_two.offset_coords(+xk_mean_dic_pad_one)


if PLOT_DATA_DIC:
    (data_dic_window + data_dic_pad_one + data_dic_pad_two) \
        .plot_data('DIC: Rescaled to Measurement Domain')
    input('plot DIC: Rescaled to Measurement Domain ...')

### IMPORTANT: Get DIC Measurements

xk_dic_window = data_dic_window.view_coords()
uk_dic_window = data_dic_window.view_displacements()

p0_dic_window = np.round(xk_dic_window.min(axis=0))
p1_dic_window = np.round(xk_dic_window.max(axis=0))


### IMPORTANT: Get Pad Measurements

xk_dic_pad_one = data_dic_pad_one.view_coords()
xk_dic_pad_two = data_dic_pad_two.view_coords()

xc_dic_pad_one = np.round(xk_dic_pad_one.mean(axis=0))
xc_dic_pad_two = np.round(xk_dic_pad_two.mean(axis=0))

ux_dic_pad_one = data_dic_pad_one.compute_mean_values()[:,0]
ux_dic_pad_two = data_dic_pad_two.compute_mean_values()[:,0]

# print(ux_dic_pad_one)
# input('printing the pad displacement from DIC')


### Get Force-Displacement Measurements

data_force_displacement = prepare.ForceMeasurement.load_from_files(
    filepath_uk=os.path.join(DATADIR, FILENAME_PAD_DISPLACEMENTS),
    filepath_fk=os.path.join(DATADIR, FILENAME_PAD_REACTION_FORCE))

# force_scale = 1.0/EXTENSIOMETER_PAD_PERIMETER
# data_force_displacement.rescale_forces(force_scale)

ux_reg_pad_one = data_force_displacement.view_displacements()
fx_reg_pad_one = data_force_displacement.view_forces()


### Filter Pad Displacements

ux_reg_pad_one_filter = ux_reg_pad_one.copy()

filter_weights = [
    np.ones((33,),float),
    np.ones((17,),float),
    np.ones(( 9,),float),
    np.ones(( 5,),float),
    np.ones(( 3,),float),
    ] # TODO: is filtering required?

for w in filter_weights:
    ux_reg_pad_one_filter = prepare.weighted_average_filter(
        ux_reg_pad_one_filter, w, count=10)

if PLOT_DATA:

    fig_name = 'Pad Displacements'
    fh = plt.figure(fig_name)
    fh.clear()

    ax = fh.add_subplot(111)
    ax.plot(ux_reg_pad_one,'r.', linewidth=0.75, markersize=2)
    ax.plot(ux_reg_pad_one_filter,'k--')
    ax.legend(['raw data', 'filtered'])

    ax.set_title(fig_name)
    ax.set_xlabel('Time snapshot (#)')
    ax.set_ylabel('Pad displacement (mm)')

    plt.tight_layout()
    plt.show()


### Filter Pad Reaction Force

fx_reg_pad_one_filter = fx_reg_pad_one.copy()

filter_weights = [
    np.ones((33,),float),
    np.ones((17,),float),
    np.ones(( 9,),float),
    np.ones(( 5,),float),
    np.ones(( 3,),float),
    ] # TODO: is filtering required?

for w in filter_weights:
    fx_reg_pad_one_filter = prepare.weighted_average_filter(
        fx_reg_pad_one_filter, w, count=1)

if PLOT_DATA:

    fig_name = 'Pad Reaction Forces'
    fh = plt.figure(fig_name)
    fh.clear()

    ax = fh.add_subplot(111)
    ax.plot(fx_reg_pad_one,'r.', linewidth=0.75, markersize=2)
    ax.plot(fx_reg_pad_one_filter, 'k--')
    ax.legend(['raw data', 'filtered'])

    ax.set_title(fig_name)
    ax.set_xlabel('Time snapshot (#)')
    ax.set_ylabel('Pad reaction force (N)')

    plt.tight_layout()
    plt.show()


### Pad Reaction Force vs. Displacement

if PLOT_DATA:

    fig_name = 'Pad Reaction Force vs. Displacement'
    fh = plt.figure(fig_name)
    fh.clear()

    ax = fh.add_subplot(111)
    ax.plot(ux_reg_pad_one, fx_reg_pad_one,'r.', linewidth=0.75, markersize=2)
    ax.plot(ux_reg_pad_one_filter, fx_reg_pad_one_filter, 'k--')
    ax.legend(['raw data', 'filtered'])

    ax.set_title(fig_name)
    ax.set_xlabel('Pad displacement (mm)')
    ax.set_ylabel('Pad reaction force (N)')

    plt.tight_layout()
    plt.show()


### Consider a sub-interval of the experment (where data is cleaner)

if ASSUME_FIRST_SNAPSHOT:

    ux_reg_pad_one = ux_reg_pad_one[ASSUME_FIRST_SNAPSHOT:]
    fx_reg_pad_one = fx_reg_pad_one[ASSUME_FIRST_SNAPSHOT:]

    ux_reg_pad_one_filter = ux_reg_pad_one_filter[ASSUME_FIRST_SNAPSHOT:]
    fx_reg_pad_one_filter = fx_reg_pad_one_filter[ASSUME_FIRST_SNAPSHOT:]


### Consider subtracting the residual displacement and force

# NOTE: Supposing the residual force is small, the energy contribution to the
# total potential energy by the residual stresses and the deformations will be
# small also. Therefore, it is okey to ignore the residual stresses and force.


if IGNORE_RESIDUAL_DISPLACEMENT:
    ux_reg_pad_one_filter_min = ux_reg_pad_one_filter.min()
    ux_reg_pad_one_filter -= ux_reg_pad_one_filter_min
    ux_reg_pad_one -= ux_reg_pad_one_filter_min

if IGNORE_RESIDUAL_REACTON_FORCE:
    fx_reg_pad_one_filter_min = fx_reg_pad_one_filter.min()
    fx_reg_pad_one_filter -= fx_reg_pad_one_filter_min
    fx_reg_pad_one -= fx_reg_pad_one_filter_min

if PLOT_DATA and (IGNORE_RESIDUAL_DISPLACEMENT or IGNORE_RESIDUAL_DISPLACEMENT):

    fig_name = 'Pad Reaction Force vs. Displacement (no residual)'
    fh = plt.figure(fig_name)
    fh.clear()

    ax = fh.add_subplot(111)
    ax.plot(ux_reg_pad_one, fx_reg_pad_one, 'r.', linewidth=0.75, markersize=2)
    ax.plot(ux_reg_pad_one_filter, fx_reg_pad_one_filter, 'k--')
    ax.legend(['raw data', 'filtered'])

    ax.set_title(fig_name)
    ax.set_xlabel('Pad displacement (mm)')
    ax.set_ylabel('Pad reaction force (N)')

    plt.tight_layout()
    plt.show()


### Decide to Use Filtered Measurements

ux_reg_pad_one_unfiltered = ux_reg_pad_one.copy()
fx_reg_pad_one_unfiltered = fx_reg_pad_one.copy()

if USE_FILTERED_MEASUREMENTS:
    ux_reg_pad_one = ux_reg_pad_one_filter
    fx_reg_pad_one = fx_reg_pad_one_filter

# NOTE: Is the information in the load-displacement curve necessary to fit
# a complex model? Should I consider a simpler load-displacement path
# by doing some regression fitting of the data ? For example, consider a
# polynomial law or an exponential law.

### As a first approach, try interpolating the filtered data

if INTERPOLATE_MEASUREMENTS:

    interp_data = scipy.interpolate.interp1d(
        ux_reg_pad_one[:,0], fx_reg_pad_one[:,0], kind='linear')

    ux_reg_pad_one_interp_dic_abs = np.abs(ux_dic_pad_one)
    fx_reg_pad_one_interp_dic_abs = interp_data(ux_reg_pad_one_interp_dic_abs)

    if PLOT_DATA:

        fig_name = 'Pad Reaction Force vs. Displacement (interpolated)'
        fh = plt.figure(fig_name)
        fh.clear()

        ax = fh.add_subplot(111)
        ax.plot(ux_reg_pad_one_unfiltered, fx_reg_pad_one_unfiltered, 'r.', linewidth=0.75, markersize=2)
        ax.plot(ux_reg_pad_one_interp_dic_abs, fx_reg_pad_one_interp_dic_abs, 'k-|')
        ax.legend(['raw data', 'model fitting range'])

        ax.set_title(fig_name)
        ax.set_xlabel('Pad displacement (mm)')
        ax.set_ylabel('Pad reaction force (N)')

        plt.tight_layout()
        plt.show()


### As a second approach, try fitting a curve

if CURVE_FIT_MEASUREMENTS:

    def func(x, a, b, c):
        return a*x + b*x**2 + c*x**3 # pass curve through origin
    func_kind = 'P3'

    # def func(x, a, b, c):
    #     return a*(np.exp(b*x)-1.0) # pass curve through origin
    # func_kind = 'exp'

    popt, pcov = scipy.optimize.curve_fit(
        func, ux_reg_pad_one[:,0], fx_reg_pad_one[:,0])

    ux_reg_pad_one_fitted_dic_abs = np.abs(ux_dic_pad_one)
    fx_reg_pad_one_fitted_dic_abs = func(ux_reg_pad_one_fitted_dic_abs, *popt)

    if PLOT_DATA:

        fig_name = 'Pad Reaction Force vs. Displacement (curve-fitted)'
        fh = plt.figure(fig_name)
        fh.clear()

        ax = fh.add_subplot(111)
        ax.plot(ux_reg_pad_one_unfiltered, fx_reg_pad_one_unfiltered, 'r.', linewidth=0.75, markersize=2)
        ax.plot(ux_reg_pad_one, func(ux_reg_pad_one,*popt), 'b--')
        ax.plot(ux_reg_pad_one_fitted_dic_abs, fx_reg_pad_one_fitted_dic_abs, 'k-|')

        ax.legend([
            'raw data',
            'data fitted model ({func_kind})',
            'hyper-elastic model fitting range'])

        ax.set_title(fig_name)
        ax.set_xlabel('Pad displacement (mm)')
        ax.set_ylabel('Pad reaction force (N)')

        ax.set_ylim([ax.get_ylim()[0] - 0.05, ax.get_ylim()[1] + 0.05])
        plt.tight_layout()
        plt.show()


### Names to Export for Solver

msr_dic_xk = xk_dic_window
msr_dic_uk = uk_dic_window

msr_pad_one_ux = ux_reg_pad_one_interp_dic_abs * (-1.0)
msr_pad_one_fx = fx_reg_pad_one_interp_dic_abs * (-1.0)


### Names to Export for Meshing

geo_dic_pad_one_xc = xc_dic_pad_one
geo_dic_pad_two_xc = xc_dic_pad_two

geo_dic_measure_p0 = p0_dic_window
geo_dic_measure_p1 = p1_dic_window

# msr_pad_one_ux_abs
# msr_pad_one_fx_abs


### Export these variable aliases also

ux_msr_pad_one = msr_pad_one_ux
fx_msr_pad_one = msr_pad_one_fx

### Export these variables

out = {
    'u_msr_dic_pnt' : msr_dic_uk,
    'x_msr_dic_pnt' : msr_dic_xk,
    'ux_msr_pad_one' : ux_msr_pad_one,
    'fx_msr_pad_one' : fx_msr_pad_one,
    'ux_msr_pad_one_raw' : ux_reg_pad_one_unfiltered,
    'fx_msr_pad_one_raw' : fx_reg_pad_one_unfiltered,
    }


if __name__ == '__main__':
    pass
    # import sys
    # import time
    #
    # import measurement_projection
    # from measurement_projection import MeshlessInterpolation
    # from measurement_projection import SimpleMeshlessInterpolation2d
    # measurement_projection.HAS_WLSQM = True
    #
    # xk = xk_dic_window
    # fk = uk_dic_window[-1]
    #
    # # fk[:,0] = 0
    # fk *= 4
    #
    # x_min, y_min = xk.min(axis=0)
    # x_max, y_max = xk.max(axis=0)
    #
    # x = np.linspace(x_min,x_max,50)
    # y = np.linspace(y_min,y_max,25)
    # x, y = np.meshgrid(x,y)
    # x = x.reshape((-1,))
    # y = y.reshape((-1,))
    #
    # xi = np.stack([x,y], axis=1)
    # # xi = xk
    #
    #
    # degree = 0
    # neighbors = 25
    # weight = 'uniform' # 'uniform' or 'center'
    #
    # degree = 1
    # neighbors = 49
    # weight = 'center'
    #
    # # degree = 2
    # # neighbors = 81
    # # weight = 'center'
    #
    #
    # fh = plt.figure('compare meshless implementations')
    # fh.clear();
    # ax = fh.add_subplot(111)
    # fh.axes[0].axis('equal')
    #
    # ax.scatter(xk[:,0]+fk[:,0],xk[:,1]+fk[:,1], s=60,
    #     facecolors='none', edgecolors='g', marker='s')
    #
    #
    # t0 = time.time()
    #
    # meshless_interp = SimpleMeshlessInterpolation2d(xk)
    # meshless_interp.set_interpolation_points(xi, neighbors)
    # fi = meshless_interp.interpolate(fk, degree, weight)
    #
    # t1 = time.time()
    # print('Time diff.:',t1-t0)
    #
    # ax.scatter(xi[:,0]+fi[:,0],xi[:,1]+fi[:,1], s=60,
    #     c='k', marker='+')
    #
    #
    # t0 = time.time()
    #
    # meshless_interp = MeshlessInterpolation(xk)
    # meshless_interp.set_interpolation_points(xi, neighbors)
    # fi = meshless_interp.interpolate(fk, degree, weight)
    #
    # t1 = time.time()
    # print('Time diff.:',t1-t0)
    #
    # ax.scatter(xi[:,0]+fi[:,0],xi[:,1]+fi[:,1], s=60,
    #     facecolors='none', edgecolors='r', marker='o')
    #
    # plt.show()
