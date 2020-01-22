
import math
import dolfin
import numpy as np
import matplotlib.pyplot as plt


SEQUENCE_TYPES = (tuple, list)

PLOTSPECS_LINE   = ['-',':','--','-.'] * (5*6//2)
PLOTSPECS_COLOR  = ['b','r','k','m','c'] * (4*6//2)
PLOTSPECS_MARKER = ['o','x','+','^','v','*'] * (4*5//2)

AXIS_TICKLABEL_FORMAT_KWARGS = dict(style='sci', scilimits=(-3,4), axis='y')

BARPLOT_HATCHING_PATTERNS = ['...', '+++', 'xxx', '///', '|||'] * (4*6//2)
BARPLOT_RELATIVE_SPACING_MAJOR = 0.25
BARPLOT_RELATIVE_SPACING_MINOR = 0.0

UNWANTED_CHRS_IN_FIGURE_NAME = (", ", ". ", ' ', '-', ',', '.')


def simplify_figure_name(name, lowercase=True,
    chrs_to_replace=UNWANTED_CHRS_IN_FIGURE_NAME):
    '''Replaces unwatned characters with underscores.'''
    for c in chrs_to_replace:
        name = name.replace(c, '_')
    if lowercase:
        return name.lower()
    else:
        return name


def plot_problem_domain(mesh:dolfin.Mesh=None,
                        domain_markers:dolfin.MeshFunction=None,
                        figname=None):

    fh = plt.figure(figname); fh.clear()
    ah = fh.add_subplot(111)

    if mesh is not None:
        if not isinstance(mesh, dolfin.Mesh):
            raise TypeError('Parameter `mesh`.')
        dolfin.plot(mesh)

    if domain_markers is not None:
        try:
            dolfin.plot(domain_markers)
        except:
            raise TypeError('Parameter `domain_markers` must be of type '
                            '`dolfin.MeshFunction`.')

    ah.set_xlabel('x (mm)')
    ah.set_ylabel('y (mm)')

    plt.axis('equal')
    plt.axis('image')

    # plt.tight_layout()

    return fh, simplify_figure_name(figname)


def plot_measurement_points(xk:np.ndarray,
                            uk:np.ndarray=None,
                            figname=None):

    if not isinstance(xk, np.ndarray):
        raise TypeError('Parameter `xk`.')

    if uk is not None:
        if not isinstance(uk, np.ndarray):
            raise TypeError('Parameter `uk`.')

        if xk.shape != uk.shape:
            raise TypeError('Parameter `xk` and `uk`.')

    fh = plt.figure(figname); fh.clear()
    ah = fh.add_subplot(111)

    if uk is not None:
        plt.quiver(xk[:,0], xk[:,1],
                   uk[:,0], uk[:,1],
                   np.sqrt((uk**2).sum(1)))
    else:
        plt.scatter(xk[:,0], xk[:,1], c=PLOTSPECS_COLOR[0])

    ah.set_xlabel('x (mm)')
    ah.set_ylabel('y (mm)')

    plt.axis('equal')
    plt.axis('image')

    # plt.tight_layout()

    return fh, simplify_figure_name(figname)


def plot_model_parameters_foreach(model_parameters_foreach,
                                  model_parameter_names=None,
                                  reference_times=None,
                                  figname=None):

    fh = plt.figure(figname); fh.clear()
    ah = fh.add_subplot(111)

    handles = []
    labels = []

    if reference_times is None:
        reference_times = range(len(model_parameters_foreach))

    for i, m_i in enumerate(np.array(model_parameters_foreach, order='F').T):
        plotspecs = PLOTSPECS_COLOR[i] + PLOTSPECS_MARKER[i]

        handles.extend(plt.plot(reference_times, m_i, plotspecs))

    if model_parameter_names is not None:
        labels.extend(model_parameter_names)

    ah.set_ylabel('Model parameter value, $m_i$')
    ah.set_xlabel('Observation time, $t$')
    ah.grid(True)

    if handles and labels:
        ah.legend(handles, labels)

    return fh, simplify_figure_name(figname)


def plot_model_parameters_forall(model_parameters_forall,
                                 model_parameter_names=None,
                                 figname=None):

    fh = plt.figure(figname); fh.clear()
    ah = fh.add_subplot(111)

    if model_parameter_names is None:
        model_parameter_names = list(range(0, len(model_parameters_forall)))


    ah.bar(model_parameter_names, model_parameters_forall, edgecolor='k')
    ah.set_ylabel('Model parameter value, $m_i$')
    ah.grid(True, axis='y')
    
    return fh, simplify_figure_name(figname)


def plot_model_cost(cost_values_final=None,
                    cost_values_initial=None,
                    reference_times=None,
                    figname=None):

    fh = plt.figure(figname); fh.clear()
    ah = fh.add_subplot(111); lh = []

    if cost_values_initial is not None:

        if reference_times is None: reference_times = range(len(cost_values_initial))
        lh.extend(plt.plot(reference_times, cost_values_initial, 'rx:'))

    if cost_values_final is not None:

        if reference_times is None: reference_times = range(len(cost_values_final))
        lh.extend(plt.plot(reference_times, cost_values_final, 'bo-'))

    # ah.set_title(figname)
    ah.ticklabel_format(**AXIS_TICKLABEL_FORMAT_KWARGS)
    ah.set_ylabel('Model cost, $J$')
    ah.set_xlabel('Observation time, $t$')
    ah.grid(True)

    if cost_values_final   is not None and \
       cost_values_initial is not None:
        ah.legend(lh, ['initial', 'final'])

    return fh, simplify_figure_name(figname)


def plot_cost_gradients(cost_gradients,
                        model_parameter_names=None,
                        reference_times=None,
                        figname=None):

    fh = plt.figure(figname); fh.clear()
    ah = fh.add_subplot(111); lh = []

    if reference_times is None:
        reference_times = range(len(cost_gradients))

    for i, DJDm_i in enumerate(np.array(cost_gradients, order='F').T):
        plotspecs = PLOTSPECS_COLOR[i] + PLOTSPECS_MARKER[i] + PLOTSPECS_LINE[i]
        lh.extend(plt.plot(reference_times, DJDm_i, plotspecs))

    ah.ticklabel_format(**AXIS_TICKLABEL_FORMAT_KWARGS)
    ah.set_ylabel('Model cost derivative, $DJ/Dm_i$')
    ah.set_xlabel('Observation time, $t$')
    ah.grid(True)

    if lh and model_parameter_names is not None:
        ah.legend(lh, model_parameter_names)

    return fh, simplify_figure_name(figname)


def plot_observation_misfit(error_observation,
                            reference_times=None,
                            figname=None,
                            ylabel=None):

    fh = plt.figure(figname); fh.clear()
    ah = fh.add_subplot(111)

    if reference_times is None:
        reference_times = range(len(error_observation))

    plt.plot(reference_times, error_observation, 'bo-')

    ah.ticklabel_format(**AXIS_TICKLABEL_FORMAT_KWARGS)
    ah.set_ylabel(ylabel)
    ah.set_xlabel('Observation time, $t$')
    ah.grid(True)

    return fh, simplify_figure_name(figname)


def plot_reaction_force_vs_displacement(
        reaction_force_observations,
        reaction_force_measurements,
        reaction_force_displacements,
        figname=None):

    f_obs = np.abs(reaction_force_observations)
    f_msr = np.abs(reaction_force_measurements)
    u_msr = np.abs(reaction_force_displacements)

    fh = plt.figure(figname); fh.clear()
    ah = fh.add_subplot(111); lh = []

    lh.extend(plt.plot(u_msr, f_msr, 'rx:'))
    lh.extend(plt.plot(u_msr, f_obs, 'bo-'))


    ah.ticklabel_format(**AXIS_TICKLABEL_FORMAT_KWARGS)
    ah.set_ylabel('Reaction force, $||f||$')
    ah.set_xlabel('Displacement, $||u||$')
    ah.legend(['measurement', 'observation'])
    ah.grid(True)

    # ah.set_title(figname)
    # plt.tight_layout()

    return fh, simplify_figure_name(figname)


def plot_model_parameter_sensitivities(
        sensitivities_dmdv_msr,
        model_parameter_names=None,
        reference_times=None,
        figname=None,
        ylabel=None,
        title=None):

    fh = plt.figure(figname); fh.clear()
    ah = fh.add_subplot(111)

    handles = []
    labels = []

    if not isinstance(sensitivities_dmdv_msr, np.ndarray):
        sensitivities_dmdv_msr = np.array(sensitivities_dmdv_msr, order='F')

    if sensitivities_dmdv_msr.ndim != 2:
        raise TypeError('Parameter `sensitivities_dmdv_msr` must be 2D '
                        'array-like whose first dimension corresponds to the '
                        'observation times, and the second dimension - the '
                        'model parameter sensitivity values.')

    number_of_observations, number_of_model_parameters = \
        sensitivities_dmdv_msr.shape

    if reference_times is None:
        reference_times = range(number_of_observations)

    elif len(reference_times) != number_of_observations:
        raise ValueError('Parameter `reference_times` (sequence) must have '
                         'length equal to the size of the first dimension of '
                         'parameter `sensitivities_dmdv_msr` (2D array-like).')

    if not isinstance(reference_times, np.ndarray):
        reference_times = np.array(reference_times)

    if reference_times.ndim != 1:
        raise TypeError('Parameter `reference_times` must be sequence-like.')

    if len(reference_times) > 1:
        width_of_all_bars = np.diff(reference_times).min()
    else:
        width_of_all_bars = 1

    width_of_all_bars -= width_of_all_bars*BARPLOT_RELATIVE_SPACING_MAJOR
    width_of_one_bar = width_of_all_bars / number_of_model_parameters

    bar_center_offsets = np.linspace(
        -width_of_all_bars*0.5 + width_of_one_bar*0.5,
         width_of_all_bars*0.5 - width_of_one_bar*0.5,
         number_of_model_parameters)

    width_of_one_bar -= width_of_one_bar*BARPLOT_RELATIVE_SPACING_MINOR

    for i, (dmdv_msr_i, bar_center_offset_i) in enumerate(
        zip(sensitivities_dmdv_msr.T, bar_center_offsets)):

        handles.append(ah.bar(
            reference_times+bar_center_offset_i,
            dmdv_msr_i, width_of_one_bar,
            edgecolor='k'))

    for bars_i, pattern_i in zip(handles, BARPLOT_HATCHING_PATTERNS):
        for patch_j in bars_i.patches:
            patch_j.set_hatch(pattern_i)

    if model_parameter_names is not None:
        labels.extend(model_parameter_names)

    ah.set_ylabel(ylabel)
    ah.set_xlabel('Observation time, $t$')
    ah.set_xticklabels(reference_times)
    ah.set_xticks(reference_times)
    ah.grid(True, axis='y')

    if handles and labels:
        ah.legend([h_i[0] for h_i in handles], labels)

    if title is not None:
        ah.set_title(title)

    return fh, simplify_figure_name(figname)


def plot_scalar_field(function, figname=None, title=None):

    if not isinstance(function, dolfin.Function):
        raise TypeError('Parameter `function` must be of type `dolfin.Function.`.')

    fh = plt.figure(figname); fh.clear()
    ah = fh.add_subplot(111)

    dolfin.plot(function)

    dim = len(function.ufl_shape)
    vec = function.vector().get_local()

    if dim > 0:
        delta_max = math.sqrt((vec**2).reshape((-1, dim)).sum(1).max())
    else:
        delta_max = np.abs(vec).max()

    # annotation = '$\delta_{max}$  = ' + f'{delta_max:.3g}'

    # plt.annotate(annotation,
    #     xy=(1, 0), xycoords='axes fraction',
    #     xytext=(-12, 12), textcoords='offset points',
    #     horizontalalignment='right',verticalalignment='bottom',
    #     bbox=dict(boxstyle="round", fc="w", ec="k", pad=0.3, alpha=0.75))

    ah.set_ylabel('y (mm)')
    ah.set_xlabel('x (mm)')

    if title is not None:
        ah.set_title(title)

    plt.axis('equal')
    plt.axis('image')

    # plt.tight_layout()

    return fh, simplify_figure_name(figname)
