'''
Some useful functions for setting up an inverse problem and manipulating output.

'''

import math
import dolfin
import numpy as np
import scipy.linalg as linalg

from dolfin import Constant
from dolfin import Function
from dolfin import Measure
from dolfin import assemble

from dolfin import dot
from dolfin import grad
from dolfin import inner

from ufl.core.expr import Expr as ufl_expr_t
from ufl.form import Form as ufl_form_t
from .invsolve import InverseSolver

from .config import logger

SEQUENCE_TYPES = (tuple, list)


def cost_displacement_misfit(u_obs, u_msr, dx_msr, subdims):
    '''Measure of the cost of the displacement misfit.

    Parameters
    ----------
    u_obs : dolfin.Function
        Observed displacement field.
    u_msr : dolfin.Expression like
        Measured displacement field.
    dx_msr : dolfin.Measure
        Integration measure.

    '''
    return sum((u_obs[i]-u_msr[i])**2*dx_msr for i in subdims)


def constraints_reaction_force(T_obs, T_msr, ds_msr, subdims):
    '''Constraint functions for the reaction force components.

    Parameters
    ----------
    T_obs : sequence of ufl.Form's
        Observed tractions.
    T_msr : sequence of ufl.Form's
        Measured tractions.
    subdims : sequence of int's
        Indices into force components.

    '''
    return tuple((T_obs[i]-T_msr[i])*ds_msr for i in subdims)


def cost_displacement_misfit_noisy(u_obs, u_msr, dx_msr, subdims=None):

    u_obs, u_msr, dx_msr = InverseSolver \
        ._std_init_args_u_obs_u_msr_dx_msr(u_obs, u_msr, dx_msr)

    assert isinstance( u_obs, Function)
    assert isinstance( u_msr, tuple)
    assert isinstance(dx_msr, tuple)

    assert all(isinstance( u_msr_i, ufl_expr_t) for  u_msr_i in  u_msr)
    assert all(isinstance(dx_msr_i,    Measure) for dx_msr_i in dx_msr)

    num_msr, dim_msr = len(u_msr), len(u_msr[0])

    if subdims is not None:
        subdims = InverseSolver._std_subdims_v_msr(subdims, num_msr, dim_msr)
    else:
        subdims = (range(dim_msr),) * num_msr

    V_obs = u_obs.function_space()
    if V_obs.dim() != u_obs.vector().size():
        V_obs = V_obs.collapse()

    du_msr = Function(V_obs) # Global perturbation
    u_msr = tuple(u_msr_i + du_msr for u_msr_i in u_msr)

    J = sum(cost_displacement_misfit(u_obs, u_msr_i, dx_msr_i, subdims_i)
            for u_msr_i, dx_msr_i, subdims_i in zip(u_msr, dx_msr, subdims))

    return J, du_msr


def constraints_reaction_force_noisy(T_obs, T_msr, ds_msr, subdims=None):

    T_obs, T_msr, ds_msr = InverseSolver \
        ._std_init_args_T_obs_T_msr_ds_msr(T_obs, T_msr, ds_msr)

    if not (len(T_obs) == len(T_msr) == len(ds_msr) == 1):
        raise TypeError('Parameters `T_obs`, `T_msr`, `ds_msr`.')

    T_obs, T_msr, ds_msr = T_obs[0], T_msr[0], ds_msr[0]

    assert isinstance( T_obs, tuple)
    assert isinstance( T_msr, tuple)
    assert isinstance(ds_msr, tuple)

    assert len(T_obs) == len(T_msr) == len(ds_msr)

    assert all(isinstance( T_obs_i, ufl_expr_t) for  T_obs_i in  T_obs)
    assert all(isinstance( T_msr_i, ufl_expr_t) for  T_msr_i in  T_msr)
    assert all(isinstance(ds_msr_i,    Measure) for ds_msr_i in ds_msr)

    n_msr, n_dim = len(T_msr), len(T_msr[0])

    if subdims is None:
        subdims = range(n_dim)
    elif isinstance(subdims, int):
        if not (-n_dim <= subdims < n_dim):
            raise ValueError('Parameter `subdims`.')
        subdims = (subdims,)
    elif isinstance(subdims, SEQUENCE_TYPES):
        if not all(isinstance(subdim_i, int) for subdim_i in subdims):
            raise TypeError('Parameter `subdims`.')
        if not all(-n_dim <= subdim_i < n_dim for subdim_i in subdims):
            raise ValueError('Parameter `subdims`.')
        pass
    else:
        raise TypeError('Parameter `subdims`.')

    # dT_msr = [Constant(0.0) for _ in subdims]
    dT_msr = Constant([0.0]*len(subdims))
    # dT_msr = Constant([0.0]*n_dim)

    T_msr = [[T_msr_i[subdim_j] + dT_msr[j]
             for j, subdim_j in enumerate(subdims)]
             for T_msr_i in T_msr]

    T_obs = [[T_obs_i[subdim_j]
             for subdim_j in subdims]
             for T_obs_i in T_obs]

    subdims = range(len(subdims))

    C = [constraints_reaction_force(T_obs_i, T_msr_i, ds_msr_i, subdims)
         for T_obs_i, T_msr_i, ds_msr_i in zip(T_obs, T_msr, ds_msr)]

    C = [sum(C[j][i] for j in range(n_msr)) for i in subdims]

    return C, dT_msr


def noramalizing_weight(form_denominator):
    '''Return an expression of a fraction whose denominator is a `dolfin.Constant`
    that can be assigned the value of the integral of the form `form_denominator`.

    Parameters
    ----------
    form_denominator : ufl.Form
        Integral expression of the denominator.

    Returns
    -------
    ufl.algebra.Division
        An expression of the weight.
    function(t_ref)
        Function that will update the weight (`dolfin.Constant`) for a given
        reference key `t_ref`. Parameter `t_ref` is merely used for memoizing
        previous values of the weights that have already been assembled.

    Examples
    --------
    >>> form_denominator = u_msr**2 * dx_msr
    >>> form_denominator = dolfin.sqrt(T_msr**2) * ds_msr

    '''

    if not isinstance(form_denominator, Form):
        raise TypeError('Parameter `form_denominator` '
                        'must be of type `ufl.Form`.')

    denominator = Constant(1.0)
    weight = 1.0 / denominator

    recompute_weight = factory_recompute_variables(
        vars=denominator, forms=form_denominator)

    return weight, recompute_weight


def factory_recompute_variables(vars, forms):
    '''Assign values of integrals (forms) to variables.

    Parameters
    ----------
    vars : dolfin.Constant, or a list or tuple of dolfin.Constant's
    forms : ufl.Form, or a list or tuple of ufl.Form's

    Important
    ---------
    All previously assembled forms are memoized. This is computationally
    efficient but can be problematic if a coefficient in a form is updated.
    In this case, the cached values of the assembled forms must be cleared.
    Call the returned function's attribute method `clear_cached` to do this.

    '''

    if not isinstance(vars, SEQUENCE_TYPES): vars = (vars,)
    if not isinstance(forms, SEQUENCE_TYPES): forms = (forms,)

    if not all(isinstance(v, Constant) for v in vars):
        raise TypeError('Expected `vars` to contain `dolfin.Constant`s.')

    if not all(isinstance(f, ufl_form_t) for f in forms):
        raise TypeError('Expected `forms` to contain `ufl.Form`s.')

    # Memoize assembled
    assembled_forms = {}

    def recompute_variables(t_ref=None):
        '''Update variables `vars` (of type `dolfin.Constant`) by evaluating
        `forms` (of type `ufl.Form`). Parameter `t_ref` is used to make a key
        identifier for the value. The value can then be reused later. Note, if
        `t_ref` is `None`, the form is reevaluated. Static method `clear_cached`
        can be called to clear the values of the previously assembled forms.
        '''

        if t_ref is not None:

            for v, f in zip(vars, forms):
                k = (t_ref, id(v))

                if k in assembled_forms:
                    value = assembled_forms[k]
                else:
                    value = assemble(f)
                    assembled_forms[k] = value

                v.assign(value)

        else:

            for v, f in zip(vars, forms):
                v.assign(assemble(f))

    # Method for clearing all previously assembled form values
    recompute_variables.clear_cached = assembled_forms.clear

    return recompute_variables


def project_expression(expr, V, cell_indices=None, method="project"):
    '''Project expression onto a function space.

    Parameters
    ----------
    expr : dolfin.Expression-like
        Expression to be projected.
    cell_indices : sequence of int's
        Cell indices defining the subdomain where the expression `expr` will
        be projected. Outside this subdomain, the value of the projection will
        be set to zero.

    Returns
    -------
    func : dolfin.Function
        Projection of the expression.

    '''

    if not isinstance(expr, ufl_expr_t):
        raise TypeError('Parameter `expr`.')

    if method == "interpolate":
        func = dolfin.interpolate(expr, V)
    elif method == "project":
        func = dolfin.project(expr, V)
    else:
        raise ValueError('Parameter `method` must be one '
                         'of "interpolate" or "project"')

    if cell_indices is not None:

        dof_mask = np.ones((V.dim(),), bool)
        get_cell_dofs = V.dofmap().cell_dofs

        for i in cell_indices:
            dof_mask[get_cell_dofs(i)] = False

        func.vector()[np.flatnonzero(dof_mask)] = 0.0

    return func


def project_subdomain_stresses(subdomain_stresses, function_space,
                               subdomain_markers, subdomain_ids):
    '''
    Parameters
    ----------
    subdomain_stresses : sequence of dolfin expression-like objects.
    function_space : dolfin.FunctionSpace
    subdomain_markers : dolfin.MeshFunctionSizet
    subdomain_ids : sequence of (sequences of) int's

    Returns
    -------
    dolfin.Function

    '''

    if not isinstance(subdomain_stresses, SEQUENCE_TYPES):
        subdomain_stresses = (subdomain_stresses,)

    if not isinstance(subdomain_ids, SEQUENCE_TYPES):
        subdomain_ids = (subdomain_ids,)

    if len(subdomain_stresses) != len(subdomain_ids):
        raise TypeError('Parameters `subdomain_stresses` and '
                        '`subdomain_markers` must have same length.')

    if type(subdomain_markers).__name__ != 'MeshFunctionSizet':
        raise TypeError('Parameter `subdomain_markers` must be '
                        'a `dolfin.MeshFunctionSizet` that marks '
                        'the subdomain of each cell in the mesh.')

    subdomain_markers = subdomain_markers.array()

    if __debug__:

        unique_subdomain_ids = []

        for subdomain_ids_i in subdomain_ids:
            if isinstance(subdomain_ids_i, SEQUENCE_TYPES):
                unique_subdomain_ids.extend(subdomain_ids_i)
            else:
                unique_subdomain_ids.append(subdomain_ids_i)

        if sorted(set(unique_subdomain_ids)) != np.unique(subdomain_markers).tolist():
            raise RuntimeError('Parameter `subdomain_ids` must contain the same '
                               'subdomain id\'s as defined by `subdomain_markers`.')

    subdomain_stress_pojections = []

    for subdomain_stress_i, subdomain_ids_i in \
            zip(subdomain_stresses, subdomain_ids):

        if isinstance(subdomain_ids_i, SEQUENCE_TYPES):
            cell_mask = subdomain_markers == subdomain_ids_i[0]

            for subdomain_ids_ij in subdomain_ids_i[1:]:
                cell_mask += subdomain_markers == subdomain_ids_ij

        else:
            cell_mask = subdomain_markers == subdomain_ids_i

        subdomain_stress_pojections.append(
            project_expression(subdomain_stress_i, function_space,
                cell_indices=np.flatnonzero(cell_mask), method="project"))

    global_stress_projection = Function(function_space)
    global_stress_vector = global_stress_projection.vector()

    for stress_projection_i in subdomain_stress_pojections:
        global_stress_vector[:] += stress_projection_i.vector()

    return global_stress_projection


def project_sensitivities_dmdu_msr(dmdu_msr_discrete, V_msr,
    apply_smoothing=True, kappa=None, *, smoothing_solver=None):
    '''Compute the model parameter sensitivity fields with respect to
    the displacement field measurements for the observation time `t`.

    Parameters
    ----------
    superpose : bool (optional)
        Wheather to superpose all measurements onto a single function.

    Returns
    -------
    sensitivity_functions : list of lists of dolfin.Function's
        Vector fields of the model parameter sensitivities with respect to
        the displacement field measurements over the measurement subdomains.
        Note, the value at `sensitivity_functions[I][J]` corresponds to the
        `I`th measurement subdomain, and the `J`th model parameter sensitivity.

    '''

    if not isinstance(dmdu_msr_discrete, np.ndarray):
        raise TypeError('Parameter `dmdu_msr_discrete`.')

    if dmdu_msr_discrete.ndim != 2:
        if dmdu_msr_discrete.ndim != 1:
            raise TypeError('Parameter `dmdu_msr_discrete`.')
        dmdu_msr_discrete = dmdu_msr_discrete[None,:]

    if not isinstance(V_msr, dolfin.FunctionSpace):
        raise TypeError('Parameter `V_msr`.')

    if dmdu_msr_discrete.shape[-1] != V_msr.dim():
        raise TypeError('Parameter `dmdu_msr_discrete` has bad shape.')

    dmdu_msr_projected = []

    for dmIdu_msr in dmdu_msr_discrete:
        dmIdu_msr_projected = Function(V_msr)
        dmIdu_msr_projected.vector()[:] = dmIdu_msr
        dmdu_msr_projected.append(dmIdu_msr_projected)

    if smoothing_solver is not None:
        if not hasattr(smoothing_solver, "solve"):
            raise TypeError("Keyword argument `smoothing_solver` "
                            "must have method `solve`.")
        apply_smoothing = True

    if apply_smoothing:

        if smoothing_solver is None:
            smoothing_solver = make_smoothing_solver(V_msr, kappa)

        solve = smoothing_solver.solve

        for f, a in zip(dmdu_msr_projected, dmdu_msr_discrete):

            x = f.vector()
            solve(x, x.copy())
            f.vector()[:] = x

    return dmdu_msr_projected


def make_smoothing_solver(V, kappa=None):
    '''
    Important
    ---------
    * Smoothing is most effective when the sensitivities are defined over
      the whole domain. In practice, this is rarely the case.
    * Smoothing works well enough when the sensitivities are defined over
      a subdomain but that is the same dimension as the mesh. In such a case,
      the smoothing will be relatively poor on the boundary of the subdomain.
    * Smoothing does not work well when the sensitivities are defined over a
      boundary of the domain. More generally, when the sensitivity domain
      is lower dimension than the mesh. In this case, this type of smoothing
      is quite useless.

    Returns
    -------
    smoothing_solver:
        Returns the solver that solves the smoothing problem that is the linear
        system `M x_smoothed = x` where `x` is the vector of unsmoothed values.
        The smoothing solver can be invoked as `smoothing_solver.solve(x, x)`.

    '''

    v0 = dolfin.TestFunction(V)
    v1 = dolfin.TrialFunction(V)

    a = dot(v1,v0)*dolfin.dx

    if kappa is not None and float(kappa) != 0.0:

        if isinstance(kappa, (float, int)):
            kappa = Constant(kappa)

        a += kappa*inner(grad(v1), grad(v0))*dolfin.dx

    smoothing_solver = dolfin.LUSolver(assemble(a), "mumps")
    smoothing_solver.parameters["symmetric"] = True

    return smoothing_solver


def test_projected_sensitivities_dmdu_msr(
        dmdu_msr_projected, dmdu_msr_discrete, rtol=1e-5, atol=1e-8):

    if not isinstance(dmdu_msr_projected, SEQUENCE_TYPES) or \
       not all(isinstance(f, dolfin.Function) for f in dmdu_msr_projected):
        raise TypeError('Parameter `dmdu_msr_projected`.')

    if not isinstance(dmdu_msr_discrete, np.ndarray) or dmdu_msr_discrete.ndim != 2:
        raise TypeError('Parameter `dmdu_msr_discrete`.')

    n_m, n_msr_dofs = dmdu_msr_discrete.shape
    V_prj = dmdu_msr_projected[0].function_space()

    if n_m != len(dmdu_msr_projected):
        raise ValeError('Parameter `dmdu_msr_projected`.')

    if n_msr_dofs != V_prj.dim():
        raise ValeError('Parameter `dmdu_msr_projected`.')

    du_msr_arr = np.random.randn(n_msr_dofs)
    du_msr_fun = dolfin.Function(V_prj)
    du_msr_fun.vector()[:] = du_msr_arr

    dm_expected = dmdu_msr_discrete.dot(du_msr_arr)

    dm_predicted = np.array([
        dolfin.assemble(dolfin.dot(dmIdu_msr, du_msr_fun)*dolfin.dx)
        for dmIdu_msr in dmdu_msr_projected])

    success = np.allclose(dm_expected, dm_predicted, rtol, atol)

    results = {
        'dm_expected': dm_expected,
        'dm_predicted': dm_predicted,
        }

    return success, results
