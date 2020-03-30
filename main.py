from dolfin import *
import dolfin

import time
import os
import sys
import logging
import importlib
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from importlib import reload
from pprint import pprint

import config

from invsolve import project
from invsolve import measure
from invsolve import invsolve

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# dolfin.set_log_level(INFO)
# dolfin.set_log_active(True)

reload = importlib.reload

# =============================================================
# *************************************************************
# =============================================================

### List of meshes

Meshes = [540, 830, 1300, 1900, 3000, 4000, 6000, 9000, 12000, 22000, 44000]


### Problem Configuration Parameters

DUMMY_DATA = True
FIX_BOUNDARY = False
PLOT_RESULTS = True

SENSITIVITY_SNAPSHOTS = False
WEAKLY_NON_LINEAR = True
HIGHLY_NON_LINEAR = False

if SENSITIVITY_SNAPSHOTS:
    assert WEAKLY_NON_LINEAR != HIGHLY_NON_LINEAR
    if WEAKLY_NON_LINEAR:
        print('Sensitiviy study with respect to the number of snapshots ...')
        import sensitiviy_study_over_snapshots_weakly_nonlinear
        print('Sensitiviy study with respect to the number of snapshots is done!')
        raise
    else:
        print('Sensitiviy study with respect to the number of snapshots ...')
        import sensitiviy_study_over_snapshots_highly_nonlinear
        print('Sensitiviy study with respect to the number of snapshots is done!')
        raise

SENSITIVITY_MESHES = False
REFERENCE_MESH = False
OPTIMIZED_MESH = True

if SENSITIVITY_MESHES:
    if REFERENCE_MESH:
        print('Sensitiviy study with respect to element size (projection on reference mesh)...')
        import sensitiviy_study_over_meshes_reference
        print('Sensitiviy study with respect to to element size is done (projection on reference mesh)!')
        raise
    if OPTIMIZED_MESH:
        print('Sensitiviy study with respect to element size (projection on optimized mesh)...')
        import sensitiviy_study_over_meshes_optimized
        print('Sensitiviy study with respect to to element size is done (projection on optimized mesh)!')
        raise

COSTS_FOR_EACH_ITERATION_STUDY = True   # Enable the the same condition inside
                                        # the file forward_fem_solver_mesh_ref
# Only on optimization process must be enabled
OPTIMIZATION_ON_DISPLACEMENT_ONLY = False
OPTIMIZATION_ON_FORCE_ONLY = False
OPTIMIZATION_CONSTRAINED = True # the opperationnel objective function

if (OPTIMIZATION_ON_DISPLACEMENT_ONLY + OPTIMIZATION_ON_FORCE_ONLY\
    + OPTIMIZATION_CONSTRAINED > 1):
        raise RuntimeError('Choose only one optimization function!')

m_initial_further = [0.01, 0.01,
                     0.01, 0.01]
if OPTIMIZATION_CONSTRAINED:
    m_initial_further.append(1e-04)

#m_initial_nearer = [0.049, 0.19,
#                    0.015, 0.39]
m_initial_nearer = [0.01, 0.01,
                    0.01, 0.01]

if OPTIMIZATION_CONSTRAINED:
    m_initial_nearer.append(1e-04)

if COSTS_FOR_EACH_ITERATION_STUDY:
    Meshes = [44000,]


### Problem Specific Imports

### Get measurements

if DUMMY_DATA:
    import forward_fem_solver_mesh_ref

    u_msr_dom_func_original = forward_fem_solver_mesh_ref.out['u_msr_dom_func']
    # du_msr_dom_func = keloid_skin_test.out['du_msr_dom_func']

    u_msr_dom_vals = None
    x_msr_dom_vals = None

    ux_msr_pad_vals = forward_fem_solver_mesh_ref.out['ux_msr_pad_left_vals']
    # uy_msr_pad_vals = keloid_skin_test.out['uy_msr_pad_left_vals']

    fx_msr_pad_vals = forward_fem_solver_mesh_ref.out['fx_msr_pad_left_vals']
    # fy_msr_pad_vals = keloid_skin_test.out['fy_msr_pad_left_vals']

    # df_msr_pad_vals = keloid_skin_test.out['df_msr_pad_left_vals']
    print (fx_msr_pad_vals)
    print (ux_msr_pad_vals)
    # input ("pause ...")

if DUMMY_DATA:
	msr_pad_one_ux = ux_msr_pad_vals
	msr_pad_one_fx = fx_msr_pad_vals


### Reload some module if needed

reload(config)
reload(invsolve.config)


### Define the Material Model

def grad_reduc(X):
	# Transform the deformation gradient tenso 'F' to a 3D tensor
	e = grad(X)
	return as_tensor([[e[0, 0], e[0, 1], 0],
                      [e[1, 0], e[1, 1], 0],
                      [0, 0, 0]])

def dim_reduc(X):
	# Transform a 3D tensor to 2Dmsr_pad_one_fx_
	return as_tensor([[X[0, 0], X[0, 1]],
                      [X[1, 0], X[1, 1]]])


def Psi_(u, material_parameters):
    '''Strain energy density'''

    I = Identity(3)

    F = variable(I + grad_reduc(u))

    C = F.T*F
    # E = 0.5*(C-I)
    B = F*F.T
    J = det(F)

    I1 = tr(C)
    I2 = 0.5*(tr(C)**2 - tr(C*C))
    I3 = det(C)
    IB = tr(B)

    mu = material_parameters['mu']
    jm = material_parameters['jm']

    psi = -0.5*mu*(jm*ln(1 - (IB - 3)/jm) + 2*ln(J)) # Gent compressible

    PK1 = diff(psi, F)
    PK2 = dot(inv(F), PK1)

    return psi, PK1, PK2

def Pi_(u, mp_sub, dx_sub): # f_msr, dx_msr_f):
    '''Potential energy

    Parameters
    ----------
    u : dolfin.Function
        The displacement field, a vector values function.
    mp_sub : iterable of dict's whose values are dolfin.Constant's
        Material parameters for each material subdomain.
    dx_sub: iterable of dolfin.Measure's
        Material integration subdomains.

    Returns
    -------
    Pi : ufl.Form
        The potential energy of the hyper-elastic solid

    '''

    W_int = W_ext = 0

    # deformation energy
    for mp_sub_i, dx_sub_i in zip(mp_sub, dx_sub):
        psi, *_ = Psi_(u, mp_sub_i)
        W_int += psi * dx_sub_i

    # # external load potential
    # for f_msr_i, dx_msr_i in zip(f_msr, dx_msr_f):
    #     W_ext += dot(u, f_msr_i) * dx_msr_i

    Pi = W_int - W_ext

    return Pi


### Define the Cost Functional

def J_u_(u, u_msr, dx_msr, w_msr=None):
    '''Cost functional.

    J^{(k)}(u^{(k)}, u_msr^{(k)}) :=
        \frac{1}{2} \int_{\Gamma_\mathrm{msr}} (u^{(k)} - u_msr^{(k)})^2 \, dx

    The weights can be precomputed
    This weight needs to be a dolfin.Constant that is assigned a pre-computed
    value for each time. This can be done in the `observatio_setter`.

    On the other hand, the weight must be differnetiable integral wrt some
    parameter, (most likely measurement) so, the weight should be a ufl.Form.

    What if the weight is not a ufl.Form; maye it's just a constant that
    depends on the measurement, like a quantity of interset.

    Parameters
    ----------
    u : dolfin.Function
        Model solution, e.g. displacement field
    u_msr : iterable of dolfin.Expression's
        Expermental measurements, e.g. 1DIC measurements
    dx_msr : iterable of dolfin.Measure's
        Integration domain, e.g. DIC measurement surface

    Returns
    -------
    J : ufl.Form
        The measure of the model cost, a cost functional.

    '''

    J = 0

    if w_msr is None:
        w_msr = (1,) * len(dx_msr)

    for w_msr_i, u_msr_i, dx_msr_i in zip(w_msr, u_msr, dx_msr):
        J += (u - u_msr_i)**2 / w_msr_i * dx_msr_i

    return J


### Define the Constraint Equation

def C_fx_(f, f_msr, dx_msr, w_msr=None):
    '''Constraint equation to impose on cost functional.

    Reaction force in x-direction.

    '''

    C = 0

    if w_msr is None:
        w_msr = (1,) * len(dx_msr)

    for w_msr_i, f_msr_i, dx_msr_i in zip(w_msr, f_msr, dx_msr):
        C += (f[0] - f_msr_i[0]) / w_msr_i * dx_msr_i
        # C += (f[0] - f_msr_i[0])**2 / w_msr_i * dx_msr_i

    return C

def C_f_(f, f_msr, dx_msr, w_msr=None):
    '''Constraint equation to impose on cost functional.

    Net reaction force.

    '''

    C = 0

    if w_msr is None:
        w_msr = (1,) * len(dx_msr)

    for w_msr_i, f_msr_i, dx_msr_i in zip(w_msr, f_msr, dx_msr):
        # C += (sqrt(f**2)-sqrt(f_msr_i**2)) / w_msr_i * dx_msr_i
        C+=1
    return C


### Printing converging paramters for each standard variation set
result_parameters_file = open('identified_parameters_with_all_meshes.txt', 'w')


### Import mesh
for Mesh in Meshes:
    mesh_file = importlib.import_module("h_p_convergence.keloid_skin_mesh_" +
                                          str(Mesh))
    mesh_domain = mesh_file.mesh_domain
    markers_domain = mesh_file.markers_domain
    markers_boundary = mesh_file.markers_boundary
    id_markers_domain = mesh_file.id_markers_domain
    id_markers_boundary = mesh_file.id_markers_boundary

    print('MESH:', Mesh)

    ### Integration domains

    dx_material = [
        dolfin.Measure('dx',
            domain=mesh_domain,
            subdomain_data=markers_domain,
            subdomain_id=(
                id_markers_domain['keloid_measure'],
                )
            ),
        dolfin.Measure('dx',
            domain=mesh_domain,
            subdomain_data=markers_domain,
            subdomain_id=(
                id_markers_domain['healthy'],
                id_markers_domain['healthy_measure'],
                )
            ),
        ]

    dx_measure = [
        dolfin.Measure('dx',
            domain=mesh_domain,
            subdomain_data=markers_domain,
            subdomain_id=(
                id_markers_domain['keloid_measure'],
                id_markers_domain['healthy_measure'],
                )
            ),
        ]

    ds_boundary_pad_one = dolfin.Measure('ds',
        domain=mesh_domain,
        subdomain_data=markers_boundary,
        subdomain_id=(id_markers_boundary['pad_one_sensor'],))

    ds_measure = [ds_boundary_pad_one]

    dx_mat = dx_material
    dx_msr_dom = dx_measure
    ds_msr_pad = ds_measure

    ### To check if the external pad is well identified.
    ds_boundary_pad_one_external = dolfin.Measure('ds',
        domain=mesh_domain,
        subdomain_data=markers_boundary,
        subdomain_id=(id_markers_boundary['pad_one'],))

    print('Sensor pad surface integration length', dolfin.assemble(1*ds_msr_pad[0]))
    print('External pad perimeter', dolfin.assemble(1*ds_boundary_pad_one_external))

    #if DUMMY_DATA:
    #    logger.warning('Assuming measurement domain to be the material domain.')
    #    dx_msr_dom = dx_mat


    # ------------------------------------------------------------------------
    # Lagrange 1
    # ----------

    INTERPOLATION_DEGREE = 1 # Interpolation Lagrange 1

    if COSTS_FOR_EACH_ITERATION_STUDY:
        INTERPOLATION_DEGREE = 1

    ### Function spaces

    V = VectorFunctionSpace(mesh_domain, 'CG', INTERPOLATION_DEGREE)
    V_msr_u = VectorFunctionSpace(mesh_domain, 'CG', INTERPOLATION_DEGREE)
    ### Dirichlet Boundary Conditions

    bcs = []

    uD_msr_pad_one = Expression(('ux','uy'), ux=0.0, uy=0.0, degree=0)
    uD_msr_pad_two = Expression(('ux','uy'), ux=0.0, uy=0.0, degree=0)

    bcs = [DirichletBC(V, uD_msr_pad_one, markers_boundary,
                        id_markers_boundary['pad_one']),
           DirichletBC(V, uD_msr_pad_one, markers_boundary,
                        id_markers_boundary['pad_one_sensor']),
           DirichletBC(V, uD_msr_pad_two, markers_boundary,
                        id_markers_boundary['pad_two'])
           ]

    EPS_DOLFIN = 1e-14
    def bottom_boundary(x, on_boundary):
    	return on_boundary and near(x[1], -5, EPS_DOLFIN)

    def top_boundary(x, on_boundary):
    	return on_boundary and near(x[1], 45, EPS_DOLFIN)

    def left_boundary(x, on_boundary):
    	return on_boundary and near(x[0], 0, EPS_DOLFIN)

    def right_boundary(x, on_boundary):
    	return on_boundary and near(x[0], 100, EPS_DOLFIN)

    if FIX_BOUNDARY:

        uD_x = Constant(0.0)
        uD_y = Constant(0.0)

        V_x, V_y = V.split()

        bcs.extend([
            DirichletBC(V_y, uD_y, bottom_boundary),
            DirichletBC(V_y, uD_y, top_boundary),
            #DirichletBC(V_x, uD_x, left_boundary),
            #DirichletBC(V_x, uD_x, right_boundary),
            ])

    ### Project generated data on identification mesh

    u_msr_dom_func = []

    for u_field in u_msr_dom_func_original:
        u_msr_dom_func.append(dolfin.project(u_field, V_msr_u))

    ### Create Measurement Expressions from Data

    if DUMMY_DATA:
        n_msr_dic = len(u_msr_dom_func)
        n_msr_pad = len(fx_msr_pad_vals)

    assert n_msr_dic == n_msr_pad

    n_msr = n_msr_dic
    t_msr = tuple(range(0,n_msr))

    if not DUMMY_DATA:
        u_msr = project.project_pointvalues_on_functions(
        xk=msr_dic_window_xk, fk=msr_dic_window_uk, V_project=V_msr_u,
        meshless_degree=0, num_neighbors=None, distance_norm=2)
        u_msr_as_functions = u_msr

    if DUMMY_DATA:
        u_msr = u_msr_dom_func


    # msr_pad_one_fx = np.take(np.array(fx_msr_pad_vals),0,1)
    msr_pad_one_fx = np.array(fx_msr_pad_vals)
    assert msr_pad_one_fx.ndim == 1
    f_msr = np.zeros((n_msr_pad, 2), float)

    f_msr[:,0] = msr_pad_one_fx # no y-component
    # input('success')
    # NOTE: Devide the reaction force by the pad perimiter to get average traction.
    # Integrating this traction along the pad perimiter will give the net force.
    f_msr /= assemble(1*ds_measure[0])

    u_msr_dic = measure.MeasurementExpression(u_msr, t_msr, degree=2)
    f_msr_pad = measure.MeasurementExpression(f_msr, t_msr, degree=0)

    # place similar measurements in some containers
    u_msr = [u_msr_dic,]
    f_msr = [f_msr_pad,]


    ### Add dummy noise to measurement `u_msr`

    # call to construct a new noise bases for `u_msr`
    # noise_basis = lambda : measure.MeasurementNoise(Function(V_msr_u), degree=2)

    noise_delta_u_msr = [Constant(0) for _ in u_msr]
    # noise_basis_u_msr = [noise_basis() for _ in u_msr]

    # du_msr_noise = [d*v for d,v in zip(noise_delta_u_msr, noise_basis_u_msr)]
    # u_msr_noisy = [u + du for u, du in zip(u_msr, du_msr_noise)]


    ### Add dummy noise to measurement `f_msr`

    # call to construct a new noise bases for df_msr
    # noise_basis = lambda : measure.MeasurementNoise(Constant((-1.0,0.0)), degree=0)

    noise_delta_f_msr = [Constant(0) for _ in f_msr]
    # noise_basis_f_msr = [noise_basis() for _ in f_msr]

    # df_msr_noise = [d*v for d,v in zip(noise_delta_f_msr, noise_basis_f_msr)]
    # f_msr_noisy = [f + df for f, df in zip(f_msr, df_msr_noise)]


    ### Weights for Normalizing Cost Terms
    # i.e. the weights will go in the denominator

    # TODO: Need an abstraction

    # Weight as a `ufl.Form`
    # NOTE: can be differentiated with respect to a `dummy_delta_*`
    # NOTE: shall not multiply anyother `ufl.Form`

    eps_w_msr_dic = Constant(1e-4) # ensure positive denominator in cost
    eps_w_msr_pad = Constant(1e-4) # ensure positive denominator in cost

    form_w_msr_dic = [ (eps_w_msr_dic + u_msr_i**2) * dx_msr_i
        for u_msr_i, dx_msr_i in zip(u_msr, dx_measure)]

    form_w_msr_pad = [ (eps_w_msr_pad + sqrt(f_msr_i**2)) * dx_msr_i
        for f_msr_i, dx_msr_i in zip(f_msr, ds_measure)]

    form_dwdu_msr_dic = [ diff(w_msr_i, d_msr_i)
        for w_msr_i, d_msr_i in zip(form_w_msr_dic, noise_delta_u_msr)]

    form_dwdf_msr_pad = [ diff(w_msr_i, d_msr_i)
        for w_msr_i, d_msr_i in zip(form_w_msr_pad, noise_delta_f_msr)]

    # Weight as a `Constant` variable
    # NOTE: can left-multiply a `ufl.Form`
    # NOTE: can not be differentiated with respect to a `dummy_delta_*`
    # NOTE: values must be assigned inside `measurement_setter`, e.g.
    #   var_w_msr_dic[i].assign(assemble(form_w_msr_dic[i]))
    #   var_w_msr_pad[i].assign(assemble(form_w_msr_pad[i]))

    var_w_msr_dic = [Constant(0.0) for _ in dx_measure]
    var_w_msr_pad = [Constant(0.0) for _ in ds_measure]

    var_dwdu_msr_dic = [Constant(0.0) for _ in dx_measure]
    var_dwdf_msr_pad = [Constant(0.0) for _ in ds_measure]


    ### Model

    u = Function(V)

    if FIX_BOUNDARY:
        material_parameters = [
            {'mu': Constant(0),
             'jm': Constant(0)},
            {'mu': Constant(0),
             'jm': Constant(0)}]
    else:
        material_parameters = [
            {'mu': Constant(0),
             'jm': Constant(0)},
            {'mu': Constant(0),
             'jm': Constant(0)}]

    Pi = Pi_(u, material_parameters, dx_material)


    ### Model Cost

    # J_u = J_u_(u, u_msr_noisy, dx_measure, var_w_msr_dic)
    J_u = J_u_(u, u_msr, dx_measure, var_w_msr_dic)

    ### Model Cost Constraint

    # NOTE:
    # T = dot(P,N) # numerical force
    # R = f_msr[0] # measured reaction

    N = FacetNormal(mesh_domain)

    psi_keloid, P_keloid, S_keloid = Psi_(u, material_parameters[0])
    psi_healthy, P_healthy, S_healthy = Psi_(u, material_parameters[1])

    f = dolfin.dot(dim_reduc(P_keloid), N)

    # C_f = C_fx_(f, f_msr_noisy, ds_measure, var_w_msr_pad)
    C_f = C_fx_(f, f_msr, ds_measure, var_w_msr_pad)
    # C_f = C_f_(f, f_msr_noisy, ds_measure, var_w_msr_pad)

    constraint_multiplier = Constant(-1e-6)
    J_f = constraint_multiplier * C_f


    ### Inverse Solver Arguments

    class model:
        u = u
        Pi = Pi
        bcs = bcs

    if OPTIMIZATION_CONSTRAINED:

        model_cost = J_u + J_f

        model_parameters = [
            material_parameters,
            constraint_multiplier]

    elif OPTIMIZATION_ON_DISPLACEMENT_ONLY:

        model_cost = J_u

        model_parameters = [material_parameters,]

    else: # OPTIMIZATION_ON_FORCE_ONLY = True
        model_cost = C_f

        model_parameters = [material_parameters,]

    observation_times = t_msr


    previously_assembled_forms = {}
    def measurement_setter(t=None):
        '''This function will be called inside the `InverseSolver` for each
        solution time `t`. The purpose of this function is to set the values
        of the measurements.
        '''

        if t is None: return
        if t == -1: t = t_msr[-1]

        # set dirichlet BC to measurement
        uD_msr_pad_one.ux = ux_msr_pad_vals[t]
        print(uD_msr_pad_one.ux)

        # set displacement measurement
        if isinstance(t, int):
            for u_msr_i in u_msr:
                u_msr_i.set_measurement_index(t)
                print('set displacement measurement done for t=', t)
        else:
            for u_msr_i in u_msr:
                u_msr_i.set_measurement_time(t)
                print('set displacement measurement done for t=', t)

        if isinstance(t, int):
            for f_msr_i in f_msr:
                f_msr_i.set_measurement_index(t)
                print('set force measurement done for t=', t)
        else:
            for f_msr_i in f_msr:
                f_msr_i.set_measurement_time(t)
                print('set force measurement done for t=', t)

        # TODO: This needs to be precomputed

        # set cost weights for the displacement measurement
        for var_w_i, form_w_msr_i in zip(var_w_msr_dic, form_w_msr_dic):

            k = (id(var_w_i), t)
            if k in previously_assembled_forms:
                assemble_form_w_msr_i = previously_assembled_forms[k]
            else:
                assemble_form_w_msr_i = assemble(form_w_msr_i)
                previously_assembled_forms[k] = assemble_form_w_msr_i

            # assemble_form_w_msr_i = assemble(form_w_msr_i)
            var_w_i.assign(assemble_form_w_msr_i)

        # set cost weights for the force measurement
        for var_w_i, form_w_msr_i in zip(var_w_msr_pad, form_w_msr_pad):

            k = (id(var_w_i), t)
            if k in previously_assembled_forms:
                assemble_form_w_msr_i = previously_assembled_forms[k]
            else:
                assemble_form_w_msr_i = assemble(form_w_msr_i)
                previously_assembled_forms[k] = assemble_form_w_msr_i

            # assemble_form_w_msr_i = assemble(form_w_msr_i)
            var_w_i.assign(assemble_form_w_msr_i)

        # set cost weight derivative values for the displacement measurement
        for var_dwdv_msr_i, form_dwdv_msr_i in zip(var_dwdu_msr_dic, form_dwdu_msr_dic):

            k = (id(var_w_i), t)
            if k in previously_assembled_forms:
                assemble_form_dwdv_msr_i = previously_assembled_forms[k]
            else:
                assemble_form_dwdv_msr_i = assemble(form_dwdv_msr_i)
                previously_assembled_forms[k] = assemble_form_dwdv_msr_i

            # assemble_form_dwdv_msr_i = assemble(form_dwdv_msr_i)
            var_dwdv_msr_i.assign(assemble_form_dwdv_msr_i)

        # set cost weight derivative values for the force measurement
        for var_dwdv_msr_i, form_dwdv_msr_i in zip(var_dwdf_msr_pad, form_dwdf_msr_pad):

            k = (id(var_w_i), t)
            if k in previously_assembled_forms:
                assemble_form_dwdv_msr_i = previously_assembled_forms[k]
            else:
                assemble_form_dwdv_msr_i = assemble(form_dwdv_msr_i)
                previously_assembled_forms[k] = assemble_form_dwdv_msr_i

            # assemble_form_dwdv_msr_i = assemble(form_dwdv_msr_i)
            var_dwdv_msr_i.assign(assemble_form_dwdv_msr_i)


    ### Initialize Inverse Solver

    ip = invsolve.InverseSolver( model_cost, model, model_parameters, J_u, C_f,
        observation_times=None, measurement_setter=None)

    ip.assign_observation_times(observation_times)
    ip.assign_measurement_setter(measurement_setter)

    ### Solve Inverse Problem

    t_obs = t_msr # All the oservation times

    m_initial = m_initial_further

    if COSTS_FOR_EACH_ITERATION_STUDY and not OPTIMIZATION_CONSTRAINED:
        m_initial = m_initial_nearer
    # Solving for the mean model parameters.

    u.vector()[:] = 0.
    # t_obs = tuple(range(0,50))
    # print(t_obs)
    ip.assign_model_parameters(m_initial)
    time_start = time.time()

    m, (num_iters, has_converged, cost_for_each_iteration,\
            param_for_each_iteration) = ip.minimize_cost_forall(t_obs,\
                                                sensitivity_method='default',\
                                                approximate_D2JDm2='default')

    t_iters = list(range(1, num_iters + 1))
    Ju_iters = cost_for_each_iteration[0]
    Jf_iters = cost_for_each_iteration[1]
    J_total_iters = cost_for_each_iteration[2]

    # =====================================================================
    ### Print final results in form: iteration | cost/paramters
    result_parameters_for_each_iteration = open('results_for_each_iteration',\
                                                'w')

    result_parameters_for_each_iteration.write('===========================\n')
    result_parameters_for_each_iteration.write('Mesh = ' + str(Mesh) + '\n')
    result_parameters_for_each_iteration.write('iterations = ' + str(t_iters))
    result_parameters_for_each_iteration.write('\n')
    result_parameters_for_each_iteration.write('total cost = '\
                                                + str(J_total_iters))
    result_parameters_for_each_iteration.write('\n')
    result_parameters_for_each_iteration.write('displacement cost = '\
                                                + str(Ju_iters))
    result_parameters_for_each_iteration.write('\n')
    result_parameters_for_each_iteration.write('force cost = ' + str(Jf_iters))
    result_parameters_for_each_iteration.write('\n')
    result_parameters_for_each_iteration.write('mu_keloid = ' + str(\
                                                param_for_each_iteration[0]))
    result_parameters_for_each_iteration.write('\n')
    result_parameters_for_each_iteration.write('jm_keloid = ' + str(\
                                                param_for_each_iteration[1]))
    result_parameters_for_each_iteration.write('\n')
    result_parameters_for_each_iteration.write('mu_healthy = ' + str(\
                                                param_for_each_iteration[2]))
    result_parameters_for_each_iteration.write('\n')
    result_parameters_for_each_iteration.write('jm_healthy = ' + str(\
                                                param_for_each_iteration[3]))
    result_parameters_for_each_iteration.write('\n')

    if OPTIMIZATION_CONSTRAINED:
        result_parameters_for_each_iteration.write('lambda = ' + str(\
                                                param_for_each_iteration[4]))
        result_parameters_for_each_iteration.write('\n')

    result_parameters_for_each_iteration.close()

    # =====================================================================
    if PLOT_RESULTS and COSTS_FOR_EACH_ITERATION_STUDY:

        figname = 'Observed costs for each iteration'

        fh = plt.figure(figname)
        ax = fh.add_subplot(111)
        ax.clear()

        ax.semilogy(t_iters, np.abs(J_total_iters), 'b-o', markerfacecolor='w')
        ax.semilogy(t_iters, np.abs(Jf_iters), 'k-', markerfacecolor='w')
        ax.semilogy(t_iters, Ju_iters, 'r--', markerfacecolor='w')

        ax.legend(['Cost of total mismatch', 'Cost of force mismatch',\
                        'Cost of displacement mismatch'])

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(figname, fontsize=14)

        ax.set_xlabel('Iterations []', fontsize=12)
        ax.set_ylabel('Costs []', fontsize=12)

        plt.savefig('results/for_each_iteration/cost_for_each_iteration.eps')

    # ---------------------------------------------------------------------
    if PLOT_RESULTS and COSTS_FOR_EACH_ITERATION_STUDY:

        figname = 'Observed force costs for each iteration'

        fh = plt.figure(figname)
        ax = fh.add_subplot(111)
        ax.clear()

        ax.plot(t_iters, np.abs(Jf_iters), 'k-')

        ax.legend(['Cost of force mismatch'])

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(figname, fontsize=14)

        ax.set_xlabel('Iterations []', fontsize=12)
        ax.set_ylabel('Costs []', fontsize=12)

        plt.savefig('results/for_each_iteration/force_cost_for_each_iteration.eps')

    # ---------------------------------------------------------------------
    if PLOT_RESULTS and COSTS_FOR_EACH_ITERATION_STUDY:

        figname = 'Observed parameter mu_keloid for each iteration'

        fh = plt.figure(figname)
        ax = fh.add_subplot(111)
        ax.clear()

        ax.plot(t_iters, param_for_each_iteration[0],\
                                'r-o', markerfacecolor='w')

        ax.legend(['mu_keloid'])

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(figname, fontsize=14)

        ax.set_xlabel('Iterations []', fontsize=12)
        ax.set_ylabel('mu_keloid [MPa]', fontsize=12)

        plt.savefig('results/for_each_iteration/mu_keloid.eps')

    # ---------------------------------------------------------------------
    if PLOT_RESULTS and COSTS_FOR_EACH_ITERATION_STUDY:

        figname = 'Observed parameter Jm_keloid for each iteration'

        fh = plt.figure(figname)
        ax = fh.add_subplot(111)
        ax.clear()

        ax.plot(t_iters, param_for_each_iteration[1],\
                                'r-o', markerfacecolor='w')

        ax.legend(['Jm_keloid'])

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(figname, fontsize=14)

        ax.set_xlabel('Iterations []', fontsize=12)
        ax.set_ylabel('Jm_keloid []', fontsize=12)

        plt.savefig('results/for_each_iteration/jm_keloid.eps')

    # ---------------------------------------------------------------------
    if PLOT_RESULTS and COSTS_FOR_EACH_ITERATION_STUDY:

        figname = 'Observed parameter mu_healthy for each iteration'

        fh = plt.figure(figname)
        ax = fh.add_subplot(111)
        ax.clear()

        ax.plot(t_iters, param_for_each_iteration[2],\
                                'r-o', markerfacecolor='w')

        ax.legend(['mu_healthy'])

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(figname, fontsize=14)

        ax.set_xlabel('Iterations []', fontsize=12)
        ax.set_ylabel('mu_healthy [MPa]', fontsize=12)

        plt.savefig('results/for_each_iteration/mu_healthy.eps')

    # ---------------------------------------------------------------------
    if PLOT_RESULTS and COSTS_FOR_EACH_ITERATION_STUDY:

        figname = 'Observed parameter Jm_healthy for each iteration'

        fh = plt.figure(figname)
        ax = fh.add_subplot(111)
        ax.clear()

        ax.plot(t_iters, param_for_each_iteration[3],\
                                'r-o', markerfacecolor='w')

        ax.legend(['Jm_healthy'])

        ax.xaxis.set_major_locator(MaxNLocator(integer=True))

        ax.set_title(figname, fontsize=14)

        ax.set_xlabel('Iterations []', fontsize=12)
        ax.set_ylabel('Jm_healthy []', fontsize=12)

        plt.savefig('results/for_each_iteration/jm_healthy.eps')

    # ---------------------------------------------------------------------
    if OPTIMIZATION_CONSTRAINED:
        if PLOT_RESULTS and COSTS_FOR_EACH_ITERATION_STUDY:

            figname = 'Observed parameter lambda for each iteration'

            fh = plt.figure(figname)
            ax = fh.add_subplot(111)
            ax.clear()

            ax.plot(t_iters, param_for_each_iteration[4],\
                                    'r-o', markerfacecolor='w')

            ax.legend(['lambda'])

            ax.xaxis.set_major_locator(MaxNLocator(integer=True))

            ax.set_title(figname, fontsize=14)

            ax.set_xlabel('Iterations []', fontsize=12)
            ax.set_ylabel('lambda []', fontsize=12)

            plt.savefig('results/for_each_iteration/lambda.eps')


    # =====================================================================

    m_forall_lagrange_1 = ip.view_model_parameters_as_list()

    time_end = time.time()
    print("inverse identification took ", time_end - time_start, "seconds")

    if not has_converged:
       raise RuntimeError('Inverse solver did not converge.')

    if COSTS_FOR_EACH_ITERATION_STUDY:
        input("Study for each iteration finished. Abort the process...")
        break
    # ------------------------------------------------------------------
    # Use the converging parameeters from Lagrange-1 mesh set as initial set
    # for Lagrange-2 mesh.
    m_initial = m_forall_lagrange_1 #

    # Lagrange 1
    # ----------

    INTERPOLATION_DEGREE = 2 # Interpolation Lagrange 1

    ### Function spaces

    V = VectorFunctionSpace(mesh_domain, 'CG', INTERPOLATION_DEGREE)
    V_msr_u = VectorFunctionSpace(mesh_domain, 'CG', INTERPOLATION_DEGREE)
    ### Dirichlet Boundary Conditions

    bcs = []

    uD_msr_pad_one = Expression(('ux','uy'), ux=0.0, uy=0.0, degree=0)
    uD_msr_pad_two = Expression(('ux','uy'), ux=0.0, uy=0.0, degree=0)

    bcs = [DirichletBC(V, uD_msr_pad_one, markers_boundary,
                        id_markers_boundary['pad_one']),
           DirichletBC(V, uD_msr_pad_one, markers_boundary,
                        id_markers_boundary['pad_one_sensor']),
           DirichletBC(V, uD_msr_pad_two, markers_boundary,
                        id_markers_boundary['pad_two'])
           ]

    EPS_DOLFIN = 1e-14
    def bottom_boundary(x, on_boundary):
    	return on_boundary and near(x[1], -5, EPS_DOLFIN)

    def top_boundary(x, on_boundary):
    	return on_boundary and near(x[1], 45, EPS_DOLFIN)

    def left_boundary(x, on_boundary):
    	return on_boundary and near(x[0], 0, EPS_DOLFIN)

    def right_boundary(x, on_boundary):
    	return on_boundary and near(x[0], 100, EPS_DOLFIN)

    if FIX_BOUNDARY:

        uD_x = Constant(0.0)
        uD_y = Constant(0.0)

        V_x, V_y = V.split()

        bcs.extend([
            DirichletBC(V_y, uD_y, bottom_boundary),
            DirichletBC(V_y, uD_y, top_boundary),
            #DirichletBC(V_x, uD_x, left_boundary),
            #DirichletBC(V_x, uD_x, right_boundary),
            ])

    ### Project generated data on identification mesh

    u_msr_dom_func = []

    for u_field in u_msr_dom_func_original:
        u_msr_dom_func.append(dolfin.project(u_field, V_msr_u))

    ### Create Measurement Expressions from Data

    if DUMMY_DATA:
        n_msr_dic = len(u_msr_dom_func)
        n_msr_pad = len(fx_msr_pad_vals)

    assert n_msr_dic == n_msr_pad

    n_msr = n_msr_dic
    t_msr = tuple(range(0,n_msr))

    if not DUMMY_DATA:
        u_msr = project.project_pointvalues_on_functions(
        xk=msr_dic_window_xk, fk=msr_dic_window_uk, V_project=V_msr_u,
        meshless_degree=0, num_neighbors=None, distance_norm=2)
        u_msr_as_functions = u_msr

    if DUMMY_DATA:
        u_msr = u_msr_dom_func


    # msr_pad_one_fx = np.take(np.array(fx_msr_pad_vals),0,1)
    msr_pad_one_fx = np.array(fx_msr_pad_vals)
    assert msr_pad_one_fx.ndim == 1
    f_msr = np.zeros((n_msr_pad, 2), float)

    f_msr[:,0] = msr_pad_one_fx # no y-component
    # input('success')
    # NOTE: Devide the reaction force by the pad perimiter to get average traction.
    # Integrating this traction along the pad perimiter will give the net force.
    f_msr /= assemble(1*ds_measure[0])

    u_msr_dic = measure.MeasurementExpression(u_msr, t_msr, degree=2)
    f_msr_pad = measure.MeasurementExpression(f_msr, t_msr, degree=0)

    # place similar measurements in some containers
    u_msr = [u_msr_dic,]
    f_msr = [f_msr_pad,]


    ### Add dummy noise to measurement `u_msr`

    # call to construct a new noise bases for `u_msr`
    # noise_basis = lambda : measure.MeasurementNoise(Function(V_msr_u), degree=2)

    noise_delta_u_msr = [Constant(0) for _ in u_msr]
    # noise_basis_u_msr = [noise_basis() for _ in u_msr]

    # du_msr_noise = [d*v for d,v in zip(noise_delta_u_msr, noise_basis_u_msr)]
    # u_msr_noisy = [u + du for u, du in zip(u_msr, du_msr_noise)]


    ### Add dummy noise to measurement `f_msr`

    # call to construct a new noise bases for df_msr
    # noise_basis = lambda : measure.MeasurementNoise(Constant((-1.0,0.0)), degree=0)

    noise_delta_f_msr = [Constant(0) for _ in f_msr]
    # noise_basis_f_msr = [noise_basis() for _ in f_msr]

    # df_msr_noise = [d*v for d,v in zip(noise_delta_f_msr, noise_basis_f_msr)]
    # f_msr_noisy = [f + df for f, df in zip(f_msr, df_msr_noise)]


    ### Weights for Normalizing Cost Terms
    # i.e. the weights will go in the denominator

    # TODO: Need an abstraction

    # Weight as a `ufl.Form`
    # NOTE: can be differentiated with respect to a `dummy_delta_*`
    # NOTE: shall not multiply anyother `ufl.Form`

    eps_w_msr_dic = Constant(1e-4) # ensure positive denominator in cost
    eps_w_msr_pad = Constant(1e-4) # ensure positive denominator in cost

    form_w_msr_dic = [ (eps_w_msr_dic + u_msr_i**2) * dx_msr_i
        for u_msr_i, dx_msr_i in zip(u_msr, dx_measure)]

    form_w_msr_pad = [ (eps_w_msr_pad + sqrt(f_msr_i**2)) * dx_msr_i
        for f_msr_i, dx_msr_i in zip(f_msr, ds_measure)]

    form_dwdu_msr_dic = [ diff(w_msr_i, d_msr_i)
        for w_msr_i, d_msr_i in zip(form_w_msr_dic, noise_delta_u_msr)]

    form_dwdf_msr_pad = [ diff(w_msr_i, d_msr_i)
        for w_msr_i, d_msr_i in zip(form_w_msr_pad, noise_delta_f_msr)]

    # Weight as a `Constant` variable
    # NOTE: can left-multiply a `ufl.Form`
    # NOTE: can not be differentiated with respect to a `dummy_delta_*`
    # NOTE: values must be assigned inside `measurement_setter`, e.g.
    #   var_w_msr_dic[i].assign(assemble(form_w_msr_dic[i]))
    #   var_w_msr_pad[i].assign(assemble(form_w_msr_pad[i]))

    var_w_msr_dic = [Constant(0.0) for _ in dx_measure]
    var_w_msr_pad = [Constant(0.0) for _ in ds_measure]

    var_dwdu_msr_dic = [Constant(0.0) for _ in dx_measure]
    var_dwdf_msr_pad = [Constant(0.0) for _ in ds_measure]


    ### Model

    u = Function(V)

    if FIX_BOUNDARY:
        material_parameters = [
            {'mu': Constant(0),
             'jm': Constant(0)},
            {'mu': Constant(0),
             'jm': Constant(0)}]
    else:
        material_parameters = [
            {'mu': Constant(0),
             'jm': Constant(0)},
            {'mu': Constant(0),
             'jm': Constant(0)}]

    Pi = Pi_(u, material_parameters, dx_material)


    ### Model Cost

    # J_u = J_u_(u, u_msr_noisy, dx_measure, var_w_msr_dic)
    J_u = J_u_(u, u_msr, dx_measure, var_w_msr_dic)

    ### Model Cost Constraint

    # NOTE:
    # T = dot(P,N) # numerical force
    # R = f_msr[0] # measured reaction

    N = FacetNormal(mesh_domain)

    psi_keloid, P_keloid, S_keloid = Psi_(u, material_parameters[0])
    psi_healthy, P_healthy, S_healthy = Psi_(u, material_parameters[1])

    f = dolfin.dot(dim_reduc(P_keloid), N)

    # C_f = C_fx_(f, f_msr_noisy, ds_measure, var_w_msr_pad)
    C_f = C_fx_(f, f_msr, ds_measure, var_w_msr_pad)
    # C_f = C_f_(f, f_msr_noisy, ds_measure, var_w_msr_pad)

    constraint_multiplier = Constant(-1e-6)
    J_f = constraint_multiplier * C_f


    ### Inverse Solver Arguments

    class model:
        u = u
        Pi = Pi
        bcs = bcs


    model_cost = J_u + J_f

    model_parameters = [
        material_parameters,
        constraint_multiplier]

    observation_times = t_msr


    previously_assembled_forms = {}
    def measurement_setter(t=None):
        '''This function will be called inside the `InverseSolver` for each
        solution time `t`. The purpose of this function is to set the values
        of the measurements.
        '''

        if t is None: return
        if t == -1: t = t_msr[-1]

        # set dirichlet BC to measurement
        uD_msr_pad_one.ux = ux_msr_pad_vals[t]
        print(uD_msr_pad_one.ux)

        # set displacement measurement
        if isinstance(t, int):
            for u_msr_i in u_msr:
                u_msr_i.set_measurement_index(t)
                print('set displacement measurement done for t=', t)
        else:
            for u_msr_i in u_msr:
                u_msr_i.set_measurement_time(t)
                print('set displacement measurement done for t=', t)

        if isinstance(t, int):
            for f_msr_i in f_msr:
                f_msr_i.set_measurement_index(t)
                print('set force measurement done for t=', t)
        else:
            for f_msr_i in f_msr:
                f_msr_i.set_measurement_time(t)
                print('set force measurement done for t=', t)

        # TODO: This needs to be precomputed

        # set cost weights for the displacement measurement
        for var_w_i, form_w_msr_i in zip(var_w_msr_dic, form_w_msr_dic):

            k = (id(var_w_i), t)
            if k in previously_assembled_forms:
                assemble_form_w_msr_i = previously_assembled_forms[k]
            else:
                assemble_form_w_msr_i = assemble(form_w_msr_i)
                previously_assembled_forms[k] = assemble_form_w_msr_i

            # assemble_form_w_msr_i = assemble(form_w_msr_i)
            var_w_i.assign(assemble_form_w_msr_i)

        # set cost weights for the force measurement
        for var_w_i, form_w_msr_i in zip(var_w_msr_pad, form_w_msr_pad):

            k = (id(var_w_i), t)
            if k in previously_assembled_forms:
                assemble_form_w_msr_i = previously_assembled_forms[k]
            else:
                assemble_form_w_msr_i = assemble(form_w_msr_i)
                previously_assembled_forms[k] = assemble_form_w_msr_i

            # assemble_form_w_msr_i = assemble(form_w_msr_i)
            var_w_i.assign(assemble_form_w_msr_i)

        # set cost weight derivative values for the displacement measurement
        for var_dwdv_msr_i, form_dwdv_msr_i in zip(var_dwdu_msr_dic, form_dwdu_msr_dic):

            k = (id(var_w_i), t)
            if k in previously_assembled_forms:
                assemble_form_dwdv_msr_i = previously_assembled_forms[k]
            else:
                assemble_form_dwdv_msr_i = assemble(form_dwdv_msr_i)
                previously_assembled_forms[k] = assemble_form_dwdv_msr_i

            # assemble_form_dwdv_msr_i = assemble(form_dwdv_msr_i)
            var_dwdv_msr_i.assign(assemble_form_dwdv_msr_i)

        # set cost weight derivative values for the force measurement
        for var_dwdv_msr_i, form_dwdv_msr_i in zip(var_dwdf_msr_pad, form_dwdf_msr_pad):

            k = (id(var_w_i), t)
            if k in previously_assembled_forms:
                assemble_form_dwdv_msr_i = previously_assembled_forms[k]
            else:
                assemble_form_dwdv_msr_i = assemble(form_dwdv_msr_i)
                previously_assembled_forms[k] = assemble_form_dwdv_msr_i

            # assemble_form_dwdv_msr_i = assemble(form_dwdv_msr_i)
            var_dwdv_msr_i.assign(assemble_form_dwdv_msr_i)


    ### Initialize Inverse Solver

    ip = invsolve.InverseSolver( model_cost, model, model_parameters, J_u, C_f,
        observation_times=None, measurement_setter=None)

    ip.assign_observation_times(observation_times)
    ip.assign_measurement_setter(measurement_setter)

    ### Solve Inverse Problem

    t_obs = t_msr # All the oservation times

    # Solving for the mean model parameters.

    u.vector()[:] = 0.
    # t_obs = tuple(range(0,50))
    # print(t_obs)
    ip.assign_model_parameters(m_initial)
    time_start = time.time()
    m, (num_iters, has_converged, cost_for_each_iteration,\
            param_for_each_iteration) = ip.minimize_cost_forall(t_obs,\
                                                sensitivity_method='default',\
                                                approximate_D2JDm2='default')

    if not has_converged:
       input('Inverse solver did not converge ... press enter to continue')


    m_forall_lagrange_2 = ip.view_model_parameters_as_list()

    time_end = time.time()
    print("inverse identification took ", time_end - time_start, "seconds")
    # ------------------------------------------------------------------

    ### Print final results in form: number of element | converging parameters
    result_parameters_file.write(str(Mesh) + ', ' \
                                + str(m_forall_lagrange_1)[1:-1] \
                                + '\n' + str(Mesh) + ', ' \
                                + str(m_forall_lagrange_2)[1:-1] + '\n' \
                                + '------------------------'
                                )
# ------------------------------------------------------------------

result_parameters_file.close()
