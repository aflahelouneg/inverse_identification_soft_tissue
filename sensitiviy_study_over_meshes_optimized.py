'''
problems/healthy_skin_fixed_pads/runner.py

Problem
-------
Healthy skin extension

Boundary conditions:
--------------------
right pad: fixed displacements
left pad: fixed displacements

Maybe should start with some prestress because at zero displacement, the force
is not zero

'''

from dolfin import *
import dolfin

import time
import os
import shutil
import sys
import logging
import importlib
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt
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


### Problem Configuration Parameters


RUN_AS_TEST = True
PLOT_RESULTS = True
FIX_BOUNDARY = False
OBSERVE_LAST_MEASUREMENT = False # otherwise, all measurement times


### Problem Specific Imports

### Get measurements

if RUN_AS_TEST:
    import keloid_skin_FEM_weakly_nonlinear_lagrange2
    #import keloid_skin_test_3000
    #import keloid_skin_test_540

    u_msr_dom_func_original = keloid_skin_FEM_weakly_nonlinear_lagrange2.out['u_msr_dom_func']

    u_msr_dom_vals = None
    x_msr_dom_vals = None

    ux_msr_pad_vals = keloid_skin_FEM_weakly_nonlinear_lagrange2.out['ux_msr_pad_left_vals']
    fx_msr_pad_vals = keloid_skin_FEM_weakly_nonlinear_lagrange2.out['fx_msr_pad_left_vals']

    V_generator = keloid_skin_FEM_weakly_nonlinear_lagrange2.out['Vector Function Space']
    dx_material_generator = keloid_skin_FEM_weakly_nonlinear_lagrange2.out['FEM domain']

if RUN_AS_TEST:
	msr_pad_one_ux = ux_msr_pad_vals
	msr_pad_one_fx = fx_msr_pad_vals

### ---------------------------
### Generating data with noise


### Noised data factories

def generate_noisy_displacement (u_msr_origin, V, std_u):

    u_msr_noisy = u_msr_origin.copy(deepcopy=True)

    x0_ZOI = 32.0
    x1_ZOI = 68.0
    y0_ZOI = 8.0
    y1_ZOI = 32.0

    dof_idx = []
    dof_map_coordinates = V.tabulate_dof_coordinates()

    n_indices = len(dof_map_coordinates)

    for i in range(n_indices):
        if ( x0_ZOI < dof_map_coordinates[i][0] < x1_ZOI and y0_ZOI < dof_map_coordinates[i][1] < y1_ZOI):
            dof_idx.append(i)

    u_msr_noisy.vector()[dof_idx] += np.random.normal(0, std_u, np.size(dof_idx))

    return u_msr_noisy


def relative_error_force(f, f_ref):

    df = np.abs(np.array(f) - np.array(f_ref))
    error_f = np.sqrt(np.dot(df,df))/np.sqrt(np.dot(f_ref,f_ref))
    return error_f

### ---------------------------

def compute_max_w_msr_dic (u_msr, dx_material):

    total_displacement = []

    for u_msr_i in u_msr:
        total_displacement.append(np.sqrt(assemble(dot(u_msr_i, u_msr_i)*dx_material[0])) +\
                                  np.sqrt(assemble(dot(u_msr_i, u_msr_i)*dx_material[1]))
                                 )
        print(total_displacement[-1])

    return max(total_displacement)


# Standard variations on displacement and raction force
std_u = [0.0, 0.04, 0.12, 0.20]
std_f = [0.0, 0.002, 0.006, 0.01]


### Import mesh

import keloid_skin_mesh_optimized
from keloid_skin_mesh_optimized import (
    mesh_domain,
    markers_domain,
    markers_boundary,
    id_markers_domain,
    id_markers_boundary)

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
	# Transform a 3D tensor to 2D
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

print('Sensor pad surface integration length', dolfin.assemble(1*ds_measure[0]))
print('External pad perimeter', dolfin.assemble(1*ds_boundary_pad_one_external))

if RUN_AS_TEST:
    logger.warning('Assuming measurement domain to be the material domain.')
    dx_msr_dom = dx_mat


### Function spaces

V = VectorFunctionSpace(mesh_domain, 'CG', 2)
V_msr_u = VectorFunctionSpace(mesh_domain, 'CG', 2)


### Dirichlet Boundary Conditions

bcs = []

uD_msr_pad_one = Expression(('ux','uy'), ux=0.0, uy=0.0, degree=0)
uD_msr_pad_two = Expression(('ux','uy'), ux=0.0, uy=0.0, degree=0)

bcs = [DirichletBC(V, uD_msr_pad_one, markers_boundary, id_markers_boundary['pad_one']),
       DirichletBC(V, uD_msr_pad_one, markers_boundary, id_markers_boundary['pad_one_sensor']),
       DirichletBC(V, uD_msr_pad_two, markers_boundary, id_markers_boundary['pad_two'])
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
        DirichletBC(V_x, uD_x, left_boundary),
        DirichletBC(V_x, uD_x, right_boundary),
        ])

msr_pad_one_fx_exact = np.array(fx_msr_pad_vals)

assert msr_pad_one_fx_exact.ndim == 1


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
            assemble_form_w_msr_i = form_w_msr_dic_max
            previously_assembled_forms[k] = assemble_form_w_msr_i

        # assemble_form_w_msr_i = assemble(form_w_msr_i)
        var_w_i.assign(assemble_form_w_msr_i)

    # set cost weights for the force measurement
    for var_w_i, form_w_msr_i in zip(var_w_msr_pad, form_w_msr_pad):

        k = (id(var_w_i), t)
        if k in previously_assembled_forms:
            assemble_form_w_msr_i = previously_assembled_forms[k]
        else:
            assemble_form_w_msr_i = form_w_msr_force_max
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


### Printing converging paramters for each standard variation set
result_parameters_file = open('identified_parameters_with_noise.txt', 'w')


for std_u_i, val_u in enumerate(std_u):
    for std_f_i, val_f in enumerate(std_f):

        print('Inverse identification for standard variations: ', val_u, '--', val_f)
        n_msr_noisy = len(u_msr_dom_func_original)
        t_msr_noisy = tuple(range(0,n_msr_noisy))

        # Applying noises
        u_msr_noisy = []
        fx_msr_pad_left_noisy = []

        for i, t in enumerate(t_msr_noisy):

            u_msr_noisy.append(u_msr_dom_func_original[i].copy(deepcopy=True))
            fx_msr_pad_left_noisy.append(fx_msr_pad_vals[i])

            if t != 0:
                if std_u[std_u_i] != 0.0 :
                    u_msr_noisy[-1] = generate_noisy_displacement(u_msr_dom_func_original[i], V_generator, val_u)
                if std_f[std_f_i] != 0.0 :
                    fx_msr_pad_left_noisy[-1] += np.random.normal(0, val_f)

            file = File("results/noisy_displacement/" + str(val_u) + '_' + str(val_f) + "/displacement_"+str(t_msr_noisy[t])+".pvd");
            file << u_msr_noisy[-1]

        # Computing relative errors between original and noisy data
        U_rel_err = []
        U_abs_err = []
        U_ref_assembled = []

        u_diff = Function(V_generator)

        for t in tuple(range(1,n_msr_noisy)):

            print("Computing relative errors in \"restricted\" ZOI case: ", t, '/', t_msr_noisy[-1], 'done')

            u_diff = u_msr_noisy[t] - u_msr_dom_func_original[t]
            diff_disp = dolfin.project(u_diff, V_generator)

            U_abs_err.append(np.sqrt(assemble(dot(u_diff,u_diff)*dx_material_generator[0])+\
		                assemble(dot(u_diff,u_diff)*dx_material_generator[1])))

            U_ref_assembled.append(np.sqrt(assemble(dot(u_msr_dom_func_original[t],u_msr_dom_func_original[t])*dx_material_generator[0])+\
		                                assemble(dot(u_msr_dom_func_original[t],u_msr_dom_func_original[t])*dx_material_generator[1])))

            U_rel_err.append(U_abs_err[-1]/U_ref_assembled[-1])

            diff_disp.vector()[:] = abs(diff_disp.vector()[:])

        print('Relative errors |reference u - dummy u| is: \n', U_rel_err)

        U_abs_err_all_times = sum(U_abs_err)/sum(U_ref_assembled)
        print('Total relative error |reference u - dummy u| is: \n', U_abs_err_all_times)

        F_rel_err = relative_error_force(fx_msr_pad_left_noisy, fx_msr_pad_vals)
        print('Total relative error |reference f - dummy f| is: ', F_rel_err)

        if PLOT_RESULTS:
            plot_path = 'results/plots/' + str(val_u) + '_' + str(val_f)
            if os.path.exists(plot_path):
                shutil.rmtree(plot_path)
            os.makedirs ('results/plots/' + str(val_u) + '_' + str(val_f))

        if PLOT_RESULTS:

            figname = 'Reaction force vs. x-displacement'
            plt.figure(figname)
            plt.clf()

            FENICS_ux_msr_pad_left_abs = np.abs(np.array(ux_msr_pad_vals))
            FENICS_fx_msr_pad_left_abs = np.abs(np.array(fx_msr_pad_vals))
            FENICS_fx_msr_pad_left_noisy_abs = np.abs(np.array(fx_msr_pad_left_noisy))

            plt.rc('xtick', labelsize=12)
            plt.rc('ytick', labelsize=12)

            plt.plot(FENICS_ux_msr_pad_left_abs, FENICS_fx_msr_pad_left_abs, 'b-.')
            plt.plot(FENICS_ux_msr_pad_left_abs, FENICS_fx_msr_pad_left_noisy_abs, 'r.')
            plt.legend(['Reference','Dummy data'])

            plt.xlabel('Pad displacement [mm]')
            plt.ylabel('Reaction force [N]')
            plt.title(figname)
            plt.savefig('results/plots/' + str(val_u) + '_' + str(val_f) +'/noised_FU_curve.png')
            plt.savefig('results/plots/' + str(val_u) + '_' + str(val_f) +'/noised_FU_curve.eps')


        ### Project generated data on identification mesh

        u_msr_dom_func = []

        for u_i in u_msr_noisy:
            u_msr_dom_func.append(dolfin.project(u_i, V_msr_u))

        ### Create Measurement Expressions from Data

        if RUN_AS_TEST:
            n_msr_dic = len(u_msr_dom_func)
            n_msr_pad = len(fx_msr_pad_left_noisy)

        assert n_msr_dic == n_msr_pad

        n_msr = n_msr_dic
        t_msr = tuple(range(0,n_msr))

        if RUN_AS_TEST:
            u_msr = u_msr_dom_func

        f_msr = np.zeros((n_msr_pad, 2), float)

        f_msr[:,0] = fx_msr_pad_left_noisy # no y-component
        f_msr /= assemble(1*ds_measure[0])

        u_msr_dic = measure.MeasurementExpression(u_msr, t_msr, degree=2)
        f_msr_pad = measure.MeasurementExpression(f_msr, t_msr, degree=0)

        # place similar measurements in some containers
        u_msr = [u_msr_dic,]
        f_msr = [f_msr_pad,]

        ### Weights for Normalizing Cost Terms
        # i.e. the weights will go in the denominator

        # TODO: Need an abstraction

        # Weight as a `ufl.Form`
        # NOTE: can be differentiated with respect to a `dummy_delta_*`
        # NOTE: shall not multiply anyother `ufl.Form`
        noise_delta_u_msr = [Constant(0) for _ in u_msr]
        noise_delta_f_msr = [Constant(0) for _ in f_msr]

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

        # Compute max of displacement

        form_w_msr_dic_max = compute_max_w_msr_dic(u_msr_dom_func, dx_material)

        # Compute max of reaction force

        form_w_msr_force_max = max(np.abs(fx_msr_pad_left_noisy))

        ### Model Cost

        J_u = J_u_(u, u_msr, dx_measure, var_w_msr_dic)

        ### Model Cost Constraint

        # NOTE:
        # T = dot(P,N) # numerical force
        # R = f_msr[0] # measured reaction

        N = FacetNormal(mesh_domain)

        psi_keloid, P_keloid, S_keloid = Psi_(u, material_parameters[0])
        psi_healthy, P_healthy, S_healthy = Psi_(u, material_parameters[1])

        f = dolfin.dot(dim_reduc(P_keloid), N)

        C_f = C_fx_(f, f_msr, ds_measure, var_w_msr_pad)

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

        ### Initialize Inverse Solver

        ip = invsolve.InverseSolver( model_cost, model, model_parameters, J_u, C_f,
             observation_times=None, measurement_setter=None)

        ip.assign_observation_times(observation_times)
        ip.assign_measurement_setter(measurement_setter)

        t_obs = t_msr # TEST

        m_initial = [0.049,  0.19,
                     0.015,  0.39,
                     1e-04]

        # Solving for the mean model parameters.

        u.vector()[:] = 0.

        ip.assign_model_parameters(m_initial)

        try:
            num_iters, has_converged = ip.minimize_cost_forall(t_obs,
                sensitivity_method='default', approximate_D2JDm2='default')
        except:
            has_converged = False

        if not has_converged:
            logger.warning('Inverse solver did not converge.')

        m_forall = ip.view_model_parameters_as_list()

        if not has_converged:
            print('has not converged')
            result_parameters_file.write('Did not converged for std_u = ' + str(val_u) + 'and std_f = '+ str(val_f) + '\n')
        else:
            print('has converged')
            result_parameters_file.write(str(val_f) + ', ' + str(F_rel_err) + ', ' +\
                                         str(val_u) + ', ' + str(U_abs_err_all_times) + ', ' +\
                                         str(m_forall[0:4])[1:-1] + '\n')


        ### Observe Model Cost

        u.vector()[:] = 0.

        J_obs, Ju_obs, Jf_obs = ip.observe_model_cost_seperate()
        Jf_obs = np.abs(Jf_obs)

        if PLOT_RESULTS :

            figname = 'Observed Cost for each observation time'

            fh = plt.figure(figname)
            ax = fh.add_subplot(111)
            ax.clear()

            ax.plot(t_obs, Ju_obs, 'r-o', markerfacecolor='w')
            ax.plot(t_obs, Jf_obs, 'b-o', markerfacecolor='w')
            ax.plot(t_obs, J_obs, 'k--', markerfacecolor='w')
            ax.legend(['Cost of displacements mismatch',
                       'Cost of reaction forces mismatch',
                       'Total cost'])

            plt.rc('xtick', labelsize=12)
            plt.rc('ytick', labelsize=12)

            ax.set_title(figname)
            ax.set_xlabel('Observation time, t ')
            ax.set_ylabel('Cost functional value, J(t)')

            plt.savefig('results/plots/' + str(val_u) + '_' + str(val_f) +'/obs_cost_for_each_obs_time.png')
            plt.savefig('results/plots/' + str(val_u) + '_' + str(val_f) +'/obs_cost_for_each_obs_time.eps')


        ### Compute observed pad reaction force from displacement control

        n_obs = n_msr_dic
        i_obs = list(range(n_obs))

        msr_pad_one_ux_abs = np.abs(msr_pad_one_ux)
        msr_pad_one_fx_abs = np.abs(fx_msr_pad_left_noisy)
        msr_pad_one_fx_exact_abs = np.abs(msr_pad_one_fx)

        obs_pad_one_ux_abs = []
        obs_pad_one_fx_abs = []

        u.vector()[:] = 0.

        for i in i_obs:

            uD_msr_pad_one.ux = msr_pad_one_ux[i]

            ip.solve_nonlinear_problem()

            obs_pad_one_ux_abs.append(abs(uD_msr_pad_one.ux))
            obs_pad_one_fx_abs.append(abs(assemble(f[0]*ds_measure[0])))

        u_msr_noisy_ref = dolfin.project(u_msr_noisy[-1], V)
        u_diff_inverse_solution = u_msr_noisy_ref - u
        diff_disp_inverse_solution = dolfin.project(u_diff_inverse_solution, V)

        file = File("results/plots/" + str(val_u) + '_' + str(val_f) +"/inverse_solution_last_observation_time.pvd");
        file << diff_disp_inverse_solution


        if PLOT_RESULTS:

            figname = 'Pad Reaction Force vs. Displacement (inverse solution)'
            fh = plt.figure(figname)

            ax = fh.add_subplot(111)
            ax.clear()

            ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_abs, 'k.', linewidth=2)
            ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_exact_abs, 'b', linewidth=2)
            ax.plot(obs_pad_one_ux_abs, obs_pad_one_fx_abs, 'r--', linewidth=2)
            ax.legend(['Dummy data', 'Reference', 'Inverse solution'])

            ax.set_title(figname)
            ax.set_xlabel('Pad displacement (mm)')
            ax.set_ylabel('Pad reaction force (N)')

            plt.savefig('results/plots/' + str(val_u) + '_' + str(val_f) +'/inverse_solution.png')
            plt.savefig('results/plots/' + str(val_u) + '_' + str(val_f) +'/inverse_solution.eps')

result_parameters_file.close()
