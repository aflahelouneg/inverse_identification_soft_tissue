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
NOISY_TEST = False
OBSERVE_LAST_MEASUREMENT = False # otherwise, all measurement times


### Problem Specific Imports

### Get measurements

if RUN_AS_TEST:
    import keloid_skin_test_44000
    #import keloid_skin_test_3000
    #import keloid_skin_test_540

    u_msr_dom_func_original = keloid_skin_test_44000.out['u_msr_dom_func']
    # du_msr_dom_func = keloid_skin_test.out['du_msr_dom_func']

    u_msr_dom_vals = None
    x_msr_dom_vals = None

    ux_msr_pad_vals = keloid_skin_test_44000.out['ux_msr_pad_left_vals']
    # uy_msr_pad_vals = keloid_skin_test.out['uy_msr_pad_left_vals']

    fx_msr_pad_vals = keloid_skin_test_44000.out['fx_msr_pad_left_vals']
    # fy_msr_pad_vals = keloid_skin_test.out['fy_msr_pad_left_vals']
    fx_msr_pad_vals_exact = keloid_skin_test_44000.out['fx_msr_pad_left_vals_exact']

    # df_msr_pad_vals = keloid_skin_test.out['df_msr_pad_left_vals']
    print (fx_msr_pad_vals)
    print (ux_msr_pad_vals)
    # input ("pause ...")
	
if RUN_AS_TEST:
	msr_pad_one_ux = ux_msr_pad_vals
	msr_pad_one_fx = fx_msr_pad_vals

### Import mesh

import keloid_skin_mesh_4000
from keloid_skin_mesh_4000 import (
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


# ds_boundary_bottom = dolfin.Measure('ds',
    # domain=mesh_domain,
    # subdomain_data=markers_boundary,
    # subdomain_id=id_markers_boundary['bottom'])

# ds_boundary_right = dolfin.Measure('ds',
    # domain=mesh_domain,
    # subdomain_data=markers_boundary,
    # subdomain_id=id_markers_boundary['right'])

# ds_boundary_top = dolfin.Measure('ds',
    # domain=mesh_domain,
    # subdomain_data=markers_boundary,
    # subdomain_id=id_markers_boundary['top'])

# ds_boundary_left = dolfin.Measure('ds',
    # domain=mesh_domain,
    # subdomain_data=markers_boundary,
    # subdomain_id=id_markers_boundary['left'])

# ds_boundary_pad_one = dolfin.Measure('ds',
    # domain=mesh_domain,
    # subdomain_data=markers_boundary,
    # subdomain_id=(id_markers_boundary['pad_one_lateral'], id_markers_boundary['pad_one_internal'], id_markers_boundary['pad_one_external']))

# ds_boundary_pad_two = dolfin.Measure('ds',
    # domain=mesh_domain,
    # subdomain_data=markers_boundary,
    # subdomain_id=id_markers_boundary['pad_two'])

# ds_boundary_pad_one_lateral = dolfin.Measure('ds',
    # domain=mesh_domain,
    # subdomain_data=markers_boundary,
    # subdomain_id=id_markers_boundary['pad_one_lateral'])

# ds_boundary_pad_one_external = dolfin.Measure('ds',
    # domain=mesh_domain,
    # subdomain_data=markers_boundary,
    # subdomain_id=id_markers_boundary['pad_one_external'])

# ds_boundary_pad_one_internal = dolfin.Measure('ds',
    # domain=mesh_domain,
    # subdomain_data=markers_boundary,
    # subdomain_id=id_markers_boundary['pad_one_internal'])
	
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

#if RUN_AS_TEST:
#    logger.warning('Assuming measurement domain to be the material domain.')
#    dx_msr_dom = dx_mat


### Function spaces

V = VectorFunctionSpace(mesh_domain, 'CG', 2)
V_msr_u = VectorFunctionSpace(mesh_domain, 'CG', 2)


### Project generated data on identification mesh

u_msr_dom_func = []

for u_field in u_msr_dom_func_original:
    u_msr_dom_func.append(dolfin.project(u_field, V_msr_u))
    


### Dirichlet Boundary Conditions

bcs = []

uD_msr_pad_one = Expression(('ux','uy'), ux=0.0, uy=0.0, degree=0)
uD_msr_pad_two = Expression(('ux','uy'), ux=0.0, uy=0.0, degree=0)

# bcs.extend([
    # DirichletBC(V, uD_msr_pad_one, markers_boundary, id_markers_boundary['pad_one']),
    # DirichletBC(V, uD_msr_pad_two, markers_boundary, id_markers_boundary['pad_two']),
    # ])
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


### Create Measurement Expressions from Data

if RUN_AS_TEST:
    n_msr_dic = len(u_msr_dom_func)
    n_msr_pad = len(fx_msr_pad_vals)

assert n_msr_dic == n_msr_pad

n_msr = n_msr_dic
t_msr = tuple(range(0,n_msr))

if not RUN_AS_TEST:
    u_msr = project.project_pointvalues_on_functions(
    xk=msr_dic_window_xk, fk=msr_dic_window_uk, V_project=V_msr_u,
    meshless_degree=0, num_neighbors=None, distance_norm=2)
    u_msr_as_functions = u_msr

if RUN_AS_TEST:
    u_msr = u_msr_dom_func


# msr_pad_one_fx = np.take(np.array(fx_msr_pad_vals),0,1)
msr_pad_one_fx = np.array(fx_msr_pad_vals)
msr_pad_one_fx_exact = np.array(fx_msr_pad_vals_exact)
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

# t_obs = t_msr[-5:]

# t_obs = t_msr[-5:] # TEST
# m_initial = [1.541546e-01,  2.657713e-03,
             # 1.692446e-02,  3.852991e-02,
             # 1.443419e-08]
# ip.model_parameters_assign(m_initial)

# t_obs = t_msr[-7:] # TEST
# m_initial = [1.463061e-01,  2.548416e-03,
             # 1.614291e-02,  3.663149e-02,
            # -2.383610e-08]
# ip.model_parameters_assign(m_initial)

# t_obs = t_msr[-10:] # TEST _ 
# SUCCESS !!
# m_initial = [1.0250479e-01,  1.500479e-03,
             # 3.500479e-02,  1.500479e-02,
            # 4.010904e-09]
# ip.model_parameters_assign(m_initial)

# m_initial = [0.1587181, 0.02975965,
             # 0.1091187,  0.00919883,
            # 4.010904e09]

# m_initial = [2.975965e-02, 1.587181e-01,
             # 9.91883e-03,  1.091187e-01,
            # 4.010904e-09]

# t_obs = t_msr[12::10] # TEST
t_obs = t_msr # TEST
# t_obs = t_msr[-1] # last observation time
m_initial = [0.049, 0.19,
             0.015, 0.39,
             1e-04]
# m_initial = [10,  10,
             # 10,  10,
             # 1e-08]
			 
# m_initial = [3.5/500, 0.2/500,
             # 1.0/500, 0.5/500,
            # 4.010904e-09]
			
# m_initial = [0.05, 0.2,
             # 0.016, 0.4,
            # 1e-9]
# ip.assign_model_parameters(m_initial)

# Solving for the mean model parameters.


u.vector()[:] = 0.
# t_obs = tuple(range(0,50))
# print(t_obs)
ip.assign_model_parameters(m_initial)
time_start = time.time()
num_iters, has_converged = ip.minimize_cost_forall(t_obs,
    sensitivity_method='default', approximate_D2JDm2='default')

if not has_converged:
   raise RuntimeError('Inverse solver did not converge.')

m_forall = ip.view_model_parameters_as_list()

# m_initial = [0.01, 0.01,
             # 0.01, 0.01,
            # 1e-4]
# ip.assign_model_parameters(m_initial)

# all_parameters = []
# f = open( 'params_out.txt', 'w' )

# for n_obs in range(10,50):
    # u.vector()[:] = 0.
    # t_obs = tuple(range(0,n_obs))
    # print(t_obs)
    # ip.assign_model_parameters(m_initial)
    # num_iters, has_converged = ip.minimize_cost_forall(t_obs,
        # sensitivity_method='default', approximate_D2JDm2='default')

    # if not has_converged:
       # raise RuntimeError('Inverse solver did not converge.')

    # m_forall = ip.view_model_parameters_as_list()
    # f.write(str(m_forall)+'\n')
    # all_parameters.append(m_forall)
    # print(np.array(all_parameters))

# f.close()
time_end = time.time()
print("inverse identification took ", time_end - time_start, "seconds")
input('pause')

### Assess model sentitivity

# compute_sensitivity_wrt_u_msr = ip.observe_model_sensitivity(noise_delta_u_msr)
# compute_sensitivity_wrt_f_msr = ip.observe_model_sensitivity(dummy_delta_f_msr)
#
# noise_basis_u_msr[0].vector()[:] = -1.0
# basis_delta_f_msr[0].dfx = -1.0
#
# dmdu_msr, dm_res_a = compute_sensitivity_wrt_u_msr()
# dmdf_msr, dm_res_b = compute_sensitivity_wrt_f_msr()
#
# pprint(dmdu_msr)
# pprint(dmdf_msr)


### Assess mismatch between observation and measurements

# J2^(k) = \int [ (u^(k)-u_msr^(k))^2 / (u_msr^(k))^2 ] dx
# J^(k) = [ J2^(k) / (\int dx) ]^(1/2)

# DJ^(k)/Dm_j = (1/2) [ J2^(k) / (\int dx) ]^(-1/2) [ DJ2^(k)/Dm_j / (\int dx) ]
# DJ^(k)/Dm_j = (1/2) [ 1 / J^(k) ] [ DJ2^(k)/Dm_j / (\int dx) ]

# A_msr = assemble(
#     sum(1.0*dx_msr_i for dx_msr_i in dx_measure))

# J2_msr = sum(((u-u_msr_i)**2/u_msr_i**2) * dx_msr_i
#     for u_msr_i, dx_msr_i in zip(u_msr, dx_measure))
    #
    # Don't forget to:
    #   (1) devide by the measurement area,
    #   (2) take the square-root.
    #


### Observe Model Cost

#J_obs, DJDm_obs = ip.observe_model_cost(compute_gradient=True)
J_obs, Ju_obs, Jf_obs = ip.observe_model_cost_seperate()
Jf_obs = np.abs(Jf_obs)

u.vector()[:] = 0.

#DJDm_obs = np.array(DJDm_obs)

if PLOT_RESULTS and False:

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

    ax.set_title(figname)
    ax.set_xlabel('Observation time, t ')
    ax.set_ylabel('Cost functional value, J(t)')

    plt.tight_layout()
    plt.show()

# if PLOT_RESULTS:

    # figname = 'Observed Cost'

    # fh = plt.figure(figname)
    # ax = fh.add_subplot(111)
    # ax.clear()

    # ax.plot(t_obs, J_obs, 'k-o', markerfacecolor='w')

    # ax.set_title(figname)
    # ax.set_xlabel('Observation time, t ')
    # ax.set_ylabel('Cost functional value, J(t)')

    # plt.tight_layout()
    # plt.show()

    # figname = 'Observed Cost Gradient'

    # fh = plt.figure(figname)
    # ax = fh.add_subplot(111)
    # ax.clear()
	
	
    # for i in range(ip._n):
        # ax.plot(t_obs, DJDm_obs[i], '-o', markerfacecolor='w')

    # ax.legend(['dJ/dmu_keloid',
               # 'dJ/djm_keloid',
               # 'dJ/dmu_healthy',
               # 'dJ/djm_healthy',
               # 'd\dlambda'])

    # ax.set_title(figname)
    # ax.set_xlabel('Observation time, t ')
    # ax.set_ylabel('Cost gradient value, DJDm')

    # plt.tight_layout()
    # plt.show()

input("Be carful this is the last plot ... !")


### Compute observed pad reaction force from displacement control

n_obs = n_msr_dic
i_obs = list(range(n_obs))

msr_pad_one_ux_abs = np.abs(msr_pad_one_ux)
msr_pad_one_fx_abs = np.abs(msr_pad_one_fx)
msr_pad_one_fx_exact_abs = np.abs(msr_pad_one_fx_exact)

obs_pad_one_ux_abs = []
obs_pad_one_fx_abs = []

u.vector()[:] = 0.

for i in i_obs:
    # print(f'Progress: {i}/{n_obs-1}')

    uD_msr_pad_one.ux = msr_pad_one_ux[i]

    ip.solve_nonlinear_problem()

    obs_pad_one_ux_abs.append(abs(uD_msr_pad_one.ux))
    obs_pad_one_fx_abs.append(abs(assemble(f[0]*ds_measure[0])))

if PLOT_RESULTS and RUN_AS_TEST:

    figname = 'Pad Reaction Force vs. Displacement'
    fh = plt.figure(figname)
    # fh.clear()

    ax = fh.add_subplot(111)
    # ax.plot(msr_pad_one_ux_raw, msr_pad_one_fx_raw, 'r.', linewidth=0.75, markersize=2)
    # ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_abs, 'k.', linewidth=2)
    ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_exact_abs, 'b', linewidth=2)
    # ax.plot(obs_pad_one_ux_abs, obs_pad_one_fx_abs, 'r--', linewidth=2)	
    ax.legend(['Reference'])

    ax.set_title(figname)
    ax.set_xlabel('Pad displacement (mm)')
    ax.set_ylabel('Pad reaction force (N)')

    plt.tight_layout()
    plt.show()
	
if PLOT_RESULTS and RUN_AS_TEST:

    figname = 'Pad Reaction Force vs. Displacement_0'
    fh = plt.figure(figname)
    # fh.clear()

    ax = fh.add_subplot(111)
    # ax.plot(msr_pad_one_ux_raw, msr_pad_one_fx_raw, 'r.', linewidth=0.75, markersize=2)
    ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_abs, 'k.', linewidth=2)
    ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_exact_abs, 'b', linewidth=2)
    # ax.plot(obs_pad_one_ux_abs, obs_pad_one_fx_abs, 'r--', linewidth=2)	
    ax.legend(['Dummy data', 'Reference'])

    ax.set_title(figname)
    ax.set_xlabel('Pad displacement (mm)')
    ax.set_ylabel('Pad reaction force (N)')

    plt.tight_layout()
    plt.show()
	
if PLOT_RESULTS and RUN_AS_TEST:

    figname = 'Pad Reaction Force vs. Displacement_1'
    fh = plt.figure(figname)
    # fh.clear()

    ax = fh.add_subplot(111)
    # ax.plot(msr_pad_one_ux_raw, msr_pad_one_fx_raw, 'r.', linewidth=0.75, markersize=2)
    ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_abs, 'k.', linewidth=2)
    # ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_exact_abs, 'b', linewidth=2)
    # ax.plot(obs_pad_one_ux_abs, obs_pad_one_fx_abs, 'r--', linewidth=2)	
    ax.legend(['Dummy data'])

    ax.set_title(figname)
    ax.set_xlabel('Pad displacement (mm)')
    ax.set_ylabel('Pad reaction force (N)')

    plt.tight_layout()
    plt.show()
	
	
if PLOT_RESULTS and RUN_AS_TEST:

    figname = 'Pad Reaction Force vs. Displacement_2'
    fh = plt.figure(figname)
    # fh.clear()

    ax = fh.add_subplot(111)
    # ax.plot(msr_pad_one_ux_raw, msr_pad_one_fx_raw, 'r.', linewidth=0.75, markersize=2)
    ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_abs, 'k.', linewidth=2)
    # ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_exact_abs, 'b', linewidth=2)
    ax.plot(obs_pad_one_ux_abs, obs_pad_one_fx_abs, 'r--', linewidth=2)	
    ax.legend(['Dummy data', 'Inverse solution'])

    ax.set_title(figname)
    ax.set_xlabel('Pad displacement (mm)')
    ax.set_ylabel('Pad reaction force (N)')

    plt.tight_layout()
    plt.show()
if PLOT_RESULTS and False:

    figname = 'Model fitting range'
    fh = plt.figure(figname)
    # fh.clear()

    ax = fh.add_subplot(111)
    ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_abs, 'k.')
    ax.plot(obs_pad_one_ux_abs, obs_pad_one_fx_abs, 'b-')
    ax.legend(['model fitting range', 'inverse solution'])

    ax.set_title(figname)
    ax.set_xlabel('Pad displacement (mm)')
    ax.set_ylabel('Pad reaction force (N)')

    plt.tight_layout()
    plt.show()

all_parameters = []
f = open( 'params_out.txt', 'w' )

for n_obs in range(10,50):
    u.vector()[:] = 0.
    t_obs = tuple(range(0,n_obs))
    print(t_obs)
    ip.assign_model_parameters(m_initial)
    num_iters, has_converged = ip.minimize_cost_forall(t_obs,
        sensitivity_method='default', approximate_D2JDm2='default')

    if not has_converged:
       raise RuntimeError('Inverse solver did not converge.')

    m_forall = ip.view_model_parameters_as_list()
    f.write(str(m_forall)+'\n')
    all_parameters.append(m_forall)
    print(np.array(all_parameters))

f.close()

### Compute displacement field mismatch wrt DIC

t_obs = t_msr[1:n_msr_dic]
n_obs = len(t_obs)

error_u_hat_rel = []
error_u_hat_tot = []

u.vector()[:] = 0.

for t in t_obs:
    # print(f'Progress: {t}/{n_msr-1}')

#    u_msr[0].set_measurement_time(t)
#    uD_msr_pad_one.ux = msr_pad_one_ux[t]

    ip.solve_nonlinear_problem(t) # It works also with the two commands above
    u_hat_t = dolfin.project(u, V_msr_u)
	# u_hat_t = u_hat_t.get_function
    # u_msr_t = u_msr[0].get_function()
    u_msr_t = dolfin.project(u_msr[0], V_msr_u)
    du_t = u_hat_t - u_msr_t

    try:
        error_tot = np.sqrt(assemble(dot(du_t,du_t)*dx_measure[0]))
        error_rel = error_tot/ np.sqrt(assemble(dot(u_msr_t,u_msr_t)*dx_measure[0]))
        error_tot /= np.sqrt(assemble(1.0*dx_measure[0]))
    except ZeroDivisionError:
        error_tot = 0.0
        error_rel = 0.0

    error_u_hat_tot.append(error_tot)
    error_u_hat_rel.append(error_rel)


if PLOT_RESULTS:

    fh = plt.figure('inverse solution: displacement mismatch (L2)')
    # fh.clear()

    ax = fh.add_subplot(111)
#    ax.plot(t_obs, error_u_hat_tot, 'r-')
    ax.plot(t_obs, error_u_hat_rel, 'b*')
#    ax.legend(['total', 'relative'])
    ax.legend(['Relative'])
    ax.set_title('Displacement field mismatch (L2)')
    ax.set_xlabel('Observation time, t ')
    ax.set_ylabel('Error norm')

    plt.tight_layout()
    plt.show()


input("continue? (y/n)")




### Export solutions for comparison

# u_hat_trans = [ip.get_observation(t) for t in t_msr]
# u_msr_trans = [u_msr[0].get_function_at_index(t) for t in t_msr]

# if FIX_BOUNDARY:
    # dirname = 'fixed-boundary'
# else:
    # dirname = 'free-boundary'

# helper_methods.write_solution(u_hat_trans, V, dirname+'/u/u.pvd')
# helper_methods.write_solution(u_hat_trans, V_msr_u, dirname+'/u_hat/u_hat.pvd')
# helper_methods.write_solution(u_msr_trans, V_msr_u, dirname+'/u_msr/u_msr.pvd')

# u_out = Function(V)
# s_out = Function(V_ten)

# _, _, PK2 = Psi_(u_out, material_parameters[0])

# fid_u = File('./results/'+dirname+'/u-s/u/u.pvd')
# fid_s = File('./results/'+dirname+'/u-s/s/s.pvd')

# for u_hat_t in u_hat_trans:

    # u_out.assign(u_hat_t)
    # s_out.assign(project(PK2,V_ten))

    # fid_u << u_out
    # fid_s << s_out




# Solving for the model parameters at each time.

m_foreach_out = ip.minimize_cost_foreach(t_obs,
     sensitivity_method='default', approximate_D2JDm2='default')
	 
if len(m_foreach_out) != len(t_obs):
    raise RuntimeError('Inverse solver did not converge.')

# index parameters first rather than time
m_foreach = np.vstack(m_foreach_out).T
m_foreach = [m_t for m_t in m_foreach]

if input('\nContinue? (y/n) ').lower() != 'y':
    sys.exit('Stopped.')
	
# if False:

    # m_foreach_out = ip.minimize_cost_foreach(t_obs,
        # sensitivity_method='default', approximate_D2JDm2='default')

    # if len(m_foreach_out) != len(t_obs):
        # raise RuntimeError('Inverse solver did not converge.')

    # index parameters first rather than time
    # m_foreach = np.vstack(m_foreach_out).T
    # m_foreach = [m_t for m_t in m_foreach]

    # if input('\nContinue? (y/n) ').lower() != 'y':
        # sys.exit('Stopped.')


# Plot model parameter calibration results

input("the last plot ...")

if PLOT_RESULTS:

    figname = 'Model Parameter Calibration for Keloid and Healthy Skin'
    fh = plt.figure(figname)
    fh.clear()

    ax = fh.add_subplot(111)

    for i in range(ip._n):

        lh_foreach = plt.plot(t_obs, m_foreach[i], 'r-^',
            markerfacecolor="None", markeredgecolor='r', markeredgewidth=1.5)

        lh_forall = plt.plot(t_obs, m_forall[i:i+1]*len(t_obs), 'k--o',
            markerfacecolor="None", markeredgecolor='k', markeredgewidth=1.5)

        ax.set_title(figname)
        ax.set_xlabel('Observation time (#)')
        ax.set_ylabel('Model parameter value')

    plt.legend(['for each observation time','for all observation times'])
    plt.tight_layout()
    plt.show()
