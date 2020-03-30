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

PLOT_RESULTS = True
OBSERVE_LAST_MEASUREMENT = False # otherwise, all measurement times

### Problem Specific Imports

### Get measurements

import healthy_skin_data
from healthy_skin_data import (
    msr_dic_xk,
    msr_dic_uk,
    ux_msr_pad_one,
    fx_msr_pad_one,
    ux_reg_pad_one_unfiltered,
    fx_reg_pad_one_unfiltered,
    )
	
msr_dic_window_xk = msr_dic_xk
msr_dic_window_uk = msr_dic_uk
msr_pad_one_ux = ux_msr_pad_one
msr_pad_one_fx = fx_msr_pad_one
msr_pad_one_ux_raw = ux_reg_pad_one_unfiltered
msr_pad_one_fx_raw = fx_reg_pad_one_unfiltered
	
### Import mesh

import healthy_skin_mesh
from healthy_skin_mesh import (
    mesh_domain,
    mesh_domain_ZOI,
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

    # F = variable(I + grad(u))
    F = variable(I + grad_reduc(u))       # Deformation gradient

    C = F.T*F
    B = F*F.T   				# Left Cauchy-Green tensor
    E = 0.5*(C-I)
    J = det(F)

    I1 = tr(C)
    I2 = 0.5*(tr(C)**2 - tr(C*C))
    I3 = det(C)
    IB = tr(B)

    mu = material_parameters['mu']
    lm = material_parameters['jm']

    # psi = 0.5*mu*(I1 - 3 - 2*ln(J)) + 0.5*lm*(ln(J))**2 # Neo-Hookean compressible
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
        Expermental measurements, e.g. DIC measurements
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
    subdomain_id=(id_markers_boundary['pad_one_internal_right'],))


ds_measure = [ds_boundary_pad_one]

dx_mat = dx_material
dx_msr_dom = dx_measure
ds_msr_pad = ds_measure


### Function spaces

V = VectorFunctionSpace(mesh_domain, 'CG', 2)
V_msr_u = VectorFunctionSpace(mesh_domain, 'CG', 2)
# V_msr_u = VectorFunctionSpace(mesh_domain_ZOI, 'CG', 1)

material_parameters = [
	{'mu': Constant(0),
	 'jm': Constant(0)},]


### Dirichlet Boundary Conditions

bcs = []

uD_msr_pad_one = Expression(('ux','uy'), ux=0.0, uy=0.0, degree=0)
uD_msr_pad_two = Expression(('ux','uy'), ux=0.0, uy=0.0, degree=0)

bcs = [DirichletBC(V, uD_msr_pad_one, markers_boundary, id_markers_boundary['pad_one_lateral']),
       DirichletBC(V, uD_msr_pad_one, markers_boundary, id_markers_boundary['pad_one_external']),
       DirichletBC(V, uD_msr_pad_one, markers_boundary, id_markers_boundary['pad_one_internal_left']),
       DirichletBC(V, uD_msr_pad_one, markers_boundary, id_markers_boundary['pad_one_internal_right']),
       DirichletBC(V, uD_msr_pad_two, markers_boundary, id_markers_boundary['pad_two'])
       ]

EPS_DOLFIN = 1e-14	   


### Create Measurement Expressions from Data

n_msr_dic = len(msr_dic_window_uk)
n_msr_pad = len(msr_pad_one_fx)

assert n_msr_dic == n_msr_pad

n_msr = n_msr_dic
t_msr = tuple(range(0,n_msr))

print(msr_dic_window_xk)

print(msr_dic_window_uk[-1])
f = open( 'ZOI_displacement.txt', 'w')
for node in msr_dic_window_uk[-1]:
	f.write(str(node)+'\n')
f.close()

# u_msr = project.project_pointvalues_on_functions(
# xk=msr_dic_window_xk, fk=msr_dic_window_uk, V_project=V_msr_u,
# meshless_degree=0, num_neighbors=None, distance_norm=2)
u_msr = project.project_pointvalues_on_functions(xk=msr_dic_window_xk, fk=msr_dic_window_uk, V_project=V_msr_u, meshless_degree=0)
u_msr_as_functions = u_msr

# print(V_msr_u.dofmap().tabulate_all_coordinates(mesh_domain))

# Plotting displacement DIC measurement field in Paraview

file = File("outputs/DIC_final_ZOI.pvd");
file << u_msr_as_functions[-1]
input('Exporting the last DIC frame to VTK file done ...')
# W = dolfin.project(u_msr_as_functions[-1],V)
# file = File("outputs/DIC_final_projected.pvd");
# file << W
# input('Exporting the last DIC frame to VTK file done ...')


# msr_pad_one_fx = np.take(np.array(fx_msr_pad_vals),0,1)
msr_pad_one_fx = np.array(msr_pad_one_fx)
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

Pi = Pi_(u, material_parameters, dx_material)


### Model Cost

J_u = J_u_(u, u_msr, dx_measure, var_w_msr_dic)

### Model Cost Constraint

# NOTE:
# T = dot(P,N) # numerical force
# R = f_msr[0] # measured reaction

N = FacetNormal(mesh_domain)

psi_healthy, P_healthy, S_healthy = Psi_(u, material_parameters[0])

f = dolfin.dot(dim_reduc(P_healthy), N)

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
def measurement_setter(t=None):
    '''This function will be called inside the `InverseSolver` for each
    solution time `t`. The purpose of this function is to set the values
    of the measurements.
    '''

    if t is None: return
    if t == -1: t = t_msr[-1]

    # set dirichlet BC to measurement
    uD_msr_pad_one.ux = msr_pad_one_ux[t]
    print(uD_msr_pad_one.ux)

    # set displacement measurement
    if isinstance(t, int):
        for u_msr_i in u_msr:
            u_msr_i.set_measurement_index(t)
    else:
        for u_msr_i in u_msr:
            u_msr_i.set_measurement_time(t)

    if isinstance(t, int):
        for f_msr_i in f_msr:
            f_msr_i.set_measurement_index(t)
    else:
        for f_msr_i in f_msr:
            f_msr_i.set_measurement_time(t)

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

ip = invsolve.InverseSolver( model_cost, model, model_parameters,
    observation_times=None, measurement_setter=None)

ip.assign_observation_times(observation_times)
ip.assign_measurement_setter(measurement_setter)

### Solve Inverse Problem

t_obs = t_msr 
			
m_initial = [0.002,  0.001,
             1e-04]	

ip.assign_model_parameters(m_initial)

# Solving for the mean model parameters.

start_time = time.time()

num_iters, has_converged = ip.minimize_cost_forall(t_obs,
    sensitivity_method='default', approximate_D2JDm2='default')
	
end_time = time.time() - start_time
print("Computation cost is: ", end_time, "seconds")

if not has_converged:
    raise RuntimeError('Inverse solver did not converge.')

m_forall = ip.view_model_parameters_as_list()

if input('\nContinue? (y/n)\n').lower() != 'y':
    sys.exit('Stopped.')


### Observe Model Cost

J_obs, DJDm_obs = ip.observe_model_cost(compute_gradient=True)

DJDm_obs = np.array(DJDm_obs)

if PLOT_RESULTS:

    figname = 'Observed Cost'

    fh = plt.figure(figname)
    ax = fh.add_subplot(111)
    ax.clear()

    ax.plot(t_obs, J_obs, 'k-o', markerfacecolor='w')

    ax.set_title(figname)
    ax.set_xlabel('Observation time, t ')
    ax.set_ylabel('Cost functional value, J(t)')

    plt.tight_layout()
    plt.show()

    figname = 'Observed Cost Gradient'

    fh = plt.figure(figname)
    ax = fh.add_subplot(111)
    ax.clear()
	
	
    for i in range(ip._n):
        ax.plot(t_obs, DJDm_obs[i], '-o', markerfacecolor='w')

    ax.legend(['\partialJ/\parial\mu_{healthy}',
               '\partialJ/\parial\lambda_{healthy}',
               '\partialJ/\parial\Lambda'])

    ax.set_title(figname)
    ax.set_xlabel('Observation time, t ')
    ax.set_ylabel('Cost gradient value, DJDm')

    plt.tight_layout()
    plt.show()



### Compute observed pad reaction force from displacement control

n_obs = n_msr_dic
i_obs = list(range(n_obs))

msr_pad_one_ux_abs = np.abs(msr_pad_one_ux)
msr_pad_one_fx_abs = np.abs(msr_pad_one_fx)


obs_pad_one_ux_abs = []
obs_pad_one_fx_abs = []

u.vector()[:] = 0.

for i in i_obs:
    # print(f'Progress: {i}/{n_obs-1}')

    uD_msr_pad_one.ux = msr_pad_one_ux[i]

    ip.solve_nonlinear_problem()

    obs_pad_one_ux_abs.append(abs(uD_msr_pad_one.ux))
    obs_pad_one_fx_abs.append(abs(assemble(f[0]*ds_measure[0])))

if PLOT_RESULTS:

    figname = 'Pad Reaction Force vs. Displacement'
    fh = plt.figure(figname)
    # fh.clear()

    ax = fh.add_subplot(111)
    # ax.plot(msr_pad_one_ux_raw, msr_pad_one_fx_raw, 'r.', linewidth=0.75, markersize=2)
    ax.plot(obs_pad_one_ux_abs, obs_pad_one_fx_abs, 'r--', linewidth=2)
    ax.plot(msr_pad_one_ux_abs, msr_pad_one_fx_abs, 'ko', linewidth=2)	
    ax.legend(['Inverse solution', 'Experimental data'])

    ax.set_title(figname)
    ax.set_xlabel('Pad displacement (mm)')
    ax.set_ylabel('Pad reaction force (N)')

    plt.tight_layout()
    plt.show()

if PLOT_RESULTS:

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


### Compute displacement field mismatch wrt DIC

input("Be carful this is the last plot ... !")

t_obs = t_msr[1:n_msr_dic]
n_obs = len(t_obs)

error_u_hat_rel = []
error_u_hat_tot = []

for t in t_obs:
    # print(f'Progress: {t}/{n_msr-1}')

    u_msr[0].set_measurement_time(t)
    uD_msr_pad_one.ux = msr_pad_one_ux[t]

    ip.solve_nonlinear_problem()
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
    ax.plot(t_obs, error_u_hat_tot, 'r-')
    ax.plot(t_obs, error_u_hat_rel, 'b*')
    ax.legend(['total', 'relative'])

    ax.set_title('Displacement field mismatch (L2)')
    ax.set_xlabel('Observation time, t ')
    ax.set_ylabel('||u-u_msr||_2/||u_msr||_2')

    plt.tight_layout()
    plt.show()

