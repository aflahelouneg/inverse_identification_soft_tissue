'''Uniaxial extension of a 3D bar that is made up of incompressible hyperelastic
material.

Displacement field measurements are taken on the top face. Dirichlet boundary
conditions of zero x-axis displacement are prescribed on left face. Neumann
boundary conditions of uniform x-axis traction are prescribed on right face.

Note, the formulation involves a pressure field (in addition to the displacement
field). The pressure field can be uniquely determined if non-zero Neumann BCs are
imposed. (In other examples, a (compressible) hyperelastic solid could be loaded
by prescribed displacements because the hyperelastic problem was solvable.)

'''

import os
import sys
import math
import logging
import numpy as np
import scipy.linalg as linalg
import matplotlib.pyplot as plt

import dolfin

from dolfin import Constant
from dolfin import DirichletBC
from dolfin import Expression
from dolfin import Function
from dolfin import assemble
from dolfin import derivative

import invsolve
import material

import examples.utility
import examples.plotting

logger = logging.getLogger()
logger.setLevel(logging.INFO)


### Problem parameters

NUM_OBSERVATIONS = 10
SMALL_DISPLACEMENTS = True

PLOT_RESULTS = True
SAVE_RESULTS = True

DEGREE_u = 2
DEGREE_p = 1

PARAMETERS_INVERSE_SOLVER = {
    'solver_method': 'newton', # 'newton' or 'gradient'
    'sensitivity_method': 'adjoint', # 'adjoint' or 'direct'
    'maximum_iterations': 10,
    'maximum_divergences': 5,
    'absolute_tolerance': 1e-6,
    'relative_tolerance': 1e-6,
    'maximum_relative_change': None,
    'error_on_nonconvergence': False,
    }

PROBLEM_DIR = os.path.dirname(os.path.relpath(__file__))
RESULTS_DIR = os.path.join(PROBLEM_DIR, "results")


### Measurements (fabricated)

observation_times = range(1, NUM_OBSERVATIONS+1)

# Box problem domain
L, W, H = 2.0, 1.0, 0.5

# Maximum horizontal displacement of right-face
if SMALL_DISPLACEMENTS:
    uxD_max = 1e-4 # Small displacement case
else:
    uxD_max = 1.5 # Large displacement case

# Fabricated model parameters
mu_target = 1.0
nu_target = 0.5

E_target = 2.0*mu_target*(1.0+nu_target)

ex_max = uxD_max / L # Engineering strain
Tx_max = E_target * ex_max # Right-face traction

measurements_Tx  = np.linspace(0,  Tx_max, NUM_OBSERVATIONS+1)
measurements_uxD = np.linspace(0, uxD_max, NUM_OBSERVATIONS+1)

# Fabricate top-face boundary displacement field measurements
u_msr = Expression(('ex*x[0]', '-nu*ex*x[1]', '-nu*ex*x[2]'), ex=0.0, nu=nu_target, degree=1)

# Right-face boundary traction measurements
T_msr = Expression(('value_x', '0.0', '0.0'), value_x=0.0, degree=0)

def measurement_setter(i):
    '''Set measurements at index'''
    T_msr.value_x = measurements_Tx[i]
    u_msr.ex = measurements_uxD[i] / L

# using_subdims_u_msr = [0, 1, 2]
using_subdims_u_msr = [0, 1]
using_subdims_T_msr = [0]


### Mesh

nx = 10
ny = max(int(nx*W/L), 1)
nz = max(int(nx*H/L), 1)

mesh = dolfin.BoxMesh(dolfin.Point(0,0,0), dolfin.Point(L,W,H), nx, ny, nz)

boundary_fix = dolfin.CompiledSubDomain(f'on_boundary && near(x[0], {0.0})')
boundary_msr = dolfin.CompiledSubDomain(f'on_boundary && near(x[0], {L})')
boundary_dic = dolfin.CompiledSubDomain(f'on_boundary && near(x[2], {H})')

fixed_vertex_000 = dolfin.CompiledSubDomain(
    f'near(x[0], {0.0}) && near(x[1], {0.0}) && near(x[2], {0.0})')

fixed_vertex_010 = dolfin.CompiledSubDomain(
    f'near(x[0], {0.0}) && near(x[1], {W}) && near(x[2], {0.0})')

# Mark the elemental entities (e.g. cells, facets) belonging to the subdomains

gdim = mesh.geometry().dim()

boundary_markers = dolfin.MeshFunction('size_t', mesh, gdim-1)
boundary_markers.set_all(0) # Assign all elements the default value

id_subdomain_fix  = 1 # Fixed boundary id
id_subdomain_msr  = 2 # Loaded boundary id
id_subdomains_dic = 3 # displacement field measurement boundary id

boundary_fix.mark(boundary_markers, id_subdomain_fix)
boundary_msr.mark(boundary_markers, id_subdomain_msr)
boundary_dic.mark(boundary_markers, id_subdomains_dic)


### Integration measures

dx = dolfin.dx(domain=mesh) #, degree=4)
ds = dolfin.ds(domain=mesh) #, degree=4)

ds_msr_T = dolfin.Measure('ds', mesh,
    subdomain_id=id_subdomain_msr,
    subdomain_data=boundary_markers)

ds_msr_u = dolfin.Measure('ds', mesh,
    subdomain_id=id_subdomains_dic,
    subdomain_data=boundary_markers)


### Finite element function spaces

element_u = dolfin.VectorElement('CG', mesh.ufl_cell(), DEGREE_u, gdim)
element_p = dolfin.FiniteElement('CG' if DEGREE_p else 'DG', mesh.ufl_cell(), DEGREE_p)
element = dolfin.MixedElement([element_u, element_p])

V = dolfin.FunctionSpace(mesh, element)

w = Function(V, name="u-p")

u, p = dolfin.split(w)
v, q = dolfin.TestFunctions(V)


### Hyperelastic material model

mu = Constant(1.0) # Guess parameter value

material_parameters = {'mu': mu} # To be inferred


### Deformation measures

deformation_measures = material.DeformationMeasures(u)

d = deformation_measures.d
I = deformation_measures.I
F = deformation_measures.F
E = deformation_measures.E
J = deformation_measures.J
I1 = deformation_measures.I1


### Stress measures

PK2_dev = 2.0*mu*E
PK1_dev = F*PK2_dev

# PK2_hyd = dolfin.inv(F)*(-p)*dolfin.inv(F).T
PK1_hyd = (-p)*dolfin.inv(F).T # A priori assumption of incompressibility (J=1)

N = dolfin.FacetNormal(mesh)
PN = dolfin.dot(PK1_dev + PK1_hyd, N)

dF = dolfin.grad(v)
dpsi = dolfin.inner(PK1_dev, dF)

dU = dpsi*dx # Variational strain energy
dW = dolfin.dot(v, T_msr)*ds_msr_T # Virtual work
dC = dolfin.derivative(-p*(J-1.0)*dx, w) # Incompressibility equation

dPi_w = dU - dW + dC


### Dirichlet boundary conditions

bcs = []

V_u, V_p = V.split()
V_ux, V_uy, V_uz = V_u.split()

zero  = Constant(0)
zeros = Constant((0,0,0))

# # Zero displacement BCs
# bcs.append(DirichletBC(V_u, zeros, boundary_markers, id_subdomain_fix))

# Zero horizontal displacement BCs
bcs.append(DirichletBC(V_ux, zero, boundary_markers, id_subdomain_fix))
bcs.append(DirichletBC(V_u, zeros, fixed_vertex_000, "pointwise"))
bcs.append(DirichletBC(V_uz, zero, fixed_vertex_010, "pointwise"))


### Model cost and constraints

u_obs = u  # Observed displacement
T_obs = PN # Observed tractions

# Displacement misfit cost
cost = sum((u_obs[i]-u_msr[i])**2*ds_msr_u for i in using_subdims_u_msr)


### Inverse problem

# Model parameters to be optimized
model_parameters = [material_parameters]

inverse_solver_basic = invsolve.InverseSolverBasic(cost, dPi_w, w, bcs,
    model_parameters, observation_times, measurement_setter,
    PARAMETERS_INVERSE_SOLVER)

inverse_solver = invsolve.InverseSolver(inverse_solver_basic,
    u_obs, u_msr, ds_msr_u, T_obs, T_msr, ds_msr_T)


### Solve inverse problem

cost_values_initial = cost_gradients_initial = None

# cost_values_initial, cost_gradients_initial = \
#     inverse_solver.assess_model_cost(compute_gradients=False)

model_parameters_foreach, iterations_count_foreach, is_converged_foreach = \
    inverse_solver.fit_model_foreach_time() # Default observation times

model_parameters_forall, iterations_count_forall, is_converged_forall = \
    inverse_solver.fit_model_forall_times() # Default observation times

cost_values_final, cost_gradients_final = \
    inverse_solver.assess_model_cost(compute_gradients=True)


### Mismatch between model and measurements

misfit_displacements = inverse_solver \
    .assess_misfit_displacements(observation_times, using_subdims_u_msr)
# NOTE: Value at `[I][J]` corresponds to the `I`th measurement, `J`th time.

misfit_reaction_forces = inverse_solver \
    .assess_misfit_reaction_forces(observation_times, using_subdims_T_msr)
# NOTE: Value at `[I][J]` corresponds to the `I`th measurement, `J`th time.


### Force-displacement curve

reaction_forces_observed = inverse_solver.observe_f_obs(observation_times)
reaction_forces_measured = inverse_solver.observe_f_msr(observation_times)
# NOTE: Value at `[I][J][K]` corresponds to the `I`th measurement, `J`th time,
#       `K`th force dimension.


### Assess cost condition number

D2JDm2 = inverse_solver.view_cumsum_D2JDm2()
cond_D2JDm2 = np.linalg.cond(D2JDm2)


### Plotting

# Model parameter names to be used in labeling plots
model_parameter_names = list(material_parameters.keys())

def plot_everything():

    plt.close('all')

    fig_handle_and_name_pairs = []

    fig_handle_and_name_pairs.append(
        examples.plotting.plot_model_parameters_foreach(
            model_parameters_foreach,
            model_parameter_names,
            observation_times,
            figname="Fitted Model Parameters for Each Observation Time"))

    fig_handle_and_name_pairs.append(
        examples.plotting.plot_model_parameters_forall(
            model_parameters_forall,
            model_parameter_names,
            figname="Fitted Model Parameters for all Observation Times"))

    fig_handle_and_name_pairs.append(
        examples.plotting.plot_model_cost(
            cost_values_final,
            cost_values_initial,
            observation_times,
            figname="Model Cost"))

    fig_handle_and_name_pairs.append(
        examples.plotting.plot_cost_gradients(
            cost_gradients_final,
            model_parameter_names,
            observation_times,
            figname="Model Cost Derivatives"))

    fig_handle_and_name_pairs.append(
        examples.plotting.plot_observation_misfit(
            misfit_reaction_forces_i,
            observation_times,
            figname="Reaction Force Misfit Error",
            ylabel="Reaction force misfit error, $||f_{obs}-f_{msr}||/||f_{msr}||$"))

    fig_handle_and_name_pairs.append(
        examples.plotting.plot_observation_misfit(
            misfit_displacements_i,
            observation_times,
            figname="Displacement Field Misfit Error",
            ylabel="Displacement field misfit error, $||u_{obs}-u_{msr}||/||u_{msr}||$"))

    fig_handle_and_name_pairs.append(
        examples.plotting.plot_reaction_force_vs_displacement(
            reaction_force_magnitude_observed_i,
            reaction_force_magnitude_measured_i,
            reaction_displacement_magnitude_i,
            figname="Reaction Force-Displacement Curve"))

    return fig_handle_and_name_pairs


if __name__ == '__main__':

    plt.interactive(True)

    i_msr_u = 0 # Assess first displacement field measurements
    i_msr_f = 0 # Assess first reaction force measurements
    i_time = -1 # Assess last observation time

    misfit_displacements_i = misfit_displacements[i_msr_u]
    misfit_reaction_forces_i = misfit_reaction_forces[i_msr_f]

    reaction_force_observed_i = reaction_forces_observed[i_msr_f]
    reaction_force_measured_i = reaction_forces_measured[i_msr_f]

    reaction_force_magnitude_observed_i = np.sqrt(np.array(
        reaction_force_observed_i)**2).sum(axis=1).tolist()

    reaction_force_magnitude_measured_i = np.sqrt(np.array(
        reaction_force_measured_i)**2).sum(axis=1).tolist()

    reaction_displacement_magnitude_i = \
        [measurements_uxD[t] for t in observation_times]

    print(f'\nmodel_parameters_foreach (converged={all(is_converged_foreach)}):')
    for t, r in zip(observation_times, np.array(model_parameters_foreach)):
        print(r, end=' '); print(f'[t={t}]')

    print(f'\nmodel_parameters_forall (converged={is_converged_forall}):')
    print(np.array(model_parameters_forall))

    print(f'\nerror_displacements (subdims={using_subdims_u_msr}):')
    for t, v in zip(observation_times, misfit_displacements_i):
        print(f'{v:12.5e} [t={t}]')

    print(f'\nerror_reaction_forces (subdims={using_subdims_T_msr}):')
    for t, v in zip(observation_times, misfit_reaction_forces_i):
        print(f'{v:12.5e} [t={t}]')

    print('\ncond(D2JDm2):')
    print(f'{cond_D2JDm2:.5e}')

    print(f'\nnorm(u):')
    print(f'{dolfin.norm(w):.5e}')

    if PLOT_RESULTS or SAVE_RESULTS:

        fig_handle_and_name_pairs = plot_everything()
        fig_handles = [f[0] for f in fig_handle_and_name_pairs]
        fig_names = [f[1] for f in fig_handle_and_name_pairs]

        if SAVE_RESULTS:

            if not os.path.isdir(RESULTS_DIR):
                os.makedirs(RESULTS_DIR)

            for handle_i, name_i in zip(fig_handles, fig_names):
                handle_i.savefig(os.path.join(RESULTS_DIR, name_i)+'.png')
                handle_i.savefig(os.path.join(RESULTS_DIR, name_i)+'.pdf')

            if not PLOT_RESULTS:
                plt.close('all')

            outfile_u = dolfin.File(os.path.join(RESULTS_DIR,'pvd','u.pvd'))
            outfile_p = dolfin.File(os.path.join(RESULTS_DIR,'pvd','p.pvd'))

            u_, p_ = w.split()

            for t in inverse_solver.observation_times:
                inverse_solver.solve_nonlinear_problem(t)
                outfile_u << u_
                outfile_p << p_
