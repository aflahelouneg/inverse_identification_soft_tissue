'''Uniaxial extension of a bar.

The displacement measurements are a sequence of point-measurements. These
are projected using least-squares meshless on a continuous function space.

Measurements
------------
- Measured displacements on the top face.
- Measured reaction (tractions) on the right face.

Boundary conditions
-------------------
- Imposed displacements on the right face.
- Imposed zero-displacement on the left face.

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

import invsolve
import material

import examples.utility
import examples.plotting

from examples.utility import SimpleTimer
from examples.utility import reload_module

logger = logging.getLogger()
logger.setLevel(logging.INFO)


### Problem parameters

FORCE_COST_FORMULATION_METHOD = "cost"
# FORCE_COST_FORMULATION_METHOD = "constraint"

NUM_OBSERVATIONS = 4

SMALL_DISPLACEMENTS = True

FINITE_ELEMENT_DEGREE = 1

MESHLESS_DEGREE = 2 # For projecting displacement point measurements
MESHLESS_WEIGHT = "center" # "center", "uniform"

PLOT_RESULTS = True
SAVE_RESULTS = True

PROBLEM_DIR = os.path.dirname(os.path.relpath(__file__))
RESULTS_DIR = os.path.join(PROBLEM_DIR, "results")

TEST_MODEL_PARAMETER_SELF_SENSITIVITIES = True
TEST_SENSITIVITY_REACTION_MEASUREMENTS = True
TEST_SENSITIVITY_DISPLACEMENT_MEASUREMENTS = True

parameters_inverse_solver = {
    'solver_method': 'newton', # 'newton' or 'gradient'
    'sensitivity_method': 'adjoint', # 'adjoint' or 'direct'
    'maximum_iterations': 25,
    'maximum_divergences': 5,
    'absolute_tolerance': 1e-6,
    'relative_tolerance': 1e-6,
    'maximum_relative_change': None,
    'error_on_nonconvergence': False,
    'is_symmetric_form_dFdu': True,
    }


### Fabricate measurements

# Box problem domain
W, L, H = 2.0, 1.0, 1.0

# Maximum horizontal displacement of right-face
if SMALL_DISPLACEMENTS:
    uxD_max = 1e-5 # Small displacement case
else:
    uxD_max = 1e-1 # Large displacement case

# Fabricated model parameters (these parameters will need to be predicted)
E_target, nu_target = 1.0, 0.3

# NOTE: The predicted model parameters will be close to the target model
# parameters when the displacements are small. This is consistent with the
# hyper-elastic model approaching the linear-elastic model in the limit of
# small strains. The difference between the solutions will be greater for
# larger displacements.

ex_max = uxD_max / W # Assume strain
Tx_max = E_target * ex_max # Traction

# Fabricate boundary displacements (Dirichlet boundary conditions)
measurement_uxD_bnd = np.linspace(0, uxD_max, NUM_OBSERVATIONS+1)[1:]

# Generate displacement measurement sample points in 2D space

x = np.linspace(0.0, W, 10)
y = np.linspace(0.0, L, 10)

x, y = np.meshgrid(x, y)

x = x.reshape(-1)
y = y.reshape(-1)

# Top surface sample points
measurements_x_smp = np.stack([x,y], axis=1)
measurements_u_smp = []

for value_i in measurement_uxD_bnd:
    measurements_u_smp.append(np.array([
        [value_i*x[0]/W,
         -nu_target*value_i/W*x[1],
         -nu_target*value_i/W*H]
        for x in measurements_x_smp]))

measurement_Tx_bnd = np.linspace(0, Tx_max, NUM_OBSERVATIONS+1)[1:]
measurement_Ty_bnd = np.zeros((NUM_OBSERVATIONS,), float)
measurement_Tz_bnd = np.zeros((NUM_OBSERVATIONS,), float)

measurements_T_bnd = np.stack([
    measurement_Tx_bnd,
    measurement_Ty_bnd,
    measurement_Tz_bnd], axis=1)


### Project displacement measurements on a function space in 2D

nx_msr = 50
ny_msr = max(round(nx_msr/W*L), 1)

mesh_msr = dolfin.RectangleMesh(
    dolfin.Point(0,0), dolfin.Point(W,L), nx_msr, ny_msr)

V_msr = dolfin.VectorFunctionSpace(mesh_msr, 'CG', 1, dim=3)

# Project point-values onto a continious function space using meshless
measurements_u_smp_projected = invsolve.project.project_pointvalues_on_functions(
    measurements_x_smp, measurements_u_smp, V_msr, MESHLESS_DEGREE, MESHLESS_WEIGHT)

u_msr  = invsolve.measure.measurement_expression(measurements_u_smp_projected)
uxD_msr = invsolve.measure.measurement_expression(measurement_uxD_bnd)
T_msr = invsolve.measure.measurement_expression(measurements_T_bnd)

def measurement_setter(i):
    '''Set measurements at index `i`.'''
    T_msr.at_index(i)
    u_msr.at_index(i)
    uxD_msr.at_index(i)

using_subdims_u_msr = [0, 1] # [0, 1], [0, 1, 2]
using_subdims_T_msr = [0]


### Mesh for hyperelastic solid

nz = 10
nx = max(round(nz/H*W), 1)
ny = max(round(nz/H*L), 1)

mesh = dolfin.BoxMesh(dolfin.Point(0,0,0), dolfin.Point(W,L,H), nx, ny, nz)

# Define the fixed boundaries and measurement subdomains

boundary_fix_u = dolfin.CompiledSubDomain(f'on_boundary && near(x[0], {0.0})')
boundary_msr_T = dolfin.CompiledSubDomain(f'on_boundary && near(x[0], {W})')
boundary_msr_u = dolfin.CompiledSubDomain(f'on_boundary && near(x[2], {H})')

fixed_vertex_000 = dolfin.CompiledSubDomain(
    f'near(x[0], {0.0}) && near(x[1], {0.0}) && near(x[2], {0.0})')

fixed_vertex_010 = dolfin.CompiledSubDomain(
    f'near(x[0], {0.0}) && near(x[1], {L}) && near(x[2], {0.0})')

# Mark the elemental entities (e.g. cells, facets) belonging to boundaries

domain_dim = mesh.geometry().dim()
boundary_dim = domain_dim - 1

boundary_markers = dolfin.MeshFunction('size_t', mesh, boundary_dim)
boundary_markers.set_all(0) # Assign all elements the default value

id_subdomain_fix_u = 1 # Fixed boundary id
id_subdomain_msr_T = 2 # Loaded boundary id
id_subdomain_msr_u = 3 # Displacement field measurement boundary id

boundary_fix_u.mark(boundary_markers, id_subdomain_fix_u)
boundary_msr_T.mark(boundary_markers, id_subdomain_msr_T)
boundary_msr_u.mark(boundary_markers, id_subdomain_msr_u)


### Integration measures

dx = dolfin.dx(domain=mesh) # for the whole domain
ds = dolfin.ds(domain=mesh) # for the entire boundary

ds_msr_T = dolfin.Measure('ds', mesh,
    subdomain_id=id_subdomain_msr_T,
    subdomain_data=boundary_markers)

ds_msr_u = dolfin.Measure('ds', mesh,
    subdomain_id=id_subdomain_msr_u,
    subdomain_data=boundary_markers)


### Finite element function spaces

V = dolfin.VectorFunctionSpace(mesh, 'CG', FINITE_ELEMENT_DEGREE)

# Displacement field
u = Function(V)


### Dirichlet boundary conditions

bcs = []

Vx, Vy, Vz = V.split()

zero  = Constant(0)
zeros = Constant((0,0,0))

bcs.append(DirichletBC(Vx, zero, boundary_markers, id_subdomain_fix_u))
bcs.append(DirichletBC(Vx, uxD_msr, boundary_markers, id_subdomain_msr_T))

bcs.append(DirichletBC(V, zeros, fixed_vertex_000, "pointwise"))
bcs.append(DirichletBC(Vz, zero, fixed_vertex_010, "pointwise"))


### Define hyperelastic material model

material_parameters = {'E': Constant(E_target*0.5),
                       'nu': Constant(nu_target*0.5)} # Guess values

E, nu = material_parameters.values()

d = len(u) # Displacement dimension

I = dolfin.Identity(d)
F = dolfin.variable(I + dolfin.grad(u))

C  = F.T*F
J  = dolfin.det(F)
I1 = dolfin.tr(C)

# Lame material parameters
lm = E*nu/((1.0 + nu)*(1.0 - 2.0*nu))
mu = E/(2.0 + 2.0*nu)

# Energy density of a Neo-Hookean material model
psi = (mu/2.0) * (I1 - d - 2.0*dolfin.ln(J)) + (lm/2.0) * dolfin.ln(J) ** 2

# First Piola-Kirchhoff
pk1 = dolfin.diff(psi, F)

# Boundary traction
N = dolfin.FacetNormal(mesh)
PN = dolfin.dot(pk1, N)

# Potential energy
Pi = psi*dx # NOTE: There is no external force potential

# Equilibrium problem
F = dolfin.derivative(Pi, u)


### Model cost and constraints

# Observed displacement
u_obs = u # NOTE: Generally a vector-valued sub-function

# Observed tractions
T_obs = PN # NOTE: Generally a sequence of vector-valued tractions

# Introduce dummy (zero-valued) noise variables for sensitivity analysis
du_msr_noise = Function(V)
dT_msr_noise = Constant((0,)*len(u))

# Superpose dummy noise
u_msr_noisy = u_msr + du_msr_noise
T_msr_noisy = T_msr + dT_msr_noise

# Displacement misfit cost
J_u = sum((u_obs[i]-u_msr_noisy[i])**2 * ds_msr_u
          for i in using_subdims_u_msr)

# Reaction force constraint
C = [(T_obs[i]-T_msr_noisy[i]) * ds_msr_T
     for i in using_subdims_T_msr]

if FORCE_COST_FORMULATION_METHOD == "cost":

    constraint_multipliers = []

    Q = J_u
    L = C[0]

    # NOTE: The final objective to be minimized will effectively be like:
    # J = Q + 0.5*L*L

elif  FORCE_COST_FORMULATION_METHOD == "constraint":

    constraint_multipliers = [Constant(1e-9) for _ in using_subdims_T_msr]
    J_c = sum(mult_i*C_i for mult_i, C_i in zip(constraint_multipliers, C))

    Q = J_u + J_c
    L = None

else:
    raise ValueError('Parameter `FORCE_COST_FORMULATION_METHOD ')


### Inverse problem

model_parameters = [material_parameters]
model_parameters.append(constraint_multipliers)
observation_times = range(0, NUM_OBSERVATIONS)

inverse_solver_basic = invsolve.InverseSolverBasic(Q, L, F, u, bcs,
    model_parameters, observation_times, measurement_setter)

inverse_solver = invsolve.InverseSolver(inverse_solver_basic,
    u_obs, u_msr, ds_msr_u, T_obs, T_msr, ds_msr_T)

inverse_solver.set_parameters_inverse_solver(parameters_inverse_solver)


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


### Assess model sensitivity

def sensitivity_supremum(dmdv, sup_dv=1):
    '''Assume worst-case measurement perturbations by a unit.'''
    return np.abs(dmdv).sum(axis=1) * sup_dv

def sensitivity_variance(dmdv, var_dv=1):
    '''Assume identical and independent variance in the measurements.'''
    return (dmdv**2).sum(axis=1) * var_dv

def sensitivity_stddev(dmdv, std_dv=1):
    '''Assume identical and independent standard deviation in the measurements.'''
    return np.sqrt((dmdv**2).sum(axis=1)) * std_dv

# Initiate functions for sensitivity analysis
inverse_solver.init_observe_dmdu_msr(v=du_msr_noise, ignore_dFdv=True)
inverse_solver.init_observe_dmdT_msr(v=dT_msr_noise, ignore_dFdv=True)

# Model parameter sensitivities wrt displacement field measurements
dmdu_msr = [[inverse_solver.observe_dmdu_msr(t)[i_msr]
            for t in inverse_solver.observation_times]
            for i_msr in range(inverse_solver.num_u_msr)]

# Model parameter sensitivities wrt boundary force measurements
dmdf_msr = [[inverse_solver.observe_dmdf_msr(t)[i_msr]
            for t in inverse_solver.observation_times]
            for i_msr in range(inverse_solver.num_f_msr)]

senssup_dmdu_msr = [[sensitivity_supremum(dmdu_msr_t)
                    for dmdu_msr_t in dmdu_msr_i]
                    for dmdu_msr_i in dmdu_msr]

sensvar_dmdu_msr = [[sensitivity_variance(dmdu_msr_t)
                    for dmdu_msr_t in dmdu_msr_i]
                    for dmdu_msr_i in dmdu_msr]

sensstd_dmdu_msr = [[sensitivity_stddev(dmdu_msr_t)
                    for dmdu_msr_t in dmdu_msr_i]
                    for dmdu_msr_i in dmdu_msr]

sensmag_dmdf_msr = [[np.sqrt(sum(dmdf_msr_t[:,i_dim]**2
                    for i_dim in using_subdims_T_msr))
                    for dmdf_msr_t in dmdf_msr_i]
                    for dmdf_msr_i in dmdf_msr]


### Assess cost condition number

D2JDm2 = inverse_solver.view_cumsum_D2JDm2()
cond_D2JDm2 = np.linalg.cond(D2JDm2)


### Assess model

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
    [measurement_uxD_bnd[t] for t in observation_times]

senssup_dmdu_msr_i = senssup_dmdu_msr[i_msr_u]
sensvar_dmdu_msr_i = sensvar_dmdu_msr[i_msr_u]
sensstd_dmdu_msr_i = sensstd_dmdu_msr[i_msr_u]
sensmag_dmdf_msr_i = sensmag_dmdf_msr[i_msr_f]

# Model parameter sesitivities at nodes
dmdu_msr_i = dmdu_msr[i_msr_u][i_time]


### Plotting

# Model parameter names to be used in labeling plots
model_parameter_names = list(material_parameters.keys())

if len(constraint_multipliers) > 1:
    model_parameter_names.extend([f'constraint_multiplier_{i}'
        for i in range(1, len(constraint_multipliers)+1)])
elif len(constraint_multipliers) == 1:
    model_parameter_names.append('constraint_multiplier')

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

    fig_handle_and_name_pairs.append(
        examples.plotting.plot_model_parameter_sensitivities(
            sensstd_dmdu_msr_i,
            model_parameter_names,
            observation_times,
            figname="Model Parameter Sensitivities wrt Displacement Measurements (Absolute)",
            ylabel="Model parameter sensitivity, $std(m_i)$",
            title="Standard Deviation in Model Parameters Assuming One\n"
                "Standard Deviation in Displacement Measurements"))

    fig_handle_and_name_pairs.append(
        examples.plotting.plot_model_parameter_sensitivities(
            sensmag_dmdf_msr_i,
            model_parameter_names,
            observation_times,
            figname="Model Parameter Sensitivitiesd wrt Force Measurements (Absolute)",
            ylabel="Model parameter sensitivity, $std(m_i)$",
            title=("Standard Deviation in Model Parameters Assuming One\n"
                "Standard Deviation in Reaction Force Measurements")))

    return fig_handle_and_name_pairs


if __name__ == '__main__':

    plt.interactive(True)

    # Model parameter sensitivities as functions
    func_dmdu_msr_i = [Function(V) for _ in \
        range(inverse_solver.num_model_parameters)]

    for func_dmjdu_msr_i, dmjdu_msr_i in zip(func_dmdu_msr_i, dmdu_msr_i):
        func_dmjdu_msr_i.vector()[:] = dmjdu_msr_i

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
    print(f'{dolfin.norm(u):.5e}')


    if TEST_MODEL_PARAMETER_SELF_SENSITIVITIES:
        logger.info('Test model parameter self-sensitivities')

        _dmdm_predicted, _dmdm_expected = inverse_solver \
            .test_model_parameter_sensitivity_dmdm()

        if np.allclose(_dmdm_predicted, _dmdm_expected, atol=1e-4):
            logger.info('Model parameter self-sensitivity test [PASSED]')
        else:
            logger.error('Model parameter self-sensitivity test [FAILED]')

        print('Expected model parameter self-sensitivities:')
        print(_dmdm_expected)
        print('Computed model parameter self-sensitivities:')
        print(_dmdm_predicted)
        print()

    if TEST_SENSITIVITY_REACTION_MEASUREMENTS:
        logger.info('Test reaction measurement sensitivity')

        # Uniform perturbation of reaction (traction) measurements
        perturb_T_msr = np.array([0.1*Tx_max, 0.0, 0.0])

        m0 = np.array(inverse_solver.view_model_parameter_values())

        dm = sum(inverse_solver.observe_dmdT_msr(t)[i_msr_f]
                 for t in inverse_solver.observation_times).dot(perturb_T_msr)

        dT_msr_noise.assign(dolfin.Constant(perturb_T_msr))

        n, b = inverse_solver.solve_inverse_problem() # Default times
        if not b: logger.error('Inverse solver did not converge')

        m1 = np.array(inverse_solver.view_model_parameter_values())

        passed_test_sensitivity_reaction_force = \
            np.allclose(m1 - m0, dm, atol=1e-2*np.abs(dm).max())

        if passed_test_sensitivity_reaction_force:
            logger.info('Reaction measurement sensitivity test [PASSED]')
        else:
            logger.error('Reaction measurement sensitivity test [FAILED]')

        print('Reference model parameter values:')
        print(m0)
        print('Estimated model parameter values:')
        print(m0+dm)
        print('Perturbed model parameter values:')
        print(m1)
        print()

        # Reset reference model state
        dT_msr_noise.assign(dolfin.Constant([0,0,0]))
        inverse_solver.assign_model_parameters(m0)
        inverse_solver.solve_inverse_problem()


    if TEST_SENSITIVITY_DISPLACEMENT_MEASUREMENTS:
        logger.info('Test displacement measurement sensitivity')

        # Uniform perturbation of all displacements
        perturb_u_msr = np.full((u.function_space().dim(),), 0.1*uxD_max)

        m0 = np.array(inverse_solver.view_model_parameter_values())

        dm = sum(inverse_solver.observe_dmdu_msr(t)[i_msr_u]
                 for t in inverse_solver.observation_times).dot(perturb_u_msr)

        du_msr_noise.vector().set_local(perturb_u_msr)

        n, b = inverse_solver.solve_inverse_problem() # Default times
        if not b: logger.error('Inverse solver did not converge')

        m1 = np.array(inverse_solver.view_model_parameter_values())

        passed_test_sensitivity_displacements = \
            np.allclose(m1 - m0, dm, atol=1e-2*np.abs(dm).max())

        if passed_test_sensitivity_displacements:
            logger.info('Displacement measurement sensitivity test [PASSED]')
        else:
            logger.error('Displacement measurement sensitivity test [FAILED]')

        print('Reference model parameter values: ')
        print(m0)
        print('Estimated model parameter values: ')
        print(m0+dm)
        print('Perturbed model parameter values: ')
        print(m1)
        print()

        # Reset reference model state
        du_msr_noise.vector()[:] = 0.0
        inverse_solver.assign_model_parameters(m0)
        inverse_solver.solve_inverse_problem()


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

            outfile = dolfin.File(os.path.join(RESULTS_DIR,'pvd','u.pvd'))
            for t in inverse_solver.observation_times:
                outfile << inverse_solver.observe_u(t, copy=False)
