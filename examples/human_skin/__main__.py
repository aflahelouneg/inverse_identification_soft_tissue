'''
Keloid skin extension problem.

The fixed pad (right pad) is fixed at zero displacement; the moving pad (left
pad) is subjected to incremental displacement. The reaction force is measured
at the moving pad.

'''

import os
import sys
import math
import time
import logging
import numpy as np
import scipy.interpolate
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

import problems.plotting
import problems.utility

from . import config

logger = logging.getLogger()
logger.setLevel(logging.INFO)

reload_module = problems.utility.reload_module
SimpleTimer = problems.utility.SimpleTimer

PROBLEM_DIRECTORY = os.path.dirname(os.path.relpath(__file__))
PROBLEM_NAME = os.path.basename(os.path.split(__file__)[0])


### Problem parameters

# PROBLEM_SUBCASE = "monolithic"
# MATERIAL_MODEL_NAMES = ["NeoHookean"]

PROBLEM_SUBCASE = "bimaterial"
MATERIAL_MODEL_NAMES = ["NeoHookean", "NeoHookean"]
# MATERIAL_MODEL_NAMES = ["Yeoh", "NeoHookean"]

MESH_NAME_TAG = "1" # "1", "2", "3"

MAXIMUM_OBSERVATIONS = 10
# MAXIMUM_OBSERVATIONS = 3

MAXIMUM_DISPLACEMENT = 3 # NOTE: # Maximum displacement is `4.112`
# MAXIMUM_DISPLACEMENT = 4

FIXED_EXTERNAL_BOUNDARY = False
PENALIZE_MATERIAL_PARAMETERS = False

COMPUTE_INITIAL_COST = False
COMPUTE_FINAL_COST = True

COMPUTE_SENSITIVITIES = True
COMPUTE_REACTION_FORCE = True
COMPUTE_MISFIT_ERROR = True
COMPUTE_MISFIT_FIELD = True
COMPUTE_STRESS_FIELD = True

OPTIMIZE_FOREACH_OBSERVATION_TIME = False
OPTIMIZE_FORALL_OBSERVATION_TIMES = True

TEST_SENSITIVITY = False
TEST_SENSITIVITY_PROJECTION = False

ELEMENT_DEGREE = 1
MESHLESS_DEGREE = 2 # (3 is ill-conditioned)
MESHLESS_WEIGHT = "center" # "center", "uniform"

PLOT_RESULTS = False
SAVE_RESULTS = False

SAVE_FIGURE_EXTENSIONS = ('.png', '.svg') # '.pdf'

if not (isinstance(MATERIAL_MODEL_NAMES, (list, tuple)) and \
        all(isinstance(name_i, str) for name_i in MATERIAL_MODEL_NAMES)):
    raise ValueError('Expected `MATERIAL_MODEL_NAMES` to be a sequence of `str`s.')

RESULTS_DIRECTORY_PARENT = os.path.join(
    "results", PROBLEM_NAME + " (temp)",
    f"subcase({PROBLEM_SUBCASE})" +
    f"-material({'_'.join(MATERIAL_MODEL_NAMES)})")

SAFE_TO_REMOVE_FILE_TYPES = \
    ('.out', '.npy', '.pvd', '.vtu', '.png', '.svg', '.eps', '.pdf')

EPS = 1e-12


### Load mesh

if PROBLEM_SUBCASE == "monolithic":

    from .monolithic import mesh as m_mesh

    if   MESH_NAME_TAG == "1": meshfiles_subdir = os.path.join('msh', '4528')
    elif MESH_NAME_TAG == "2": meshfiles_subdir = os.path.join('msh', '6631')
    elif MESH_NAME_TAG == "3": meshfiles_subdir = os.path.join('msh', '12023')
    elif MESH_NAME_TAG == "4": meshfiles_subdir = os.path.join('msh', '26003')
    elif MESH_NAME_TAG == "5": meshfiles_subdir = os.path.join('msh', '103672')
    else: raise ValueError("`MESH_NAME_TAG`?")

elif PROBLEM_SUBCASE == "bimaterial":

    from .bimaterial import mesh as m_mesh

    if   MESH_NAME_TAG == "1": meshfiles_subdir = os.path.join('msh', '4543')
    elif MESH_NAME_TAG == "2": meshfiles_subdir = os.path.join('msh', '6760')
    elif MESH_NAME_TAG == "3": meshfiles_subdir = os.path.join('msh', '12125')
    elif MESH_NAME_TAG == "4": meshfiles_subdir = os.path.join('msh', '26615')
    elif MESH_NAME_TAG == "5": meshfiles_subdir = os.path.join('msh', '105167')
    else: raise ValueError("`MESH_NAME_TAG`?")

else:
    raise ValueError("`PROBLEM_SUBCASE`?")

mesh_data = m_mesh.load_mesh(meshfiles_subdir)

mesh                     = mesh_data['mesh']
domain_markers           = mesh_data['domain_markers']
boundary_markers         = mesh_data['boundary_markers']
id_subdomains_material   = mesh_data['id_subdomains_material']
id_subdomains_dic        = mesh_data['id_subdomains_dic']
id_boundaries_pad_moving = mesh_data['id_boundaries_pad_moving']
id_boundaries_pad_fixed  = mesh_data['id_boundaries_pad_fixed']
id_boundaries_exterior   = mesh_data['id_boundaries_exterior']

# NOTE: The marker id's of material subdomains (`id_subdomains_material`) can be
#       defined as a sequence of sequences's. In this case, `id_subdomains_material
#       [I][J]` shall refer to the `I`th material, and the `J`th material subdomain.
#       Alternatively, `id_subdomains_material` can be defined as a sequence of
#       int's. In this case, `id_subdomains_material[I]` shall refer to the `I`th
#       material in the `I`th subdomain. Some practical examples are as follows:
#
#       1) First material defined on subdomain `1`:
#          id_subdomains_material = [(1,),] OR [1,]
#       2) First material defined on subdomains `1` and `2`
#          id_subdomains_material = [(1,2),]
#       3) First material defined on subdomain `1`, second material -- `2`
#          id_subdomains_material = [(1,),(2,)] OR [1, 2]


### Load measurements

if PROBLEM_SUBCASE == "monolithic":
    from .monolithic.data.displacement import measurements as measurements_disp
    from .monolithic.data.reactionforce import measurements as measurements_force

elif PROBLEM_SUBCASE == "bimaterial":
    from .bimaterial.data.displacement import measurements as measurements_disp
    from .bimaterial.data.reactionforce import measurements as measurements_force

else:
    raise ValueError("`PROBLEM_SUBCASE`?")

x_dic             = measurements_disp['x_dic']
u_dic             = measurements_disp['u_dic']
u_dic_pad_moving  = measurements_disp['u_pad_mov']

ux_msr_pad_moving = measurements_force['ux_pad_mov']
fx_msr_pad_moving = measurements_force['fx_pad_mov']

if not all(isinstance(mtx, np.ndarray) and mtx.ndim == 2 for mtx in u_dic):
    raise TypeError('Expecting `u_dic` to be a sequence of 2D arrays.')

if not all(isinstance(vec, np.ndarray) and vec.ndim == 1 for vec in u_dic_pad_moving):
    raise TypeError('Expecting `u_dic_pad_moving` to be a sequence of 1D arrays.')


### Synchronize force measurements with DIC

# Trim the measurement that contains too much data
if u_dic_pad_moving[-1][0] < ux_msr_pad_moving[-1]:

    # Too much data at the end of `u_dic_pad_moving`
    mask = u_dic_pad_moving[:,0] >= ux_msr_pad_moving[-1]
    idx_end = np.flatnonzero(mask)[-1] + 2

    u_dic_pad_moving = u_dic_pad_moving[:idx_end,:]
    u_dic = u_dic[:idx_end]

elif ux_msr_pad_moving[-1] < u_dic_pad_moving[-1][0]:

    # Too much data at the end of `ux_msr_pad_moving`
    mask = ux_msr_pad_moving >= u_dic_pad_moving[-1,0]
    idx_end = np.flatnonzero(mask)[-1] + 2

    ux_msr_pad_moving = ux_msr_pad_moving[:idx_end]
    fx_msr_pad_moving = fx_msr_pad_moving[:idx_end]

interp_fx_pad_moving = scipy.interpolate.interp1d(
    ux_msr_pad_moving, fx_msr_pad_moving,
    kind='linear', fill_value="extrapolate")

ux_msr_pad_moving = np.asarray(u_dic_pad_moving)[:,0]
fx_msr_pad_moving = interp_fx_pad_moving(ux_msr_pad_moving)

uy_msr_pad_moving = np.zeros_like(ux_msr_pad_moving)
fy_msr_pad_moving = np.zeros_like(fx_msr_pad_moving)

if PROBLEM_SUBCASE == "monolithic":

    def coordinates_of_moving_pad():
        return np.array([[32.0, 20.0]])

elif PROBLEM_SUBCASE == "bimaterial":

    def coordinates_of_moving_pad():
        '''Since the DIC coordinates do not match the mesh coordinates,
        compute the required vector for offsetting the DIC coordinates.

        Assuming the DIC coordinates are relative the moving pad.
        Assuming the pads are equidistant from the mesh center.

        '''

        PAD_SEPARATION_DISTANCE = 4.072727e+01
        # This number was estimated from DIC measurements.
        # The precise pad separation distance is unknown.

        x = mesh.coordinates(); x_mesh_center = (x.max(0) + x.min(0)) * 0.5
        x_pad_center = x_mesh_center - [PAD_SEPARATION_DISTANCE * 0.5, 0.0]

        return x_pad_center

else:
    raise ValueError("`PROBLEM_SUBCASE`?")

# NOTE: All measurements should be full dimension. The relevant subdimension(s)
#       can be specified in the definitions of the model cost and constraint(s).

measurement_x_dic = x_dic + coordinates_of_moving_pad()
measurement_u_dic = u_dic

measurement_u_bnd = np.stack([ux_msr_pad_moving, uy_msr_pad_moving], axis=1)
measurement_f_bnd = np.stack([fx_msr_pad_moving, fy_msr_pad_moving], axis=1)

num_measurements = len(measurement_f_bnd)

if not (num_measurements == len(measurement_u_bnd) == len(measurement_u_dic)):
    raise RuntimeError('Numbers of measurements need to be the same.')


### Consider displacement measurements within bounds

if MAXIMUM_DISPLACEMENT is not None:

    assert MAXIMUM_DISPLACEMENT > 0, "Expected a positive value."

    ind = np.flatnonzero((measurement_u_bnd**2).sum(1)
                          > MAXIMUM_DISPLACEMENT**2)

    if ind.size:

        num_measurements = ind[0]
        if num_measurements == 0:
            raise RuntimeError

        measurement_u_dic = measurement_u_dic[:num_measurements]
        measurement_u_bnd = measurement_u_bnd[:num_measurements]
        measurement_f_bnd = measurement_f_bnd[:num_measurements]


### Observation times

# NOTE: Prefer observation times as a sequence of indices.
# NOTE: Avoid time `0` if the deformation is zero.

model_observation_start = 1

model_observation_times = problems.utility.linspace_range(
    first=model_observation_start, last=num_measurements-1,
    count=min(num_measurements, MAXIMUM_OBSERVATIONS),
    start_from="back")


### Mark the DIC subdomain

def compute_measurement_markers_dic():
    '''Mark the elements in the mesh that are overlain by DIC measurements.'''

    p0 = measurement_x_dic.min(axis=0)
    p1 = measurement_x_dic.max(axis=0)

    tol = np.abs(p1-p0).max() * EPS

    p0 -= tol
    p1 += tol

    measurement_markers_dic, id_measruement_markers_dic = \
        problems.utility.mark_rectangular_subdomain(p0, p1, mesh)

    id_measruement_markers_dic = (id_measruement_markers_dic,)

    if not measurement_markers_dic.array().any():
        raise RuntimeError('Null measurement markers')

    return measurement_markers_dic, id_measruement_markers_dic

measurement_markers_dic, id_measruement_markers_dic = \
    compute_measurement_markers_dic()


### Integration measures

dx = dolfin.dx(domain=mesh) # Entire domain
ds = dolfin.ds(domain=mesh) # Entire boundary

# Integration over material subdomains
dx_mat = tuple(dolfin.Measure('dx', mesh, subdomain_data=domain_markers,
    subdomain_id=ids_mat_i) for ids_mat_i in id_subdomains_material)

# Integration over measurement subdomains: one subdomain
dx_msr = (dolfin.Measure('dx', mesh, subdomain_data=measurement_markers_dic,
    subdomain_id=id_measruement_markers_dic),)

# Integration over the measurement boundary: at least one subdomain
ds_msr = (tuple(dolfin.Measure('ds', mesh, subdomain_data=boundary_markers,
    subdomain_id=id) for id in id_boundaries_pad_moving),)

# NOTE: `ds_msr` generally contains boundary measures that are split among
#       different material subdomains. This is necessary for the integration
#       of the observed traction since it may be defined on several materials.

domain_size = assemble(1*dx)

if abs(sum(assemble(1*dx_i) for dx_i in dx_mat)-domain_size) > domain_size*EPS:
    raise RuntimeError('Material domain(s) do not constitute geometric domain.')

if any(assemble(1*dx_i) < EPS for dx_i in dx_msr):
    raise RuntimeError('Zero-size measurement subdomain.')

if any(assemble(1*ds_ij) < EPS for ds_i in ds_msr for ds_ij in ds_i):
    raise RuntimeError('Zero-size measurement boundary.')


### Function spaces

# element_u = dolfin.VectorElement("CG", mesh.ufl_cell(), 1)
# element_p = dolfin.FiniteElement("CG", mesh.ufl_cell(), 1)
# mixed_element = dolfin.MixedElement([element_u, element_p])

# V = FunctionSpace(mesh, mixed_element)
# V_obs = FunctionSpace(mesh, element_u)

V = dolfin.VectorFunctionSpace(mesh, 'CG', ELEMENT_DEGREE) # for primary field
S = dolfin.FunctionSpace(mesh, 'CG', ELEMENT_DEGREE) # for generic scalar fields
W = dolfin.TensorFunctionSpace(mesh, 'DG', 0) # for tensor fields (e.g. stresses)

# NOTE: `V` is generally a mixed function space that accounts for all fields,
#       e.g. displacement field, hydro-static pressure field, etc. `V_obs`,
#       on the other hand, must just account for the displacement field.

V_obs = V

# Primary field
u = Function(V)


### Model parameters

# NOTE: Model parameters consist of material parameters any any auxiliary
#       parameters (e.g. constraint multpliers)
#
# NOTE: Model parameters that are exclusively `dolfin.Constant`s are the only
#       parameters that will be optimized.
#
# NOTE: Material parameters should be a `list` of `dict`s so that each `dict`
#       may refer to a particular material subdomain.

auxiliary_parameters = {
    'lagrange_multiplier': 0, # For constraining the model reaction force
                              # to match the measured reaction force.
    }

material_classes = []
material_parameters = []

if PROBLEM_SUBCASE == "monolithic":

    if MATERIAL_MODEL_NAMES[0] == "NeoHookean":

        material_classes.append(material.NeoHookean)

        if FIXED_EXTERNAL_BOUNDARY:

            if   MESH_NAME_TAG == "1": model_parameter_init = (2.963e-02, 1.598e-01,  5.214e-08) # GUESS
            else:
                raise NotImplementedError

        else:

            if   MESH_NAME_TAG == "1": model_parameter_init = (2.963011414641114269e-02, 1.598240659106513750e-01,  5.213673572562297402e-08)
            elif MESH_NAME_TAG == "2": model_parameter_init = (2.944986705163990179e-02, 1.453630485862034694e-01,  3.423335586152275298e-08)
            elif MESH_NAME_TAG == "3": model_parameter_init = (2.838323203496151068e-02, 1.566668537617276202e-01,  5.060665448534135552e-08)
            elif MESH_NAME_TAG == "4": model_parameter_init = (2.803433650745537631e-02, 1.466638526492443639e-01, -1.960886460213584679e-08)
            elif MESH_NAME_TAG == "5": model_parameter_init = (2.679497761354525734e-02, 1.360910810533804971e-01,  5.198569875346106152e-08)
            else:
                raise NotImplementedError

        material_parameters.append({
            'E':  Constant(model_parameter_init[0]),
            'nu': Constant(model_parameter_init[1]),
            })

        auxiliary_parameters['lagrange_multiplier'] = \
            Constant(model_parameter_init[-1])

    else:
        raise NotImplementedError(f'`MATERIAL_MODEL_NAMES`: {MATERIAL_MODEL_NAMES}.')


elif PROBLEM_SUBCASE == "bimaterial":

    if MATERIAL_MODEL_NAMES[0] == "NeoHookean" and \
       MATERIAL_MODEL_NAMES[1] == "NeoHookean":

        material_classes.append(material.NeoHookean)
        material_classes.append(material.NeoHookean)

        if not FIXED_EXTERNAL_BOUNDARY:

            number_of_model_parameters = 3

            if number_of_model_parameters == 3:

                if   MESH_NAME_TAG == "1": model_parameter_init = (1.090813107096536699e-01, 4.062347391656254136e-01, 9.221195915302710534e-10)
                elif MESH_NAME_TAG == "2": model_parameter_init = (9.975743862934521866e-02, 4.166089662522097226e-01, 1.267970122177255419e-14)
                elif MESH_NAME_TAG == "3": model_parameter_init = (1.094745007470644405e-01, 3.893025082154862315e-01, 1.546436898099080772e-09)
                elif MESH_NAME_TAG == "4": model_parameter_init = (1.062777335344494423e-01, 3.706073657389226117e-01, 1.565697153212684302e-09)
                elif MESH_NAME_TAG == "5": model_parameter_init = (1.082390169459683943e-01, 3.505140390616134916e-01, 1.134785448024687904e-13)
                else:
                    raise NotImplementedError(f'`MESH_NAME_TAG`: {MESH_NAME_TAG}')

            elif number_of_model_parameters == 5:

                if MESH_NAME_TAG == "1":
                    model_parameter_init = (
                        8.448515290490056506e-02, # E_keloid
                        4.418101500996806097e-01, # nu_keloid
                        8.839441415078111430e-02, # E_skin
                        9.721150345672150417e-01, # nu_skin
                        9.482169924747038366e-15, # lagrange_multiplier
                        )
                elif MESH_NAME_TAG == "2":
                    model_parameter_init = (
                        7.682032083930719069e-02, # E_keloid
                        4.480419393211982348e-01, # nu_keloid
                        8.038242480796391198e-02, # E_skin
                        9.135268918419600093e-01, # nu_skin
                        1.978373984940486353e-14, # lagrange_multiplier
                        )
                elif MESH_NAME_TAG == "3":
                    model_parameter_init = (
                        1.072104502354688793e-01, # E_keloid
                        3.928123565427622532e-01, # nu_keloid
                        2.981129112394806033e-02, # E_skin
                        1.121347614450838442e-01, # nu_skin
                        1.154366552508418024e-14, # lagrange_multiplier
                        )
                elif MESH_NAME_TAG == "4":
                    model_parameter_init = (
                        1.130818461553569770e-01, # E_keloid
                        3.579165194801819383e-01, # nu_keloid
                        1.492998765400741119e-02, # E_skin
                        2.056669586409408390e-01, # nu_skin
                       -9.755503071082305750e-13, # lagrange_multiplier
                        )
                elif MESH_NAME_TAG == "5":
                    model_parameter_init = (
                        1.158259578773082332e-01, # E_keloid
                        3.354661354469088974e-01, # nu_keloid
                        1.209123148969074718e-02, # E_skin
                        2.009210948264882890e-01, # nu_skin
                       -7.786493276916434244e-13, # lagrange_multiplier
                        )
                else:
                    raise NotImplementedError(f'`MESH_NAME_TAG`: {MESH_NAME_TAG}')

            else:
                raise NotImplementedError('`number_of_model_parameters`: {number_of_model_parameters}')

            material_parameters.append({
                'E':  Constant(model_parameter_init[0]),
                'nu': Constant(model_parameter_init[1]),
                }) # Keloid skin

            auxiliary_parameters['lagrange_multiplier'] = \
                Constant(model_parameter_init[-1])

            if len(model_parameter_init) == 3:

                material_parameters.append({
                    'E':  2.679497761354525734e-02,
                    'nu': 1.360910810533804971e-01,
                    }) # Healthy skin

            elif len(model_parameter_init) == 5:

                material_parameters.append({
                    'E':  Constant(model_parameter_init[2]),
                    'nu': Constant(model_parameter_init[3]),
                    }) # Healthy skin

            else:
                raise NotImplementedError(f'`len(model_parameter_init)`: {len(model_parameter_init)}')

        else:
            raise NotImplementedError(f'`FIXED_EXTERNAL_BOUNDARY`: {FIXED_EXTERNAL_BOUNDARY}')


    elif MATERIAL_MODEL_NAMES[0] == "Yeoh" and \
         MATERIAL_MODEL_NAMES[1] == "NeoHookean":

        material_classes.append(material.Yeoh)
        material_classes.append(material.NeoHookean)

        if not FIXED_EXTERNAL_BOUNDARY:

            if MESH_NAME_TAG: #  == "2"

                model_parameter_init = (
                    1.740154e-03, # C1
                    2.451183e+02, # lagrange_multiplier
                    )

                # model_parameter_init = (
                #     2.900731108327908939e-02, # C1
                #     2.563200442527592937e+00, # C2
                #     2.235629515274709789e+01, # lagrange_multiplier
                #     )

                # model_parameter_init = (
                #     2.900731108327908939e-02, # C1
                #     2.563200442527592937e+00, # C2
                #     0.0, # C3
                #     2.235629515274709789e+01, # lagrange_multiplier
                #     )

            else:
                raise NotImplementedError(f'`MESH_NAME_TAG`: {MESH_NAME_TAG}')

            material_parameters.append({
                f'C{i}': Constant(value) for i, value in
                    enumerate(model_parameter_init[:-1], start=1)
                }) # Keloid skin

            auxiliary_parameters['lagrange_multiplier'] = \
                Constant(model_parameter_init[-1])

            material_parameters.append({
                'E':  2.679497761354525734e-02,
                'nu': 1.360910810533804971e-01,
                }) # Healthy skin

        else:
            raise NotImplementedError(f'`FIXED_EXTERNAL_BOUNDARY`: {FIXED_EXTERNAL_BOUNDARY}')

    else:
        raise NotImplementedError(f'`MATERIAL_MODEL_NAMES`: {MATERIAL_MODEL_NAMES}')

else:
    raise NotImplementedError(f'`PROBLEM_SUBCASE`: "{PROBLEM_SUBCASE}"')


if not isinstance(material_parameters, (list, tuple)):
    material_parameters = (material_parameters,)

if len(material_parameters) != len(id_subdomains_material):
    raise TypeErrpr('Number of materials does not equal '
                    'the number of material subdomains.')

if not all(isinstance(m, dict) for m in material_parameters):
    raise TypeError('`material_parameters` must contain `dict`s.')


model_parameters = [material_parameters,]
model_parameters.append(auxiliary_parameters)

material_parameter_names = problems.utility \
    .list_model_parameter_names(material_parameters, Constant)

auxiliary_parameter_names = problems.utility \
    .list_model_parameter_names(auxiliary_parameters, Constant)

model_parameter_names = material_parameter_names + auxiliary_parameter_names


### Regularization of model parameters (OPTIONAL)

def make_material_parameter_penalty(parameter, parameter_expected_value):
    '''Penalize the difference between parameter value and expected value.'''

    if not isinstance(parameter, Constant):
        raise TypeError('Parameter `parameter`.')

    if not isinstance(parameter_expected_value, Constant):
        parameter_expected_value = Constant(parameter_expected_value)

    return (parameter - parameter_expected_value)**2 * dx

if PENALIZE_MATERIAL_PARAMETERS:

    penalized_parameter_penalty = Constant(1e-2)
    penalized_parameter = material_parameters[-1]['nu']
    penalized_parameter_expected_value = float(penalized_parameter)

    material_parameter_penalty = \
        penalized_parameter_penalty * make_material_parameter_penalty(
            penalized_parameter, penalized_parameter_expected_value)

else:
    material_parameter_penalty = None


### Hyperelastic model

material_models = []

psi = []
pk1 = []
pk2 = []

for material_class_i, material_parameters_i \
    in zip(material_classes, material_parameters):

    material_model_i = material_class_i(material_parameters_i)
    material_model_i.initialize(u) # Defines `psi`, `pk1`, `pk2`

    psi.append(material_model_i.psi[0])
    pk1.append(material_model_i.pk1[0])
    pk2.append(material_model_i.pk2[0])

# Potential energy of the hyperelastic solid
Pi = sum(psi_i*dx_i() for psi_i, dx_i in zip(psi, dx_mat))

# NOTE: Since Neumann BC's are zero, potential `Pi` is just the strain energy.
#       Note that the reaction force measurements are the constraint equations;
#       consequently, they are not part of the Neumann boundary conditions.

N = dolfin.FacetNormal(mesh) # Normal vector to boundary
PN = tuple(dolfin.dot(pk1_mat_i, N) for pk1_mat_i in pk1)


### Measurement expressions

# NOTE: Generally, `ds_msr` is a list of lists. The first index refers to a
#       measurement boundary and the second index to the boundary subdomain.

# Get total boundary by combining subdomains
ds_msr_0 = sum(ds_msr[0][1:], ds_msr[0][0])

measurement_T_bnd = measurement_f_bnd / assemble(1*ds_msr_0)
# NOTE: We require average traction on measurement boundary

# Project point-values onto a continious function space using meshless
measurement_u_dic_prj = invsolve.project.project_pointvalues_on_functions(
    measurement_x_dic, measurement_u_dic, V_obs, MESHLESS_DEGREE, MESHLESS_WEIGHT)

u_msr  = invsolve.measure.measurement_expression(measurement_u_dic_prj)
uD_msr = invsolve.measure.measurement_expression(measurement_u_bnd)
T_msr  = invsolve.measure.measurement_expression(measurement_T_bnd)

def measurement_setter(i):
    '''Set measurements at index.'''
    T_msr.at_index(i)
    u_msr.at_index(i)
    uD_msr.at_index(i)


### Dirichlet boundary conditions

bcs = []

zeros = Constant((0.0,0.0))

bcs.extend([DirichletBC(V, zeros, boundary_markers, i)
            for i in id_boundaries_pad_fixed])

bcs.extend([DirichletBC(V, uD_msr, boundary_markers, i)
            for i in id_boundaries_pad_moving])

if FIXED_EXTERNAL_BOUNDARY:
    bcs.extend([DirichletBC(V, zeros, boundary_markers, i)
                for i in id_boundaries_exterior])


### Model cost and constraints

using_subdims_u_msr = (0, 1)
using_subdims_T_msr = (0,)

u_obs = u # Observed displacement (generally a sub-function)
T_obs = [PN,] # Observation of tractions (generally a sequence)

# NOTE: `T_obs` is generally a sequence of sequences. This is to allow the
#       specification of multiple tractions on multiple boundary subdomains.
#       An example case is a traction defined on a multi-material boundary.

J, du_msr = invsolve.functions.cost_displacement_misfit_noisy(
    u_obs, u_msr, dx_msr, subdims=using_subdims_u_msr)

C, dT_msr = invsolve.functions.constraints_reaction_force_noisy(
    T_obs, T_msr, ds_msr, subdims=using_subdims_T_msr)

J += auxiliary_parameters['lagrange_multiplier'] * C[0]

if material_parameter_penalty is not None:
    J += material_parameter_penalty


### Inverse solver

F = dolfin.derivative(Pi, u)

inverse_solver_basic = invsolve.InverseSolverBasic(J, F, u, bcs,
    model_parameters, model_observation_times, measurement_setter)

inverse_solver = invsolve.InverseSolver(inverse_solver_basic,
    u_obs, u_msr, dx_msr, T_obs, T_msr, ds_msr)

# Update inverse solver parameters

inverse_solver.set_parameters_inverse_solver(
    config.parameters_inverse_solver)

inverse_solver.set_parameters_nonlinear_solver(
    config.parameters_nonlinear_solver)

# Initiate functions for sensitivity analysis

inverse_solver.init_observe_dmdu_msr(v=du_msr, ignore_dFdv=True)
inverse_solver.init_observe_dmdT_msr(v=dT_msr, ignore_dFdv=True)


### Custom model parameter update function

# def compute_force_constraint():
#     '''Constraint: Average reaction force on moving pad in the x-axis.'''
#     # return sum(inverse_solver.observe_dfdm(inverse_solver.observation_times)[0])[0,:]
#     return inverse_solver.observe_dfdm(inverse_solver.observation_times[-1])[0][0,:]

# def _compute_dm_method_gradient(DJDm, D2JDm2):

#     C = compute_force_constraint()

#     dm = -DJDm

#     dm -= C * (C.dot(dm)/C.dot(C))

#     d2J = D2JDm2.dot(dm).dot(dm)

#     if d2J > 0:
#         dm *= (dm.dot(dm)/d2J)
#     else:
#         dm /= math.sqrt(dm.dot(dm))
#         dm *= 0.5

#     return dm

# inverse_solver._compute_dm_method_gradient = _compute_dm_method_gradient
# inverse_solver.parameters_inverse_solver['solver_method'] = "gradient"
# inverse_solver.parameters_inverse_solver['maximum_relative_change'] = 0.05


### Solve inverse problem

if COMPUTE_INITIAL_COST:
    with SimpleTimer('Compute initial cost values at observation times'):
        cost_values_initial, cost_gradients_initial = inverse_solver \
            .assess_model_cost(model_observation_times, compute_gradients=False)
else:
    cost_values_initial = None
    cost_gradients_initial = None


if OPTIMIZE_FOREACH_OBSERVATION_TIME:

    with SimpleTimer('Optimize model parameters at each observation time'):
        model_parameters_foreach, iterations_count_foreach, is_converged_foreach = \
            inverse_solver.fit_model_foreach_time() # Using model observation times

    model_parameters_foreach_T = tuple(m_i.tolist()
        for m_i in np.array(model_parameters_foreach, order='F').T)

    # Minimizing cost for all observation times may be faster if
    # the model parameters for each observation time are averaged.

    model_parameters_foreach_mean = \
        tuple(sum(m)/len(m) for m in model_parameters_foreach_T)

    inverse_solver.assign_model_parameters(model_parameters_foreach_mean)

else:
    model_parameters_foreach = None


if OPTIMIZE_FORALL_OBSERVATION_TIMES:
    with SimpleTimer('Optimize model parameters for all observation times'):
        model_parameters_forall, iterations_count_forall, is_converged_forall = \
            inverse_solver.fit_model_forall_times() # Using model observation times
else:
    model_parameters_forall = None


if COMPUTE_FINAL_COST:
    with SimpleTimer('Compute final cost values at observation times'):
        cost_values_final, cost_gradients_final = inverse_solver \
            .assess_model_cost(model_observation_times, compute_gradients=True)
else:
    cost_values_final = None
    cost_gradients_final = None


if COMPUTE_MISFIT_ERROR:
    with SimpleTimer('Model misfit at observation times'):

        misfit_displacements = inverse_solver.assess_misfit_displacements(
            model_observation_times, subdims=using_subdims_u_msr)

        # NOTE: Value at indices `[I][J]` refers to:
        #       `I`th measurement, `J`th observation time.

        misfit_reaction_forces = inverse_solver.assess_misfit_reaction_forces(
            model_observation_times, subdims=using_subdims_T_msr)

        # NOTE: Value at indices `[I][J][K]` refers to:
        #       `I`th measurement, `J`th observation time, and
        #       `K`th reaction force component (dimension).


if COMPUTE_REACTION_FORCE:
    with SimpleTimer('Reaction force misfit at observation times'):

        # NOTE: Each variable here will be a list of lists such that the value at
        #       `[I][J]` refer to the `I`th measurement and `J`th observation time.

        reaction_forces_observed = inverse_solver.observe_f_obs(model_observation_times)
        reaction_forces_measured = inverse_solver.observe_f_msr(model_observation_times)

        reaction_displacements = [
            [uD_msr.at_index(t).get_value() for t in model_observation_times],
            ] # NOTE: Wrapped in lists for consistency


### Assess model sensitivity

def sensitivity_sup(dmdv, sup_dv=1):
    '''Sensitivity measure: SUPREMUM.
    Assuming worst-case measurement perturbations by one unit.'''
    return np.abs(dmdv).sum(axis=1) * sup_dv

def sensitivity_var(dmdv, var_dv=1):
    '''Sensitivity measure: VARIANCE.
    Assume identical and independent variance in the measurements.'''
    return (dmdv**2).sum(axis=1) * var_dv

def sensitivity_std(dmdv, std_dv=1):
    '''Sensitivity measure: STANDARD DEVIATION.
    Assume identical and independent standard deviation in the measurements.'''
    return np.sqrt(sensitivity_var(dmdv, std_dv**2))

if COMPUTE_SENSITIVITIES:
    with SimpleTimer(f'Assess model parameter sensitivities wrt measurements'):

        # NOTE: Each variable here will be a list of lists such that the value at
        #       `[I][J]` refer to the `I`th measurement and `J`th observation time.

        # Model parameter sensitivities wrt DIC measurements
        dmdu_msr = [[inverse_solver.observe_dmdu_msr(t)[i_msr]
                    for t in inverse_solver.observation_times]
                    for i_msr in range(inverse_solver.num_u_msr)]

        # Model parameter sensitivities wrt force measurements
        dmdf_msr = [[inverse_solver.observe_dmdf_msr(t)[i_msr]
                    for t in inverse_solver.observation_times]
                    for i_msr in range(inverse_solver.num_f_msr)]


        senssup_dmdu_msr = [[sensitivity_sup(dmdu_msr_t)
            for dmdu_msr_t in dmdu_msr_i] for dmdu_msr_i in dmdu_msr]

        sensstd_dmdu_msr = [[sensitivity_std(dmdu_msr_t)
            for dmdu_msr_t in dmdu_msr_i] for dmdu_msr_i in dmdu_msr]

        senssup_dmdf_msr = [[sensitivity_sup(dmdf_msr_t)
            for dmdf_msr_t in dmdf_msr_i] for dmdf_msr_i in dmdf_msr]

        sensstd_dmdf_msr = [[sensitivity_std(dmdf_msr_t)
            for dmdf_msr_t in dmdf_msr_i] for dmdf_msr_i in dmdf_msr]


        MINIMUM_NORMALIZATION = 1e-6

        _sensitivity_normalizations = np.abs(
            inverse_solver.view_model_parameter_values())

        _sensitivity_normalizations[_sensitivity_normalizations
            < MINIMUM_NORMALIZATION] = MINIMUM_NORMALIZATION

        senssup_dmdu_msr_relative = [[dmdu_msr_t / _sensitivity_normalizations
            for dmdu_msr_t in dmdu_msr_i] for dmdu_msr_i in senssup_dmdu_msr]

        sensstd_dmdu_msr_relative = [[dmdu_msr_t / _sensitivity_normalizations
            for dmdu_msr_t in dmdu_msr_i] for dmdu_msr_i in sensstd_dmdu_msr]

        senssup_dmdf_msr_relative = [[dmdf_msr_t / _sensitivity_normalizations
            for dmdf_msr_t in dmdf_msr_i] for dmdf_msr_i in senssup_dmdf_msr]

        sensstd_dmdf_msr_relative = [[dmdf_msr_t / _sensitivity_normalizations
            for dmdf_msr_t in dmdf_msr_i] for dmdf_msr_i in sensstd_dmdf_msr]


### Test model parameter sensitivities

def test_model_parameter_sensitivities(h=1e-3):
    '''Finite difference test for verifying model parameter sensitivities.

    Important
    ---------
    The tolerance parameters need to be sufficiently tight, e.g.
    `parameters_inverse_solver['absolute_tolerance'] = 1e-9`,
    `parameters_inverse_solver['relative_tolerance'] = 1e-6`.

    Parameters
    ----------
    h : float
        Finite difference step size in the model parameters (L2-norm).

    Returns
    -------
    test_success : list of bool's
        List of success flags.
    test_results : list of tuple's of numpy.ndarray's
        List of tuples containing the predicted and the expected result.

    '''

    test_successes = []
    test_results = []

    test_results.append(inverse_solver.test_model_parameter_sensitivity_dmdm())
    test_results.append(inverse_solver.test_model_parameter_sensitivity_dmdT_msr(h))
    test_results.append(inverse_solver.test_model_parameter_sensitivity_dmdu_msr(h))

    for res_predicted, res_expected in test_results:
        test_successes.append(np.allclose(res_predicted, res_expected, atol=1e-6, rtol=1e-2))

    return test_successes, test_results

if TEST_SENSITIVITY:

    try:
        test_successes, test_results = test_model_parameter_sensitivities()
    except:
        test_successes, test_results = False, None

    if not all(test_successes):
        logger.error('Failed model parameter sensitivity test(s).')
        if test_results is not None:
            print('\"test_results:\"')
            print(test_results)


### Model observations at specific time

# Observe model at final time
t_obs = model_observation_times[-1]

inverse_solver.update_nonlinear_solution(t_obs)

# Smoothing solver for projecting discrete model parameter sensitivities
smoothing_solver = invsolve.functions.make_smoothing_solver(V_obs, kappa=None)

def compute_correlation_dmdu_msr(dmdu_msr):

    if not isinstance(dmdu_msr, np.ndarray) or dmdu_msr.ndim != 2:
        raise TypeError('Parameter `dmdu_msr`.')

    if dmdu_msr.shape != (inverse_solver.num_model_parameters,
                          du_msr.function_space().dim()):
        raise ValueError('Parameter `dmdu_msr`.')

    correlation_matrix = np.array([[ v_i.dot(v_j)
        / max(math.sqrt(v_i.dot(v_i) * v_j.dot(v_j)), EPS)
        for v_j in dmdu_msr] for v_i in dmdu_msr])

    return correlation_matrix

if COMPUTE_SENSITIVITIES:
    with SimpleTimer(f'Observe model parameter sensitivities (method-1)'):

        # Observe model parameter sensitivities with respect to the displacement
        # field measurements using the method based on the model cost variations.
        #
        # Solve for `dm_` given an arbitrary `du_msr_`
        #   D2J/Dm2 dm_ = - d(DJ/Dm)/du_msr du_msr_
        #
        # NOTE: The sensitivities are defined only on the subdomain(s) where
        # the measurements are made. The sensitivities can be used to estimate
        # the change in model parameters for a given change in the displacement
        # measurements with good accuracy. The sensitivity values are not simple
        # to project on a function space because the values are not exactly the
        # dof values but rather also depend on the size of the nodal support
        # (i.e. local mesh size). In general, a straighforward L2-projection
        # will not be optimal and the resulting field will be quite rough.

        dmdu_msr_at_t_as_array_values = \
            inverse_solver.observe_dmdu_msr(t_obs)

        correlation_dmdu_msr_at_t = [compute_correlation_dmdu_msr(
            dmdu_msr_i) for dmdu_msr_i in dmdu_msr_at_t_as_array_values]

        dmdu_msr_at_t_as_vector_field = \
            [invsolve.functions.project_sensitivities_dmdu_msr(
            dmdu_msr_i, V_obs, smoothing_solver=smoothing_solver)
            for dmdu_msr_i in dmdu_msr_at_t_as_array_values]

        # NOTE: 'Smoothing solver for smoothing model parameter sensitivities
        #       with respect to the displacement field measurements is not
        #       effective if the measurement (sub)domain is a boundary, i.e.
        #       the measurement domain has a smaller dimension than the mesh.

        dmdu_msr_at_t_as_scalar_field = \
            [[dolfin.project(dolfin.sqrt(dmjdu_msr_i**2), S)
            for dmjdu_msr_i in dmdu_msr_i]
            for dmdu_msr_i in dmdu_msr_at_t_as_vector_field]

        for i, dmdu_msr_i in enumerate(dmdu_msr_at_t_as_vector_field):
            for j, dmjdu_msr_i in enumerate(dmdu_msr_i):
                dmjdu_msr_i.rename(f'dm[{j}]/du_msr[{i}]','')

        for i, dmdu_msr_i in enumerate(dmdu_msr_at_t_as_scalar_field):
            for j, dmjdu_msr_i in enumerate(dmdu_msr_i):
                dmjdu_msr_i.rename(f'dm[{j}]/du_msr[{i}]','')

        if TEST_SENSITIVITY_PROJECTION:

            for i in range(inverse_solver.num_u_msr):

                test_success, test_results = \
                    invsolve.functions.test_projected_sensitivities_dmdu_msr(
                    dmdu_msr_at_t_as_vector_field[i], dmdu_msr_at_t_as_array_values[i],
                    rtol=1e-5, atol=1e-8)

                if not test_success:

                    logger.warning("Projection error of discrete sensitivities "
                                   "is not within specified tolerance.")

                    if logger.level <= logging.WARNING:
                        print("Results from the test:")
                        for key_i, value_i in test_results.items():
                            print(f'  {key_i:13s}: {value_i}')

    with SimpleTimer(f'Observe model parameter sensitivities (method-2)'):

        # Compute model parameter sensitivities by minimizing the distance
        # between the variation in the primary field with respect to the model
        # parameters and some arbitrary variation in the primary field. The
        # constness of the external force can be taken into account.
        #
        # Solve for `dm_` given an arbitrary `du_`:
        #   d(du)/dm (du/dm dm_ - du_) = 0,
        # subject to:
        #   dfdm dm_ = 0
        #
        # NOTE: The sensitivities are defined on the whole domain. They can be
        # used to estimate the change in model parameters for a given change
        # in the primary/displacement field. However, the method has a major
        # assumptions; specifically, it is assumed that the model can satisfy
        # all measurements exactly. Similar to the previous method, projecting
        # the discrete sensitivities on a function space is not trivial; a
        # simple L2-projection will result in quite a rough function.

        i_msr = 0 # Index of measured force
        i_dim = 0 # Index of force component

        dfdm_at_t = inverse_solver.observe_dfdm(t_obs)
        constraint_vector = dfdm_at_t[i_msr][i_dim,:]

        dmdu_at_t_as_array_values = inverse_solver \
            .observe_dmdu(t_obs, constraint_vector)

        dmdu_at_t_as_vector_field = invsolve.functions \
            .project_sensitivities_dmdu_msr(dmdu_at_t_as_array_values, V,
            smoothing_solver=(smoothing_solver if V == V_obs else None))

        dmdu_at_t_as_scalar_field = [dolfin.project(dolfin.sqrt(f**2), S)
                                     for f in dmdu_at_t_as_vector_field]

        for i, dmidu in enumerate(dmdu_at_t_as_vector_field):
            # NOTE: Appending "_msr" because makes better sense
            dmidu.rename(f'dm_{i}/du_msr','')

        for i, dmidu in enumerate(dmdu_at_t_as_scalar_field):
            # NOTE: Appending "_msr" because makes better sense
            dmidu.rename(f'dm_{i}/du_msr','')

        if TEST_SENSITIVITY_PROJECTION:

            test_success, test_results = \
                invsolve.functions.test_projected_sensitivities_dmdu_msr(
                dmdu_at_t_as_vector_field, dmdu_at_t_as_array_values,
                rtol=1e-5, atol=1e-8)

            if not test_success:
                logger.warning("Projection error of discrete sensitivities "
                               "is not within specified tolerance.")
                if logger.level <= logging.WARNING:
                    print("Results from the test:")
                    for key_i, value_i in test_results.items():
                        print(f'  {key_i:13s}: {value_i}')


    with SimpleTimer(f'Observe model parameter sensitivities (method-3)'):

        # Compute the model parameter sensitivities from the eigenvectors of
        # the model cost. The constness of the external force can be taken
        # into account.
        #
        # NOTE: The sensitivities can only be determined in relative terms. As
        # such, the sensitivities can not be used to estimate the change in the
        # model parameters for a given change in the displacement measurements.
        # In contrast to the previous methods, the sensitivities are directly
        # obtained as smooth functions. The sensitivity function are normalized
        # because the magnitude of sensitivity does not matter.

        d2J_principal_unconstrained, dm_principal_unconstrained = \
            inverse_solver.assess_cost_sensitivity(constraint_vectors=None)
        # NOTE: Sorted in descending order of absolute magnitude.

        i_msr = 0 # Index of measured force
        i_dim = 0 # Index of force component

        # Assume the constraint to be the `i_msr`th force
        dfdm_at_t = inverse_solver.observe_dfdm(t_obs)[i_msr]
        # NOTE: It may be more appropriate to use the mean force.

        # Consider only the `i_dim`th component
        constraint_vector = dfdm_at_t[i_dim, :]

        d2J_principal_constrained, dm_principal_constrained = \
            inverse_solver.assess_cost_sensitivity(constraint_vector)
        # NOTE: Sorted in descending order of absolute magnitude.

        # # Last principal value is zero because a constraint was used
        # d2J_principal_constrained = d2J_principal_constrained[:-1]
        # dm_principal_constrained = dm_principal_constrained[:-1]

        dudm_dm_principal_at_t_as_vector_field = \
            [inverse_solver.observe_dudm_dm(dm_i, t_obs)
             for dm_i in dm_principal_constrained]

        dudm_dm_principal_at_t_as_scalar_field = \
            [dolfin.project(dolfin.sqrt(f**2), S)
            for f in dudm_dm_principal_at_t_as_vector_field]

        for i, fn_i in enumerate(dudm_dm_principal_at_t_as_vector_field):
            fn_i.rename(f'du/dm[dm_principal_{i}]', '')

        for i, fn_i in enumerate(dudm_dm_principal_at_t_as_scalar_field):
            fn_i.rename(f'du/dm[dm_principal_{i}]', '')

        ### Normalize the sensitivity functions since the magnitudes do not matter

        assert V==V_obs, "Can not handle mixed function spaces. [TODO]"
        dofmaps = problems.utility.list_subspace_dofs(V_obs)

        for fn_i in dudm_dm_principal_at_t_as_vector_field:

            vec = fn_i.vector()
            arr = vec.get_local()

            norm_vec = np.sqrt(
                (np.stack([arr[dofmap_i] for dofmap_i in dofmaps], axis=1)**2)
                .sum(axis=1)).max() # l-infinity norm of the vector magnitudes

            vec /= norm_vec + EPS

        for fn_i in dudm_dm_principal_at_t_as_scalar_field:
            fn_i.vector()[:] /= (np.abs(fn_i.vector().get_local()).max() + EPS)


if COMPUTE_MISFIT_FIELD:
    with SimpleTimer(f'Observe displacement field misfit'):

        # Cell indices of the displacement measurement subdomain (DIC)
        cell_indices = np.flatnonzero(measurement_markers_dic.array())

        u_msr_at_t = invsolve.functions.project_expression(
            u_msr, V_obs, cell_indices, method='interpolate')

        u_obs_at_t = invsolve.functions.project_expression(
            u_obs, V_obs, cell_indices, method='interpolate')

        u_mis_at_t_as_vector_field = Function(V_obs, u_obs_at_t.vector()-u_msr_at_t.vector())
        u_mis_at_t_as_scalar_field = dolfin.project(dolfin.sqrt(u_mis_at_t_as_vector_field**2), S)

        u_mis_at_t_as_vector_field.rename('misfit displacement field','')
        u_mis_at_t_as_scalar_field.rename('misfit displacement field (magnitude)','')


### Stress field

def compute_stress_field(stress_measure_name='pk2'):
    '''
    Notes
    -----
    This method needs to perform `len(id_subdomains_material)` projections.

    '''

    if stress_measure_name == 'pk1':
        subdomain_stresses = pk1

    elif stress_measure_name == 'pk2':
        subdomain_stresses = pk2

    else:
        raise ValueError('Parameter `stress_measure_name` '
                         'must be valued: "pk1" or "pk2".')

    stress_field = invsolve.functions.project_subdomain_stresses(
        subdomain_stresses, W, domain_markers, id_subdomains_material)

    return stress_field

def compute_stress_field_fast(stress_measure_name='pk2'):
    '''
    Notes
    -----
    This method needs to performs 1 projection.

    '''

    material_parameters_as_expressions = problems.utility \
        .convert_material_parameters_in_subdomains_to_single_expressions(
            material_parameters, id_subdomains_material, domain_markers)

    material_model = MaterialModel(material_parameters_as_expressions)

    material_model.initialize(u)

    if stress_measure_name == 'pk1':
        stress_measure = material_model.stress_measure_pk1()

    elif stress_measure_name == 'pk2':
        stress_measure = material_model.stress_measure_pk2()

    else:
        raise ValueError('Parameter `stress_measure_name` '
                         'must be valued: "pk1" or "pk2".')

    return dolfin.project(stress_measure, W)

if COMPUTE_STRESS_FIELD:
    with SimpleTimer(f'Observe stress field at time {t_obs:.3g}'):
        stress_field_pk1_at_t = None # compute_stress_field_fast('pk1')
        stress_field_pk2_at_t = compute_stress_field('pk2')
else:
    stress_field_pk1_at_t = None
    stress_field_pk2_at_t = None

# if COMPUTE_STRESS_FIELD:
#     with SimpleTimer(f'Observe stress field at time {t_obs:.3g}'):
#         stress_field_pk1_at_t = None # compute_stress_field_fast('pk1')
#         stress_field_pk2_at_t = compute_stress_field_fast('pk2')
# else:
#     stress_field_pk1_at_t = None
#     stress_field_pk2_at_t = None


### Results

def plot_results():

    MEASUREMENT_INDEX = 0

    fig_handle_and_name_pairs = []

    fig_handle_and_name_pairs.append(
        problems.plotting.plot_problem_domain(
            mesh=None, domain_markers=domain_markers,
            figname="Material Subdomains with Superposed Displacement Measurement Points"))

    idx_figure_handle = 0
    handle_scatter_plot = fig_handle_and_name_pairs[-1][idx_figure_handle].get_axes()[0] \
        .scatter(measurement_x_dic[:,0], measurement_x_dic[:,1], s=1, c='r', marker='.')
    plt.legend([handle_scatter_plot], ["DIC data"])

    fig_handle_and_name_pairs.append(
        problems.plotting.plot_problem_domain(
            mesh=None, domain_markers=measurement_markers_dic,
            figname="Displacement Measurement Subdomain"))

    idx_figure_handle = 0
    handle_scatter_plot = fig_handle_and_name_pairs[-1][idx_figure_handle].get_axes()[0] \
        .scatter(measurement_x_dic[:,0], measurement_x_dic[:,1], s=1, c='r', marker='.')
    plt.legend([handle_scatter_plot], ["DIC data"])

    if OPTIMIZE_FOREACH_OBSERVATION_TIME:

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_model_parameters_foreach(
                model_parameters_foreach,
                model_parameter_names,
                model_observation_times,
                figname="Fitted Model Parameters for Each Observation Time"))

    if OPTIMIZE_FORALL_OBSERVATION_TIMES:

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_model_parameters_forall(
                model_parameters_forall,
                model_parameter_names,
                figname="Fitted Model Parameters for all Observation Times"))

    if COMPUTE_FINAL_COST:

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_model_cost(
                cost_values_final,
                cost_values_initial,
                model_observation_times,
                figname="Model Cost"))

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_cost_gradients(
                cost_gradients_final,
                model_parameter_names,
                model_observation_times,
                figname="Model Cost Derivatives"))

    if COMPUTE_MISFIT_ERROR:

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_observation_misfit(
                misfit_reaction_forces[MEASUREMENT_INDEX],
                model_observation_times,
                figname="Reaction Force Misfit Error",
                ylabel="Reaction force misfit error, $||f_{obs}-f_{msr}||/||f_{msr}||$"))

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_observation_misfit(
                misfit_displacements[MEASUREMENT_INDEX],
                model_observation_times,
                figname="Displacement Field Misfit Error",
                ylabel="Displacement field misfit error, $||u_{obs}-u_{msr}||/||u_{msr}||$"))

    if COMPUTE_REACTION_FORCE:

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_reaction_force_vs_displacement(
                np.array(reaction_forces_observed[MEASUREMENT_INDEX])[:,0],
                np.array(reaction_forces_measured[MEASUREMENT_INDEX])[:,0],
                np.array(reaction_displacements[MEASUREMENT_INDEX])[:,0],
                figname="Reaction Force-Displacement Curve"))

    if COMPUTE_SENSITIVITIES:

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_model_parameter_sensitivities(
                sensstd_dmdu_msr[MEASUREMENT_INDEX],
                model_parameter_names,
                model_observation_times,
                figname="Model Parameter Sensitivities wrt Displacement Measurements (Absolute)",
                ylabel="Model parameter sensitivity, $std(m_i)$",
                title="Standard Deviation in Model Parameters Assuming One\n"
                    "Standard Deviation in Displacement Measurements"))

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_model_parameter_sensitivities(
                sensstd_dmdf_msr[MEASUREMENT_INDEX],
                model_parameter_names,
                model_observation_times,
                figname="Model Parameter Sensitivitiesd wrt Force Measurements (Absolute)",
                ylabel="Model parameter sensitivity, $std(m_i)$",
                title=("Standard Deviation in Model Parameters Assuming One\n"
                    "Standard Deviation in Reaction Force Measurements")))

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_model_parameter_sensitivities(
                sensstd_dmdu_msr_relative[MEASUREMENT_INDEX],
                model_parameter_names,
                model_observation_times,
                figname="Model Parameter Sensitivities wrt Displacement Measurements (Relative)",
                ylabel="Relative Model parameter sensitivity, $std(m_i)/|m_i|$",
                title="Relative Deviation in Model Parameters Assuming One\n"
                    "Standard Deviation in Displacement Measurements"))

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_model_parameter_sensitivities(
                sensstd_dmdf_msr_relative[MEASUREMENT_INDEX],
                model_parameter_names,
                model_observation_times,
                figname="Model Parameter Sensitivities wrt Force Measurements (Relative)",
                ylabel="Relative Model parameter sensitivity, $std(m_i)/|m_i|$",
                title=("Relative Deviation in Model Parameters Assuming One\n"
                    "Standard Deviation in Reaction Force Measurements")))

        for i, fn_i in enumerate(dmdu_msr_at_t_as_scalar_field[MEASUREMENT_INDEX]):

            fig_handle_and_name_pairs.append(
                problems.plotting.plot_scalar_field(fn_i,
                    figname=f'Sensitivity of Model Parameter ({i})',
                    title=('Sensitivity of Model Parameter '
                        f'"{model_parameter_names[i]}"\n'
                        'with Respect to Displacement Field Measurements')))

        for i, fn_i in enumerate(dmdu_at_t_as_scalar_field):

            fig_handle_and_name_pairs.append(
                problems.plotting.plot_scalar_field(fn_i,
                    figname=f'Estimated Global Sensitivity ({i})',
                    title=('Estimated Global Sensitivity\n'
                        f'of Model Parameter "{model_parameter_names[i]}"')))

        for i, fn_i in enumerate(dudm_dm_principal_at_t_as_scalar_field):
            i_prc = i + 1 # Principal values are usually enumerated from `1`

            fig_handle_and_name_pairs.append(
                problems.plotting.plot_scalar_field(fn_i,
                    figname=f'Principal Model Cost Sensitivity ({i})',
                    title=(f'Principal Model Cost Sensitivity ({i_prc})\n'
                        'with Respect to Displacement Measurements')))

    if COMPUTE_MISFIT_FIELD:

        fig_handle_and_name_pairs.append(
            problems.plotting.plot_scalar_field(
                u_mis_at_t_as_scalar_field,
                figname='Displacement Field Misfit',
                title='Misfit Between Model and Measured Displacements'))

    if COMPUTE_STRESS_FIELD:

        if stress_field_pk1_at_t is not None:
            fig_handle_and_name_pairs.append(
                problems.plotting.plot_scalar_field(
                    dolfin.project(stress_field_pk1_at_t**2, S),
                    figname='Stress Field Magnitude (PK1)',
                    title='Stress Field Magnitude (PK1)'))

        if stress_field_pk2_at_t is not None:
            fig_handle_and_name_pairs.append(
                problems.plotting.plot_scalar_field(
                    dolfin.project(stress_field_pk2_at_t**2, S),
                    figname='Stress Field Magnitude (PK2)',
                    title='Stress Field Magnitude (PK2)'))

    return fig_handle_and_name_pairs


def save_results(fig_handle_and_name_pairs=None):

    if FIXED_EXTERNAL_BOUNDARY:
        results_subdir_name = "boundary(fixed)"
    else:
        results_subdir_name = "boundary(free)"

    maximum_displacement = math.sqrt((measurement_u_bnd[-1]**2).sum())
    results_subdir_name += (f"-maxdisp({int(maximum_displacement+0.5)})"
                            f"-meshtag({mesh.num_vertices():07d})")

    results_outdir = os.path.join(
        RESULTS_DIRECTORY_PARENT, results_subdir_name)

    if not os.path.isdir(results_outdir):
        os.makedirs(results_outdir)

    if fig_handle_and_name_pairs is not None:
        close_figures_finally = False
    else:
        fig_handle_and_name_pairs = plot_results()
        close_figures_finally = True

    fig_handles = [f[0] for f in fig_handle_and_name_pairs]
    fig_names = [f[1] for f in fig_handle_and_name_pairs]

    outdir_arrays = os.path.join(results_outdir, "arrays")
    outdir_figures = os.path.join(results_outdir, "figures")
    outdir_functions = os.path.join(results_outdir, "functions")

    if not os.path.isdir(outdir_arrays): os.makedirs(outdir_arrays)
    if not os.path.isdir(outdir_figures): os.makedirs(outdir_figures)
    if not os.path.isdir(outdir_functions): os.makedirs(outdir_functions)

    problems.utility.remove_outfiles(outdir_arrays, SAFE_TO_REMOVE_FILE_TYPES)
    problems.utility.remove_outfiles(outdir_figures, SAFE_TO_REMOVE_FILE_TYPES)
    problems.utility.remove_outfiles(outdir_functions, SAFE_TO_REMOVE_FILE_TYPES)

    for handle_i, name_i in zip(fig_handles, fig_names):
        subdir_i = os.path.join(outdir_figures, name_i)

        for ext_j in SAVE_FIGURE_EXTENSIONS:
            handle_i.savefig(subdir_i + ext_j)

    file_name = "model_observation_times.out"
    np.savetxt(os.path.join(outdir_arrays, file_name),
               model_observation_times)

    file_name = "cost_gradient.out"
    np.savetxt(os.path.join(outdir_arrays, file_name),
               inverse_solver.view_cumsum_DJDm(),
               header='Model cost gradient `DJDm`.')

    file_name = "cost_hessian.out"
    np.savetxt(os.path.join(outdir_arrays, file_name),
               inverse_solver.view_cumsum_D2JDm2(),
               header='Model cost Hessian `D2JDm2`.')

    if cost_values_initial is not None:

        file_name = "cost_values_initial"
        np.savetxt(os.path.join(outdir_arrays, file_name),
                   np.array(cost_values_initial)[:,None],
                   header='Model cost at observation times.')

    if cost_values_final is not None:

        file_name = "cost_values_final"
        np.savetxt(os.path.join(outdir_arrays, file_name),
                   np.array(cost_values_final)[:,None],
                   header='Model cost at observation times.')

    if model_parameters_foreach is not None:

        file_name = "model_parameters_for_each_time.out"
        np.savetxt(os.path.join(outdir_arrays, file_name),
                   np.array(model_parameters_foreach, ndmin=2),
                   header=' '.join(model_parameter_names))

    if model_parameters_forall is not None:

        file_name = "model_parameters_for_all_times.out"
        np.savetxt(os.path.join(outdir_arrays, file_name),
                   np.array(model_parameters_forall, ndmin=2),
                   header=' '.join(model_parameter_names))

    if COMPUTE_MISFIT_FIELD:

        dolfin.File(os.path.join(outdir_functions,
            f"u_msr_at_t({t_obs:02d}).pvd")) << u_msr_at_t

        dolfin.File(os.path.join(outdir_functions,
            f"u_obs_at_t({t_obs:02d}).pvd")) << u_obs_at_t

        dolfin.File(os.path.join(outdir_functions,
            f"u_mis_at_t({t_obs:02d}).pvd")) << u_mis_at_t_as_vector_field

    if COMPUTE_STRESS_FIELD:

        if stress_field_pk1_at_t is not None:

            file_name = f"stress_field_pk1_at_t({t_obs:02d}).pvd"
            file_path = os.path.join(outdir_functions, file_name)
            dolfin.File(file_path) << stress_field_pk1_at_t

        if stress_field_pk2_at_t is not None:

            file_name = f"stress_field_pk2_at_t({t_obs:02d}).pvd"
            file_path = os.path.join(outdir_functions, file_name)
            dolfin.File(file_path) << stress_field_pk2_at_t

    if COMPUTE_SENSITIVITIES:

        file_name = "cost_sensitivity_constrained.out"
        file_path = os.path.join(outdir_arrays, file_name)
        np.savetxt(file_path, np.array(d2J_principal_constrained, ndmin=2),
                   header= 'Eigenvalues of model cost Hessian `D2JDm2` '
                           '(constrained wrt. reaction force).')

        file_name = "cost_sensitivity_unconstrained.out"
        file_path = os.path.join(outdir_arrays, file_name)
        np.savetxt(file_path, np.array(d2J_principal_unconstrained, ndmin=2),
                   header='Eigenvalues of model cost Hessian `D2JDm2`.')

        if inverse_solver.num_u_msr == 1:

            i_msr = 0
            i_obs = model_observation_times.index(t_obs)

            file_name = f"dmdu_msr_at_t({t_obs:02d}).npy"
            file_path = os.path.join(outdir_arrays, file_name)
            np.save(file_path, dmdu_msr[i_msr][i_obs])

            file_name = f"dmdf_msr_at_t({t_obs:02d}).npy"
            file_path = os.path.join(outdir_arrays, file_name)
            np.save(file_path, dmdf_msr[i_msr][i_obs])

            file_name = f"correlation_dmdu_msr_at_t.out"
            file_path = os.path.join(outdir_arrays, file_name)
            file_header = ("Correlation between model parameter sensitivities "
                           "with respect to displacement field measurements.\n"
                           "Row -> i\'th model parameter sensitivity vector\n"
                           "Column -> j\'th model parameter sensitivity vector")
            np.savetxt(file_path, correlation_dmdu_msr_at_t[i_msr], header=file_header)

            file_name = f"senssup_dmdu_msr_for_each_time.out"
            file_path = os.path.join(outdir_arrays, file_name)
            file_header = ("Sensitivity measure: SUPREMUM\n"
                           "Row -> observation time\n"
                           "Column -> model parameter")
            np.savetxt(file_path, np.array(senssup_dmdu_msr[i_msr]), header=file_header)

            file_name = f"sensstd_dmdu_msr_for_each_time.out"
            file_path = os.path.join(outdir_arrays, file_name)
            file_header = ("Sensitivity measure: STANDARD DEVIATION\n"
                           "Row -> observation time\n"
                           "Column -> model parameter")
            np.savetxt(file_path, np.array(sensstd_dmdu_msr[i_msr]), header=file_header)

            file_name = f"senssup_dmdf_msr_for_each_time.out"
            file_path = os.path.join(outdir_arrays, file_name)
            file_header = ("Sensitivity measure: SUPREMUM\n"
                           "Row -> observation time\n"
                           "Column -> model parameter")
            np.savetxt(file_path, np.array(senssup_dmdf_msr[i_msr]), header=file_header)

            file_name = f"sensstd_dmdf_msr_for_each_time.out"
            file_path = os.path.join(outdir_arrays, file_name)
            file_header = ("Sensitivity measure: STANDARD DEVIATION\n"
                           "Row -> observation time\n"
                           "Column -> model parameter")
            np.savetxt(file_path, np.array(sensstd_dmdf_msr[i_msr]), header=file_header)

            for j_m, dmjdu_msr_i in enumerate(dmdu_msr_at_t_as_vector_field[i_msr]):
                file_name = f"dm({j_m})du_msr_at_t({t_obs:02d}).pvd"
                file_path = os.path.join(outdir_functions, file_name)
                dolfin.File(file_path) << dmjdu_msr_i

        else:
            raise NotImplementedError

        for i_m, dmidu in enumerate(dmdu_at_t_as_vector_field):
            file_name = f"dm({i_m})du_msr_at_t({t_obs:02d})_estimated.pvd"
            file_path = os.path.join(outdir_functions, file_name)
            dolfin.File(file_path) << dmidu

        for i, dudm_i in enumerate(dudm_dm_principal_at_t_as_vector_field):
            file_name = f"dudm_dm_principal({i})_at_t({t_obs:02d}).pvd"
            file_path = os.path.join(outdir_functions, file_name)
            dolfin.File(file_path) << dudm_i

    if close_figures_finally:
        for fig_handle_i in fig_handles:
            plt.close(fig_handle_i)


if __name__ == '__main__':

    plt.interactive(True)

    if PLOT_RESULTS:
        fig_handle_and_name_pairs = plot_results()
    else:
        fig_handle_and_name_pairs = None

    if SAVE_RESULTS:
        save_results(fig_handle_and_name_pairs)

    print(f'norm(u): {dolfin.norm(u):.4g}')
