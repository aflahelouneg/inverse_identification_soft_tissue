
import os
import sys
import time
import dolfin
import logging
import importlib
import numpy as np


SEQUENCE_TYPES = (tuple, list)
logger = logging.getLogger()


class SimpleTimer:

    def __init__(self, label=""):
        self.t0 = self.dt = 0.0
        self.label = f'{label!s}'

    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, *args):
        self.dt = time.time() - self.t0
        print(f'\n{self.label}\n  Time taken: {self.dt:.3f}\n')


class RectangleSubdomain(dolfin.SubDomain):
    def __init__(self, p0, p1, atol):
        super().__init__()

        self.x0 = p0[0] - atol
        self.y0 = p0[1] - atol
        self.x1 = p1[0] + atol
        self.y1 = p1[1] + atol

        if self.x0 > self.x1:
            raise ValueError

        if self.y0 > self.y1:
            raise ValueError

    def inside(self, x, on_boundary):
        return x[0] > self.x0 and x[0] < self.x1 and \
               x[1] > self.y0 and x[1] < self.y1


class UnitWeightOnRectangleSubdomain(dolfin.UserExpression):
    '''Scalar flattop weight.'''

    def __init__(self, p0, p1, mesh, atol):
        super().__init__(degree=0)

        self.rect_subd = RectangleSubdomain(p0, p1, atol)
        self.mf = dolfin.MeshFunction('size_t', mesh, mesh.topology().dim(), 0)
        self.rect_subd.mark(self.mf, 1)

    def eval_cell(self, value, x, ufc_cell):
        value[0] = self.mf[ufc_cell.index]

    def value_shape(self):
        return tuple() # Empty tuple means scalar field


class ExpressionFromSubdomainValues(dolfin.UserExpression):
    '''

    '''

    def __new__(cls, *args, **kwargs):

        if 'degree' not in kwargs and 'element' not in kwargs:
            raise TypeError('Require `degree` or `element` as keyword argument.')

        self = super().__new__(cls)
        self._ufl_shape = ()

        return self

    def __repr__(self):
        return f'<{self.__class__.__name__} at {hex(id(self))}>'

    def __init__(self, subdomain_values, subdomain_markers, **kwargs):
        '''

        '''

        # Must initialize base class
        super().__init__(**kwargs)

        if not isinstance(subdomain_values, dict):
            raise TypeError('Parameter `subdomain_values` must be a `dict`.')

        if not type(subdomain_markers).__name__.startswith('MeshFunction'):
            raise TypeError('Parameter `subdomain_markers` must be '
                            'a `dolfin.MeshFunction` that marks the '
                            'subdomain of each cell in the mesh.')

        unique_subdomains = np.unique(subdomain_markers.array()).tolist()
        if sorted(set(subdomain_values.keys())) != unique_subdomains:
            raise TypeError('Parameter `subdomain_values` must have keys '
                            'that are the same as the subdomain id\'s '
                            'defined by parameter `subdomain_markers`.')

        self.subdomain_values = subdomain_values
        self.subdomain_markers = subdomain_markers

    def eval_cell(self, values, x, cell):

        values[0] = self.subdomain_values[
                    self.subdomain_markers[
                    cell.index]]

    def value_shape(self):
        return self._ufl_shape


def transform_material_parameters_to_key_subdomain_value_representation(
        material_parameters, material_subdomain_ids, subdomain_markers):
    '''
    Parameters
    ----------
    material_parameters : sequence of dict's
    material_subdomain_ids : sequence of (sequences of) int's
    subdomain_markers : dolfin.MeshFunctionSizet

    Returns
    -------
    material_parameter_subdomain_values : dict

    '''

    if not (isinstance(material_parameters, SEQUENCE_TYPES) and
            all(isinstance(mps_i, dict) for mps_i in material_parameters)):
        raise TypeError('Parameter `material_parameters` must be a sequence of `dict`s.')

    if len(material_parameters) != len(material_subdomain_ids):
        raise TypeError('Parameters `material_parameters` and `material_subdomain_ids` '
                        'must have the same length.')

    if type(subdomain_markers).__name__ != 'MeshFunctionSizet':
        raise TypeError('Parameter `subdomain_markers` must be '
                        'a `dolfin.MeshFunctionSizet` that marks '
                        'the subdomain of each cell in the mesh.')

    material_parameter_keys = []

    for mps_subd_i in material_parameters:
        material_parameter_keys.extend(mps_subd_i.keys())

    material_parameter_keys = set(material_parameter_keys)

    unique_subdomains = []

    for subdomain_ids_i in material_subdomain_ids:
        if isinstance(subdomain_ids_i, SEQUENCE_TYPES):
            unique_subdomains.extend(subdomain_ids_i)
        else:
            unique_subdomains.append(subdomain_ids_i)

    unique_subdomains = sorted(set(unique_subdomains))

    if unique_subdomains != np.unique(subdomain_markers.array()).tolist():
        raise RuntimeError('Parameter `material_subdomain_ids` must contain the same '
                           'subdomain id\'s that are defined by `subdomain_markers`.')

    material_parameter_subdomain_values = \
        {key_i : {subdomain_j : 0.0
        for subdomain_j in unique_subdomains}
        for key_i in material_parameter_keys}

    for key_i, subdomain_values_i in \
            material_parameter_subdomain_values.items():

        for subdomain_j, material_parameters_j in \
                zip(material_subdomain_ids, material_parameters):

            value_ij = material_parameters_j[key_i]

            if isinstance(subdomain_j, SEQUENCE_TYPES):
                for subdomain_jk in subdomain_j:
                    subdomain_values_i[subdomain_jk] = value_ij
            else:
                subdomain_values_i[subdomain_j] = value_ij

    return material_parameter_subdomain_values


def convert_material_parameters_in_subdomains_to_single_expressions(
        material_parameters, material_subdomain_ids, subdomain_markers):

    material_parameter_subdomain_values = \
        transform_material_parameters_to_key_subdomain_value_representation(
            material_parameters, material_subdomain_ids, subdomain_markers)

    material_parameter_expressions = {key_i :
        ExpressionFromSubdomainValues(subdomain_values_i, subdomain_markers, degree=0)
        for key_i, subdomain_values_i in material_parameter_subdomain_values.items()}

    return material_parameter_expressions


def reload_module(name):
    '''(Re)import module.'''

    if not isinstance(name, str):
        if hasattr(name, '__name__'): name = name.__name__
        else: raise TypeError('Expected `name` to be a `str`.')

    module = sys.modules.get(name, None)
    if module: return importlib.reload(module)
    else: return importlib.import_module(name)


def remove_outfiles(subdir, file_extensions):

    if not isinstance(file_extensions, (list, tuple)):
        file_extensions = (file_extensions,)

    if not all(isinstance(ext, str) for ext in file_extensions):
        raise ValueError('Parameter `file_extensions` must be '
                         'a (`list` or `tuple` of) `str`(s).')

    file_extensions = tuple(ext if ext.startswith('.')
        else ('.' + ext) for ext in file_extensions)

    for item in os.listdir(subdir):
        item = os.path.join(subdir, item)

        if os.path.isfile(item):
            _, ext = os.path.splitext(item)

            if ext in file_extensions:
                os.remove(item)


def mark_rectangular_subdomain(p0, p1, mesh):
    '''Get markers for a rectangular subdomain.

    Parameters
    ----------
    p0 : point-like
        Bottom-left vertex coordinates.
    p1 : point-like
        Top-right vertex coordinates.
    mesh : dolfin.Mesh
        Domain mesh.

    Returns
    -------
    domain_markers : dolfin.MeshFunction
        Domain markers where displacement measurements are known. The subdomain
        is identified with mesh function values equal to 1.

    '''

    SUBDOMAIN_MARKER_ID = 1

    subdomain = dolfin.CompiledSubDomain(
        "x[0] > x0 && x[0] < x1 && x[1] > y0 && x[1] < y1",
        x0=p0[0], y0=p0[1], x1=p1[0], y1=p1[1])

    subdomain_markers = dolfin.MeshFunction('size_t',
        mesh, dim=mesh.topology().dim(), value=0)

    subdomain.mark(subdomain_markers, SUBDOMAIN_MARKER_ID)

    return subdomain_markers, SUBDOMAIN_MARKER_ID


def meshgrid_inside_mesh2d(mesh, step, xlim=(-np.inf,np.inf), ylim=(-np.inf, np.inf)):
    '''Generate a uniform grid of sample points inside a 2D mesh.'''

    if not isinstance(mesh, dolfin.Mesh) or mesh.geometry().dim() != 2:
        raise TypeError('Parameter `mesh`.')

    if not isinstance(step, (float, int)) or step <= 0.0:
        raise TypeError('Parameter `step`.')

    if not hasattr(xlim, '__getitem__') or len(xlim) != 2 or xlim[1]-xlim[0] < step:
        raise TypeError('Parameter `xlim`.')

    if not hasattr(ylim, '__getitem__') or len(ylim) != 2 or ylim[1]-ylim[0] < step:
        raise TypeError('Parameter `ylim`.')

    xmin, ymin = mesh.coordinates().min(axis=0)
    xmax, ymax = mesh.coordinates().max(axis=0)

    x0 = xmin if xmin > xlim[0] else xlim[0]
    y0 = ymin if ymin > ylim[0] else ylim[0]

    x1 = xmax if xmax < xlim[1] else xlim[1]
    y1 = ymax if ymax < ylim[1] else ylim[1]

    x = np.linspace(x0, x1, (x1-x0) // step + 1)
    y = np.linspace(y0, y1, (y1-y0) // step + 1)

    x, y = np.meshgrid(x, y)

    x = x.reshape((-1,))
    y = y.reshape((-1,))

    grid = np.stack([x,y], axis=1)

    compute_first_collision = mesh.bounding_box_tree().compute_first_collision
    def detect_collision(p): return compute_first_collision(p) != max_uint

    max_uint = compute_first_collision(dolfin.Point(np.inf))
    mask = np.empty(shape=(len(grid),), dtype=bool)

    for i, x in enumerate(grid):
        mask[i] = detect_collision(dolfin.Point(x))

    return np.ascontiguousarray(grid[mask])


def linspace_range(first, last, count, start_from="front"):
    '''Uniformly space the range starting either from the front or the back.'''

    if not isinstance(start_from, str) \
        or start_from not in ("front", "back"):
        raise ValueError('Parameter `start_from`.')

    if count == 0:
        return []

    if count == 1:
        if start_from == "front":
            return [first,]
        else: # start_from == "back"
            return [last,]

    if count < 2:
        raise ValueError('Require `count >= 0`.')

    x = range(first, last + 1)

    if count > len(x):
        raise ValueError('Require `count <= last - first + 1`.')

    dx = int((len(x)-1)/(count-1))

    if start_from == "front":

        x = x[::dx]
        if len(x) > count:
            x = x[:count]

    else: # start_from == "back"

        x = x[::-1][::dx]
        if len(x) > count:
            x = x[:count]
        x = x[::-1]

    return x


def list_model_parameter_names(dict_sequence, value_types):

    names = []

    if not isinstance(dict_sequence, SEQUENCE_TYPES):
        dict_sequence = (dict_sequence,)

    if len(dict_sequence) == 1:
        def make_name(key, _):
            return key
    else:
        def make_name(key, postfix):
            return key + f'_{postfix}'

    if not all(isinstance(dict_i, dict) for dict_i in dict_sequence):
        raise TypeError('`dict_sequence` must be a (sequence of) `dict`(s).')

    for i, dict_i in enumerate(dict_sequence):
        for key_j, value_j in dict_i.items():
            if isinstance(value_j, value_types):
                names.append(make_name(key_j, i))

    return names


def list_subspace_dofs(V):
    '''Get the degrees of freedom of each subspace of a vector function space.'''

    if not isinstance(V, dolfin.FunctionSpace):
        raise TypeError('Parameter `V` must be a `dolfin.FunctionSpace`.')

    return tuple(V_i.dofmap().dofs() for V_i in V.split()) \
        if V.num_sub_spaces() > 1 else (V.dofmap().dofs(),)


def apply_mean_filter(w, a):
    '''Weighted-mean filtering of an array.

    Parameters
    ----------
    w : sequence of float's
        The filter weights. Note, the filter needs to be symmetric about
        the center node; hence, the length of `w` must be an odd number.
    a : sequence of float's or numpy.ndarray's
        The values to be filtered.

    '''

    if not (isinstance(a, list) and
            all(isinstance(a_i, (float, int, np.ndarray)) for a_i in a)):
        raise TypeError('Parameter `a` must be a `list` that contains '
                        '`float`s, `int`s or `numpy.ndarray`s.')

    if not (hasattr(w, '__len__') and hasattr(w, '__getitem__')
            and all(isinstance(w_i, (float, int)) for w_i in w)):
        raise TypeError('Parameter `w` must be a sequence '
                        'of either `float`s or `int`s.')

    if len(w) % 2 != 1:
        raise ValueError('Number of weights (`len(w)`) should be an odd number so '
                         'that each point coincides with the center of the filter.')

    sum_w = sum(w, 0.0)
    w = [w_i/sum_w for w_i in w]
    i0 = di = (len(w)-1) // 2

    a[i0:len(a)-i0] = [sum(a_i * w_i
        for a_i, w_i in zip(a[i-di:i+di+1], w))
        for i in range(i0, len(a)-i0)]
