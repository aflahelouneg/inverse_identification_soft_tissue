'''
For projecting discrete point-like measurements onto continuous functions.

'''

import dolfin
import logging
import numpy as np

from numpy import ndarray
from numpy.linalg import det
from scipy.linalg import solve
from scipy.spatial import cKDTree

SEQUENCE_TYPES = (tuple, list)
MESHLESS_NEIGHBORS_FROM_DEGREE = {0:3, 1:4, 2:9, 3:16}
TOLERANCE_VANDERMONDE_FROM_DEGREE = {0:0, 1:1e-2, 2:1e-6, 3:1e-8}


def project_pointvalues_on_functions(xk, fk, V, meshless_degree=1,
        meshless_weight="center", distance_norm=2, copy=True):
    '''Project measurements on functions using meshless interpolation.

    Given an array of points and a list of arrays of data values at those
    points, project these discrete values onto the degrees of freedom of a
    (vector-valued) function constructed from the function space by using
    the moving least squares meshless interpolation method.

    Parameters
    ----------
    xk : numpy.ndarray (2D)
        Reference points in 2D space.
    fk : numpy.ndarray or a sequence of numpy.ndarray's
        Reference values.
    V : dolfin.FunctionSpace
        Function space. Reference values `fk` will be interpolated at the
        degrees of freedom of a function in `V` using the meshless method.
    meshless_degree : integer (optional)
        Degree of the meshless interpolation.
    meshless_weight : string (optional)
        The type of the weighting method: "center" or "uniform".
    distance_norm : integer (optional, default=2)
        The distance norm to use in finding nearest neighbors. The common
        choices are: 2 (Euclidean distance), and 1 (Manhattan distance).

    Returns
    -------
    fn : (list of) dolfin.Function('s)
        Interpolation function(s).

    '''

    GDIM = 2

    if not isinstance(xk, ndarray) or xk.ndim != 2 or xk.shape[1] != GDIM:
        raise TypeError('Parameter `xk` must be a 2D `numpy.ndarray` whose '
                        f'second dimension has length equal to {GDIM}.')

    if not isinstance(V, dolfin.FunctionSpace) or V.mesh().geometry().dim() != GDIM:
        raise TypeError('Parameter `V` must be a `dolfin.FunctionSpace` on a 2D mesh.')

    if isinstance(fk, SEQUENCE_TYPES):
        is_return_type_sequence = True

        if len(fk) == 0:
            raise ValueError('Parameter `fk` is empty.')

        if not all(isinstance(fk_i, ndarray) for fk_i in fk):
            raise TypeError('Parameter `fk`.')

        if any(fk_i.shape != fk[0].shape for fk_i in fk[1:]):
            raise TypeError('Parameter `fk`.')

    elif isinstance(fk, ndarray):
        is_return_type_sequence = False
        fk = (fk,)
    else:
        raise TypeError('Parameter `fk`.')

    if not (1 <= fk[0].ndim <= 2):
        raise TypeError('Parameter `fk`.')

    vdim_data = fk[0].shape[1] if fk[0].ndim==2 else 1
    vdim_proj = max(V.ufl_element().num_sub_elements(), 1)

    if vdim_data != vdim_proj:
        raise ValueError('Different value dimensions of parameters '
                         f'`fk` ({vdim_data}) and `V` ({vdim_proj}).')

    if vdim_proj > 1: # `V` is vector-valued (can be split)
        dofmap = [V_i.dofmap().dofs() for V_i in V.split()]
    else: # `V` is a scalar-valued (can not be split)
        dofmap = [V.dofmap().dofs()]

    dofcrd = V.tabulate_dof_coordinates().reshape((-1,GDIM))
    dofcrd = [dofcrd[dofmap_i,:] for dofmap_i in dofmap]

    if __debug__:
        if not all(np.allclose(dofcrd[0], dofcrd_i) for dofcrd_i in dofcrd[1:]):
            raise RuntimeError('Expected same DOF coordinates of sub-dimensions.')

    xi = dofcrd[0]

    fi = project_pointvalues_on_points(xk, fk, xi,
        meshless_degree, meshless_weight, distance_norm, copy)

    fn = []

    for fi_t in fi:

        fn_t = dolfin.Function(V)
        lhs = fn_t.vector().get_local()

        for dim, dofs in enumerate(dofmap):
            lhs[dofs] = fi_t[:, dim]

        fn_t.vector()[:] = lhs
        fn.append(fn_t)

    return fn if is_return_type_sequence else fn[0]


def project_pointvalues_on_points(xk, fk, xi, meshless_degree=1,
        meshless_weight="center", distance_norm=2, copy=True):
    '''Interpolate point-values using meshless interpolation.

    Parameters
    ----------
    xk : numpy.ndarray (2D)
        Reference points in 2D space.
    fk : numpy.ndarray or a sequence of numpy.ndarray's
        Reference values.
    xi : numpy.ndarray (2D)
        Interpolation points in 2D space.
    meshless_degree : integer (optional)
        Degree of the meshless interpolation.
    meshless_weight : string (optional)
        The type of the weighting method: "center" or "uniform".
    distance_norm : integer (optional, default=2)
        The distance norm to use in finding nearest neighbors. The common
        choices are: 2 (Euclidean distance), and 1 (Manhattan distance).

    Returns
    -------
    fi : (list of) numpy.ndarray('s)
        Interpolated values at points `xi`.

    '''

    GDIM = 2

    if not isinstance(xk, ndarray) or xk.ndim != 2 or xk.shape[1] != GDIM:
        raise TypeError('Parameter `xk` must be a 2D `numpy.ndarray` '
                        'whose second dimension has length equal to 2.')

    if not isinstance(xi, ndarray) or xi.ndim != 2 or xi.shape[1] != GDIM:
        raise TypeError('Parameter `xi` must be a 2D `numpy.ndarray` '
                        'whose second dimension has length equal to 2.')

    if isinstance(fk, SEQUENCE_TYPES):
        is_return_type_sequence = True

        if len(fk) == 0:
            raise ValueError('Parameter `fk` is empty.')

        if not all(isinstance(fk_i, ndarray) for fk_i in fk):
            raise TypeError('Parameter `fk`.')

        if any(fk_i.shape != fk[0].shape for fk_i in fk[1:]):
            raise TypeError('Parameter `fk`.')

    elif isinstance(fk, ndarray):
        is_return_type_sequence = False
        fk = (fk,)
    else:
        raise TypeError('Parameter `fk`.')

    if not (1 <= fk[0].ndim <= 2):
        raise TypeError('Parameter `fk`.')

    if any(len(fk_i) != len(xk) for fk_i in fk):
        raise TypeError('Parameters `fk` and `xk`.')

    len_fk = len(fk)
    dim_fk = fk[0].ndim

    if dim_fk == 1: fk = np.stack(fk, axis=1)
    else: fk = np.concatenate(fk, axis=1)

    meshless = SimpleMeshlessInterpolation2d(xk, copy)
    num_neighbors = MESHLESS_NEIGHBORS_FROM_DEGREE[meshless_degree]
    meshless.set_interpolation_points(xi, num_neighbors, distance_norm, copy)
    fi = meshless.interpolate(fk, meshless_degree, meshless_weight, copy=False)

    # Un-stack/un-concatenate arrays
    fi = np.split(fi, len_fk, axis=1)

    if dim_fk == 1: fi = [f.squeeze() for f in fi]
    return fi if is_return_type_sequence else fi[0]


class SimpleMeshlessInterpolation2d:
    '''Does not depend on third party libraries. Use as fallback. However,
    the computational speed is several times slower than "wlsqm". Note, the
    solutions will be a bit different due to different weight functions used.'''

    @staticmethod
    def _eval_weight_uniform(r):
        '''Uniform weight function.'''
        return np.ones_like(r)

    @staticmethod
    def _eval_weight_center(r):
        '''The weight function "WEIGHT_CENTER" used in `wlsqm` module.'''
        return 1e-4 + (1.0-1e-4) * (1.0-r/r.max())**2

    @staticmethod
    def _eval_basis_p0(ones, x, y):
        '''Linear polynomial basis in two spatial dimensions.'''
        return np.stack([ones], axis=1)

    @staticmethod
    def _eval_basis_p1(ones, x, y):
        '''Linear polynomial basis in two spatial dimensions.'''
        return np.stack([ones, x, y], axis=1) # C-contiguous

    @staticmethod
    def _eval_basis_p2(ones, x, y):
        '''Quadratic polynomial basis in two spatial dimensions.'''
        return np.stack([ones, x, y, x*x, x*y, y*y], axis=1) # C-contiguous

    @staticmethod
    def _eval_basis_p3(ones, x, y):
        '''Cubic polynomial basis in two spatial dimensions.'''
        return np.stack([ones, x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y], axis=1)

    def __init__(self, xk, copy=True):
        '''
        Parameters
        ----------
        xk : numpy.ndarray (2D)
            Reference points.

        '''

        if not isinstance(xk, ndarray) or xk.ndim != 2 or xk.shape[1] != 2:
            raise TypeError('Parameter `xk` must be a 2D `numpy.ndarray` '
                            'whose second dimension has length equal to 2.')

        if copy: self._xk = np.copy(xk, float)
        else: self._xk = np.asarray(xk, float)
        self._kdtree_xk = cKDTree(self._xk)

        self._gdim = 2
        self._vdim = None

        self._xi = None
        self._fk = None

        self._neighbors_idx = None
        self._neighbors_dst = None

    def set_reference_values(self, fk, copy=True):
        '''
        Parameters
        ----------
        fk : numpy.ndarray (2D)
            Reference values.

        '''

        if not isinstance(fk, ndarray) or \
           fk.ndim != 2 or len(fk) != len(self._xk):
            raise TypeError('Parameter `fk` must be a 2D `numpy.ndarray` '
                            'whose length is equal to the length of `xk`.')

        if copy: self._fk = np.copy(fk, float)
        else: self._fk = np.asarray(fk, float)

        self._vdim = fk.shape[1]

    def set_interpolation_points(self, xi, num_neighbors, distance_norm=2, copy=True):
        '''
        Parameters
        ----------
        xi : numpy.ndarray (2D)
            Interpolation points.
        num_neighbors : integer
            Number of nearest neighbors.
        distance_norm : integer (optional)
            Order of the dinstance norm.

        '''

        if not isinstance(xi, ndarray) or xi.ndim != 2 or xi.shape[1] != self._gdim:
            raise TypeError('Parameter `xi` must be a 2D `numpy.ndarray` whose '
                            f'second dimension has length equal to {self._gdim}.')

        if copy: self._xi = np.copy(xi, float)
        else: self._xi = np.asarray(xi, float)

        self._neighbors_dst, self._neighbors_idx = self._kdtree_xk \
            .query(self._xi, k=num_neighbors, p=distance_norm)

        if self._neighbors_dst.ndim == 1: # Convert to 2D
            self._neighbors_dst = self._neighbors_dst[:,None]

        if self._neighbors_idx.ndim == 1: # Convert to 2D
            self._neighbors_idx = self._neighbors_idx[:,None]

        assert self._neighbors_dst.ndim == 2
        assert self._neighbors_idx.ndim == 2

    def interpolate(self, fk=None, degree=1, weight="center", copy=True):
        '''Interpolate values `fk` at interpolation points `xi`.

        Parameters
        ----------
        fk : numpy.ndarray (2D)
            Values at `xk`.
        degree: integer (optional)
            Meshless degree.
        weight: str (optional)
            Weighting method: "center" or "uniform".

        Returns
        ------
        fi : numpy.ndarray (2D)
            Values at `xi`.

        '''

        tol = TOLERANCE_VANDERMONDE_FROM_DEGREE[degree]

        if fk is not None:
            self.set_reference_values(fk, copy)

        elif self._fk is None:
            raise ValueError('`fk` is not set yet.')

        if self._xi is None:
            raise ValueError('`xi` is not set yet.')

        if   degree == 0: eval_basis = self._eval_basis_p0
        elif degree == 1: eval_basis = self._eval_basis_p1
        elif degree == 2: eval_basis = self._eval_basis_p2
        elif degree == 3: eval_basis = self._eval_basis_p3
        else: raise ValueError('Require 0 <= degree <= 3')

        if weight == "uniform": eval_weight = self._eval_weight_uniform
        elif weight == "center": eval_weight = self._eval_weight_center
        else: raise ValueError('Expected parameter `weight` to be '
                               'either `"uniform"` or `"center"`.')

        fk = self._fk
        xk = self._xk
        xi = self._xi

        neighbors_idx = self._neighbors_idx
        neighbors_dst = self._neighbors_dst

        I = np.ones((neighbors_idx.shape[1],), float, order='C')
        fi = np.empty((len(xi), self._vdim), float, order='C')

        for i, x0 in enumerate(xi):

            q = neighbors_idx[i,:]
            r = neighbors_dst[i,:]

            x = xk[q,0] - x0[0]
            y = xk[q,1] - x0[1]

            B = eval_basis(I, x, y)
            W = eval_weight(r)

            BTW = B.T*W
            A = BTW.dot(B)
            b = BTW.dot(fk[q,:])

            if det(A) > np.diag(A).prod() * tol:
                fi[i,:] = solve(A, b, assume_a='pos')[0]
            else:
                fi[i,:] = W.dot(fk[q,:])/W.sum()

        return fi

    @property
    def gdim(self):
        return self._gdim

    @property
    def vdim(self):
        return self._vdim


class MeshlessInterpolation:
    '''Depends on third party library.

    Meshless projection at points `xi` from a scatter of points `xk` and
    (vector-valued) function values `fk`.'''

    def __new__(cls, *args, **kwargs):
        if not HAS_WLSQM:
            raise ModuleNotFoundError('Require package "wlsqm".')
        return super(cls, cls).__new__(cls)

    def __init__(self, xk):
        '''
        Parameters
        ----------
        xk : numpy.ndarray
            Data points where values of a function are known.

        '''

        if not isinstance(xk, ndarray) or xk.ndim != 2 or xk.shape[1] != 2:
            raise TypeError('Expected parameter `xk` to be a 2D `numpy.ndarray`.')

        self._xk = np.array(xk, np.float64, order='C', copy=copy)
        self._kdtree_xk = cKDTree(self._xk, copy_data=False)

        self._gdim = 2
        self._vdim = None

        self._xi = None
        self._fk = None

        self._neighbors_idx = None
        self._neighbors_dst = None

    def set_reference_values(self, fk):
        '''
        Parameters
        ----------
        fk : array or a list of 1D arrays:
            Function values at `xk`. If `fk` is a numpy.ndarray, it must be
            either 1D or 2D. If `fk` is a list or tuple, the items must be
            1D arrays of equal length.

        '''

        exc = TypeError('Parameter `fk` must either be a sequence (`list` or '
            '`tuple`) of 1D `numpy.ndarray`s of equal length, or a single 1D '
            '`numpy.ndarray` or a single 2D `numpy.ndarray`.')

        if isinstance(fk, (list,tuple)):

            if all(len(fk_i) == len(fk[0])
              and isinstance(fk_i, ndarray)
              and fk_i.ndim == 1 for fk_i in fk):

                if all(len(fk_i) == len(self._xk) for fk_i in fk):
                    self._fk = [np.array(fk_i, np.float64) for fk_i in fk]
                    self._vdim = len(self._fk)
                    return

                else: raise exc
            else: raise exc

        elif isinstance(fk, ndarray):

            if len(fk) == len(self._xk):

                if fk.ndim == 2:
                    self._fk = [np.array(fk_j, np.float64) for fk_j in fk.T]
                    self._vdim = len(self._fk)
                    return

                elif fk.ndim == 1:
                    self._fk = [np.array(fk, np.float64)]
                    self._vdim = 1
                    return

                else: raise exc
            else: raise exc
        else: raise exc

    def set_interpolation_points(self, xi, num_neighbors, distance_norm=2):
        '''
        Parameters
        ----------
        xi : numpy.ndarray
            Interpolation points where a function is to be interpolated.
        num_neighbors : integer
            Number of nearest neighbors to find.
        distance_norm : integer, optional
            Order of the dinstance norm for finding the nearest neighbors.

        '''

        if not isinstance(xi, ndarray) or xi.ndim != 2:
            raise TypeError('Expected parameter `xi` to be a 2D `numpy.ndarray`')

        if xi.shape[1] != self._gdim:
            raise ValueError('Expected geometric dimension of '
                'parameter `xi` to be the same as that of `xk`.')

        self._xi = np.array(xi, np.float64, order='C')

        self._neighbors_dst, self._neighbors_idx = self._kdtree_xk \
            .query(self._xi, k=num_neighbors, p=distance_norm)

    def interpolate(self, fk=None, degree=1, weight="center"):
        '''Interpolate previously given function values `fk` at new points `xi`.

        Parameters
        ----------
        fk : numpy.ndarray, list of numpy.ndarray's
            Discrete function values at `xk`. If `fk` is a numpy.ndarray, its
            shape must be either 1D or 2D. If `fk` is a list or tuple, the
            elements must be equal length 1D arrays.
        degree: integer (optional)
            The degree of meshless interpolation.
        weight: string (optional)
            The type of the weighting method: "center" or "uniform".

        Returns
        ------
        fi : numpy.ndarray
            Interpolated function values at interpolation points `xi`.

        '''

        if self._xi is None:
            raise AttributeError('`xi` is not set yet.')

        if fk is not None:
            self.set_reference_values(fk)

        elif self._fk is None:
            raise ValueError('`fk` is not set yet.')

        if weight == "uniform": weight_method = wlsqm.WEIGHT_UNIFORM
        elif weight == "center": weight_method = wlsqm.WEIGHT_CENTER
        else: raise ValueError('Expected parameter `weight` to be '
                               'either `"uniform"` or `"center"`.')

        fk = self._fk
        xk = self._xk

        xi = self._xi
        ni = len(xi)

        neighbors_idx = self._neighbors_idx
        neighbors_dst = self._neighbors_dst

        ki = neighbors_idx.size // len(neighbors_idx) # number of neighbors

        if ki == 1: # single neighbor edge case
            neighbors_idx = neighbors_idx[:,None]
            neighbors_dst = neighbors_dst[:,None]

        fi = np.empty((ni, self._vdim), order='F', dtype=float)

        solution = np.zeros(
            shape=(ni, wlsqm.number_of_dofs(self._gdim, degree)),
            dtype=np.float64) # worker array

        knowns = np.zeros(
            shape=(ni,),
            dtype=np.int64) # nothing's known about `fi`

        neighbors = np.full(
            shape=(ni,),
            fill_value=ki,
            dtype=np.int32)

        order = np.full(
            shape=(ni,),
            fill_value=degree,
            dtype=np.int32)

        weighting_method = np.full(
            shape=(ni,),
            fill_value=weight_method,
            dtype=np.int32)

        for i, fk_i in enumerate(fk):

            wlsqm.fit_2D_many_parallel(xk[neighbors_idx], fk_i[neighbors_idx],
                nk=neighbors, xi=xi, fi=solution, sens=None, do_sens=False,
                order=order, knowns=knowns, weighting_method=weighting_method)

            fi[:,i] = solution[:,0] # interpolated values are in 0th column
            q_nan = np.isnan(fi[:,i])

            if np.any(q_nan):
                print('WARNING: Number of nearest neighbors is too small. '
                    'Assuming the mean value of the neighbors as the solution.')
                fi[q_nan,i] = fk_i[neighbors_idx[q_nan,:]].mean(axis=1)

        return fi

    @property
    def gdim(self):
        return self._gdim

    @property
    def vdim(self):
        return self._vdim
