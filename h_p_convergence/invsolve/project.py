'''
invsolve/project.py

'''

import dolfin
import logging
import numpy as np
import scipy.linalg as linalg
from scipy.spatial import cKDTree

try:
    import wlsqm # optimized meshless
except:
    # logging.log(logging.WARNING, repr(ModuleNotFoundError))
    HAS_WLSQM = False
else:
    HAS_WLSQM = True

MESHLESS_NEIGHBORS_FROM_DEGREE = {
    0 : round(5*5 * np.pi/4),
    1 : round(7*7 * np.pi/4),
    2 : round(9*9 * np.pi/4),
    } # Concervative estimate of the required number of nearest neighbors
    # in a circular 2D domain for projecting a (rather) noisy pointcloud.


def project_pointvalues_on_functions(xk, fk, V_project, meshless_degree=0,
    num_neighbors=None, distance_norm=2, subdims_geo=None, subdims_val=None):
    '''Project measurements on functions using meshless interpolation.

    Given an array of points and a list of arrays of data values at those
    points, project these discrete values onto the degrees of freedom of a
    (vector-valued) function constructed from the function space by using
    the moving least squares meshless interpolation method.

    Parameters
    ----------
    xk : numpy.ndarray
        Points where data values are known.
    fk : list of numpy.ndarray's
        A sequence of data values at the points `xk`.
    V_project : dolfin.FunctionSpace
        Vector-valued function space onto which to do meshless projection.
        The projection will be onto the degrees of freedom the function.
    meshless_degree : integer, optional (default=2)
        Degree of the meshless interpolation.
    num_neighbors : integer, optional
        Number of nearest neighbors to use in meshless interpolation. If `None`
        An appropriate number is chosen from experience.
    distance_norm : integer, optional (default=2)
        The distance norm to use in finding nearest neighbors. The common
        choices are: 2 (Euclidean distance), and 1 (Manhattan distance).
    subdims_geo : list of integers
        Indices into the dof-coordinate dimensions of `V_project`. The indexed dof
        coordinates will be used as the interpolation points `xi` where `fk`,
        which is only defined at `xk`, will be interpolated.
    subdims_val : list of integers
        Indices into the dof value dimensions of `V_project`. The indexed dof values
        of `V_project` will be the values obtained by interpolating `fk` at `xi`.

    Returns
    -------
    fn : list of dolfin.Function's
        A list of functions. The functions correspond to the snapshots `fk`
        interplated at the degrees-of-freedom coordinates of the function space
        `V_project`.

    '''

    if not isinstance(xk, np.ndarray) or not (1 <= xk.ndim <= 2):
        raise TypeError('Parameter `xk` must be a 1D or 2D `numpy.ndarray`.')

    if not isinstance(V_project, dolfin.FunctionSpace):
        raise TypeError('Parameter `V_project` must be a `dolfin.FunctionSpace`.')

    if isinstance(fk, np.ndarray):
        fk = (fk,) # must be a sequence
    else:
        if not isinstance(fk, (list,tuple)) or \
           not all(isinstance(uk_i, np.ndarray) for uk_i in fk):
            raise TypeError('Parameter `fk` must be a `numpy.ndarray`, '
                'or a `list` or `tuple` of `numpy.ndarray`s.')

        if any(uk_i.shape != fk[0].shape for uk_i in fk[1:]):
            raise TypeError('Parameter `fk` must contain '
                '`numpy.ndarray`s of the same shape.')

    if not (1 <= fk[0].ndim <= 2):
        raise TypeError('Parameter `fk` must contain '
            'either 1D or 2D `numpy.ndarray`s.')

    V = V_project # alias

    gdim_data = xk.size//len(xk)
    vdim_data = fk[0].size//len(fk[0])

    gdim_proj = V.element().geometric_dimension()
    vdim_proj = V.element().num_sub_elements()

    if vdim_proj == 0: # scalar function space
        vdim_proj = 1 # supposed value dimension

    if gdim_data != gdim_proj:
        if not subdims_geo or gdim_data != len(subdims_geo):
            raise ValueError('Expected the same geometric dimension of parameter '
                '`xk` (={gdim_data:d}) and parameter `V` (={gdim_proj:d}).')

    if vdim_data != vdim_proj:
        if not subdims_val or vdim_data != len(subdims_val):
            raise ValueError('Expected the same value dimension of parameter '
                '`fk` (={vdim_data:d}) and parameter `V` (={vdim_proj:d}).')

    if vdim_proj > 1: # `V` is vector-valued (can be split)
        dofmap = [V_i.dofmap().dofs() for V_i in V.split()]
    else: # `V` is a scalar-valued (can not be split)
        dofmap = [V.dofmap().dofs()]

    dofcrd = V.tabulate_dof_coordinates().reshape((-1,gdim_proj))
    dofcrd = [dofcrd[dofmap_i,:] for dofmap_i in dofmap]

    if __debug__:
        if not all(np.allclose(dofcrd[0], dofcrd_i) for dofcrd_i in dofcrd[1:]):
            raise TypeError('DOF coordinates of sub-dimensions are not the same.')

    xi = dofcrd[0]

    if gdim_data != gdim_proj:
        xi = xi[:,subdims_geo]

    if vdim_data != vdim_proj:
        dofmap = [dofmap[i] for i in subdims_val]

    fi = project_pointvalues_on_points(xk, fk, xi,
        meshless_degree, num_neighbors, distance_norm)

    fn = []

    for fi_t in fi:

        fn.append(dolfin.Function(V))
        fn_t = fn[-1].vector().get_local()

        for dofmap_j, fi_tj in zip(dofmap, fi_t.T):
            fn_t[dofmap_j] = fi_tj

        fn[-1].vector()[:] = fn_t

    return fn # list of functions


def project_pointvalues_on_points(xk, fk, xi,
    meshless_degree=0, num_neighbors=None, distance_norm=2):
    '''
    Parameters
    ----------
    xk : numpy.ndarray
        Points where data values are known.
    fk : list of numpy.ndarray's
        A sequence of data values at the points `xk`.
    xi : numpy.ndarray
        Points where the known point values are to be projected.
    meshless_degree : integer, optional (default=2)
        Degree of the meshless interpolation.
    num_neighbors : integer, optional
        Number of nearest neighbors to use in meshless interpolation. If `None`
        An appropriate number is chosen from experience.
    distance_norm : integer, optional (default=2)
        The distance norm to use in finding nearest neighbors. The common
        choices are: 2 (Euclidean distance), and 1 (Manhattan distance).

    Returns
    -------
    ui : list of numpy.ndarray's
        The projected point values.

    '''

    if not isinstance(xk, np.ndarray) or not (1 <= xk.ndim <= 2):
        raise TypeError('Parameter `xk` must be a 1D or 2D `numpy.ndarray`.')

    if not isinstance(xi, np.ndarray) or not (1 <= xk.ndim <= 2):
        raise TypeError('Parameter `xk` must be a 1D or 2D `numpy.ndarray`.')

    if isinstance(fk, np.ndarray):
        fk = (fk,) # must be a sequence
    else:
        if not isinstance(fk, (list,tuple)) or \
           not all(isinstance(fk_i, np.ndarray) for fk_i in fk):
            raise TypeError('Parameter `fk` must be a `numpy.ndarray`, '
                'or a `list` or `tuple` of `numpy.ndarray`s.')

        if any(fk_i.shape != fk[0].shape for fk_i in fk[1:]):
            raise TypeError('Parameter `fk` must contain '
                '`numpy.ndarray`s of the same shape.')

    if not (1 <= fk[0].ndim <= 2):
        raise TypeError('Parameter `fk` must contain '
            'either 1D or 2D `numpy.ndarray`s.')

    if HAS_WLSQM: meshless = MeshlessInterpolation(xk, copy=False)
    else: meshless = SimpleMeshlessInterpolation2d(xk, copy=False)

    if not num_neighbors:
        num_neighbors = MESHLESS_NEIGHBORS_FROM_DEGREE[meshless_degree]

    len_fk = len(fk)
    dim_fk = fk[0].ndim

    if dim_fk == 1:
        fk = np.stack(fk, axis=1)
    else: # fk[0].ndim == 2:
        fk = np.concatenate(fk, axis=1)

    meshless.set_reference_values(fk, copy=False)

    meshless.set_interpolation_points(xi,
        num_neighbors, distance_norm, copy=False)

    fi = meshless.interpolate(degree=meshless_degree)

    fi = np.split(fi, len_fk, axis=1)
    if dim_fk == 1 and fi[0].ndim == 2:
        fi = [f.squeeze() for f in fi]

    return fi


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
        '''The weight function "WEIGHT_CENTER" used in `wlsqm`.'''
        return np.where(r < 1.0, 1e-4 + (1.0-1e-4) * (1.0-r)**2, 1e-4)

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

    def __init__(self, xk, fk=None, copy=True):

        if not isinstance(xk, np.ndarray) or xk.ndim != 2 or xk.shape[1] != 2:
            raise TypeError('Expected parameter `xi` to be a 2D `numpy.ndarray`.')

        self._gdim = 2 # NOTE: 2D meshless interpolation

        self._xk = np.array(xk, float, order='C', copy=copy)
        self._kdtree_xk = cKDTree(self._xk, copy_data=False)

        self._xi = None
        self._fk = None
        self._vdim = None

        self._neighbors_xi = None
        self._neighbors_ri = None

        if fk is not None:
            self.set_reference_values(fk, copy) # -> self._fk, self._vdim

    def set_reference_values(self, fk, copy=True):
        '''
        Parameters
        ----------
        fk : array or a list of 1-D arrays:
            Function values at `xk`. If `fk` is a np.ndarray, it must be
            either 1-D or 2D. If `fk` is a list or tuple, the items must be
            1-D arrays of equal length.
        copy : bool (optional)
            Whether to copy data `fk` or not.

        '''

        exc = TypeError('Parameter `fk` must either be a sequence (`list` or '
            '`tuple`) of 1-D `np.ndarray`s of equal length, or a single 1-D '
            '`np.ndarray` or a single 2D `np.ndarray`.')

        if isinstance(fk, (list,tuple)):

            if all(len(fk_i) == len(fk[0])
              and isinstance(fk_i, np.ndarray)
              and fk_i.ndim == 1 for fk_i in fk):

                if all(len(fk_i) == len(self._xk) for fk_i in fk):
                    self._fk = np.asarray(np.stack(fk, axis=1), order='C')
                    self._vdim = len(fk)
                    return

                else: raise exc
            else: raise exc

        elif isinstance(fk, np.ndarray):

            if len(fk) == len(self._xk):

                if fk.ndim == 2:
                    self._fk = np.array(fk, order='C', copy=copy)
                    self._vdim = fk.shape[1]
                    return

                elif fk.ndim == 1:
                    self._fk = np.array(fk[:,None], copy=copy)
                    self._vdim = 1
                    return

                else: raise exc
            else: raise exc
        else: raise exc

    def set_interpolation_points(self, xi, num_neighbors, distance_norm=2, copy=True):
        '''
        Parameters
        ----------
        xi : np.ndarray
            Interpolation points where a function is to be interpolated.
        num_neighbors : integer
            Number of nearest neighbors to find.
        distance_norm : integer, optional
            Order of the dinstance norm for finding the nearest neighbors.

        '''

        if not isinstance(xi, np.ndarray) or xi.ndim != 2:
            raise TypeError('Expected parameter `xi` to be a 2D `numpy.ndarray`.')

        if xi.shape[1] != self._gdim:
            raise ValueError('Expected geometric dimension of '
                'parameter `xi` to be the same as that of `xk`.')

        self._xi = np.array(xi, float, order='C', copy=copy)

        self._neighbors_ri, self._neighbors_xi = \
            self._kdtree_xk.query(self._xi, k=num_neighbors, p=distance_norm)

    def interpolate(self, fk=None, degree=1, weight='uniform'):
        '''Interpolate previously given function values `fk` at new points `xi`.

        Parameters
        ----------
        fk : np.ndarray, list of np.ndarray's
            Discrete function values at `xk`. If `fk` is a np.ndarray, its
            shape must be either 1D or 2D. If `fk` is a list or tuple, the
            elements must be equal length 1D arrays.
        degree: integer (optional)
            The degree of meshless interpolation.
        weight: string (optional)
            The kind of weighting method to use. There are two options:
            "uniform" or "center". The former is better suited to interpolating
            data that arises from a smooth function. The latter is better
            suited to interpolating data that arises from not a smooth function.

        Returns
        ------
        fi : np.ndarray
            Interpolated function values at interpolation points `xi`.

        '''

        if self._xi is None:
            raise AttributeError('`xi` is not set yet.')

        if fk is not None:
            self.set_reference_values(fk, copy=False)
        elif self._fk is None:
            raise AttributeError('`fk` is not set yet.')

        if   degree == 0: eval_basis = self._eval_basis_p0
        elif degree == 1: eval_basis = self._eval_basis_p1
        elif degree == 2: eval_basis = self._eval_basis_p2
        else: raise ValueError('degree?')

        if   weight == 'uniform' : eval_weight = self._eval_weight_uniform
        elif weight == 'center'  : eval_weight = self._eval_weight_center
        else: raise ValueError('Parameter `weight`: "uniform" or "center" ?')

        fk = self._fk
        xk = self._xk
        xi = self._xi

        neighbors_xi = self._neighbors_xi
        neighbors_ri = self._neighbors_ri

        ki = neighbors_xi.size // len(neighbors_xi) # number of neighbors

        if ki == 1: # single neighbor edge case
            neighbors_xi = neighbors_xi[:,None]
            neighbors_ri = neighbors_ri[:,None]

        I = np.ones((ki,), float)

        fi = np.empty((len(xi), self._vdim), float, order='C')

        for i, x0 in enumerate(xi):

            q = neighbors_xi[i,:]
            r = neighbors_ri[i,:]

            x = xk[q,0] - x0[0]
            y = xk[q,1] - x0[1]

            B = eval_basis(I, x, y)
            W = eval_weight(r/r.max())

            BTW = B.T*W
            A = BTW.dot(B)
            b = BTW.dot(fk[q,:])

            try:
                fi[i,:] = linalg.solve(A, b, sym_pos=True)
                # 0th item gives interpolation values at x0
            except linalg.LinAlgError as err:
                print('WARNING: Number of nearest neighbors is too small. '
                    'Assuming the mean value of the neighbors as the solution.')
                fi[i,:] = W.dot(fk[q,:])/W.sum()

        return fi

    @property
    def gdim(self):
        return self._gdim

    @property
    def vdim(self):
        return self._vdim


class MeshlessInterpolation:
    '''Depends on third party library

    Meshless projection at points `xi` from a scatter of points `xk` and
    (vector-valued) function values `fk`.'''

    def __new__(cls, *args, **kwargs):
        if not HAS_WLSQM:
            raise ModuleNotFoundError('Require package "wlsqm".')
        return super(cls, cls).__new__(cls)

    def __init__(self, xk, fk=None, copy=True):
        ''' Give data points: point coordinates `xk` and (vector-valued)
        function values `fk`, prepare instance for meshless projection.

        Parameters
        ----------
        xk : np.ndarray
            Data points where values of a function are known.
        fk : np.ndarray, list of np.ndarray's
            Discrete function values at `xk`. If `fk` is a np.ndarray, its
            shape must be either 1D or 2D. If `fk` is a list or tuple, the
            elements must be equal length 1D arrays.


        '''

        if not isinstance(xk, np.ndarray) or xk.ndim != 2 or xk.shape[1] != 2:
            raise TypeError('Expected parameter `xi` to be a 2D `numpy.ndarray`.')

        self._gdim = 2 # NOTE: 2D meshless interpolation

        self._xk = np.array(xk, np.float64, order='C', copy=copy)
        self._kdtree_xk = cKDTree(self._xk, copy_data=False)

        self._xi = None
        self._fk = None
        self._vdim = None

        self._neighbors_xi = None
        self._neighbors_ri = None

        if fk is not None:
            self.set_reference_values(fk, copy)
            # -> self._fk, self._vdim

    def set_reference_values(self, fk, copy=True):
        '''
        Parameters
        ----------
        fk : array or a list of 1-D arrays:
            Function values at `xk`. If `fk` is a np.ndarray, it must be
            either 1-D or 2D. If `fk` is a list or tuple, the items must be
            1-D arrays of equal length.
        copy : bool (optional)
            Whether to copy data `fk` or not.

        '''

        exc = TypeError('Parameter `fk` must either be a sequence (`list` or '
            '`tuple`) of 1-D `np.ndarray`s of equal length, or a single 1-D '
            '`np.ndarray` or a single 2D `np.ndarray`.')

        if isinstance(fk, (list,tuple)):

            if all(len(fk_i) == len(fk[0])
              and isinstance(fk_i, np.ndarray)
              and fk_i.ndim == 1 for fk_i in fk):

                if all(len(fk_i) == len(self._xk) for fk_i in fk):
                    self._fk = [np.array(fk_i, np.float64,
                        order='C', copy=copy) for fk_i in fk]
                    self._vdim = len(self._fk)
                    return

                else: raise exc
            else: raise exc

        elif isinstance(fk, np.ndarray):

            if len(fk) == len(self._xk):

                if fk.ndim == 2:
                    self._fk = [np.array(fk_j, np.float64,
                        order='C', copy=copy) for fk_j in fk.T]
                    self._vdim = len(self._fk)
                    return

                elif fk.ndim == 1:
                    self._fk = [np.array(fk, np.float64, copy=copy)]
                    self._vdim = 1
                    return

                else: raise exc
            else: raise exc
        else: raise exc

    def set_interpolation_points(self, xi, num_neighbors, distance_norm=2, copy=True):
        '''
        Parameters
        ----------
        xi : np.ndarray
            Interpolation points where a function is to be interpolated.
        num_neighbors : integer
            Number of nearest neighbors to find.
        distance_norm : integer, optional
            Order of the dinstance norm for finding the nearest neighbors.

        '''

        if not isinstance(xi, np.ndarray) or xi.ndim != 2:
            raise TypeError('Expected parameter `xi` to be a 2D `np.ndarray`')

        if xi.shape[1] != self._gdim:
            raise ValueError('Expected geometric dimension of '
                'parameter `xi` to be the same as that of `xk`.')

        self._xi = np.array(xi, np.float64, order='C', copy=copy)

        self._neighbors_ri, self._neighbors_xi = \
            self._kdtree_xk.query(self._xi, k=num_neighbors, p=distance_norm)

    def interpolate(self, fk=None, degree=1, weight='uniform'):
        '''Interpolate previously given function values `fk` at new points `xi`.

        Parameters
        ----------
        fk : np.ndarray, list of np.ndarray's
            Discrete function values at `xk`. If `fk` is a np.ndarray, its
            shape must be either 1D or 2D. If `fk` is a list or tuple, the
            elements must be equal length 1D arrays.
        degree: integer (optional)
            The degree of meshless interpolation.
        weight: string (optional)
            The kind of weighting method to use. There are two options:
            "uniform" or "center". The former is better suited to interpolating
            data that arises from a smooth function. The latter is better
            suited to interpolating data that arises from not a smooth function.

        Returns
        ------
        fi : np.ndarray
            Interpolated function values at interpolation points `xi`.

        '''

        if self._xi is None:
            raise AttributeError('`xi` is not set yet.')

        if fk is not None:
            self.set_reference_values(fk, copy=False)
        elif self._fk is None:
            raise AttributeError('`fk` is not set yet.')

        if   weight == 'uniform' : weight_method = wlsqm.WEIGHT_UNIFORM
        elif weight == 'center'  : weight_method = wlsqm.WEIGHT_CENTER
        else: raise ValueError('Parameter `weight`: "uniform" or "center" ?')

        fk = self._fk
        xk = self._xk

        xi = self._xi
        ni = len(xi)

        neighbors_xi = self._neighbors_xi
        neighbors_ri = self._neighbors_ri

        ki = neighbors_xi.size // len(neighbors_xi) # number of neighbors

        if ki == 1: # single neighbor edge case
            neighbors_xi = neighbors_xi[:,None]
            neighbors_ri = neighbors_ri[:,None]

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

            wlsqm.fit_2D_many_parallel(xk[neighbors_xi], fk_i[neighbors_xi],
                nk=neighbors, xi=xi, fi=solution, sens=None, do_sens=False,
                order=order, knowns=knowns, weighting_method=weighting_method)

            fi[:,i] = solution[:,0] # interpolated values are in 0th column
            q_nan = np.isnan(fi[:,i])

            if np.any(q_nan):
                print('WARNING: Number of nearest neighbors is too small. '
                    'Assuming the mean value of the neighbors as the solution.')
                fi[q_nan,i] = fk_i[neighbors_xi[q_nan,:]].mean(axis=1)

        return fi

    @property
    def gdim(self):
        return self._gdim

    @property
    def vdim(self):
        return self._vdim
