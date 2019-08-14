'''
For converting a sequence of measurements into a measurement expression.

'''

import numpy as np
from dolfin import Function
from dolfin import UserExpression

from matplotlib.tri import Triangulation
from matplotlib.tri import LinearTriInterpolator

SEQUENCE_TYPES = (tuple, list)


def make_measurement_setter_with_time_as_argument(*args):
    '''Make a measurement setter function for expressions.

    Parameters
    ----------
    args : MeasurementExpressionBase
        Measurement expressions.

    Returns
    -------
    measurement_setter : function(t:float)
        Measurement setting function.

    '''

    if not all(isinstance(arg, MeasurementExpressionBase) for arg in args):
        raise TypeError('`args` must have base type `MeasurementExpressionBase`.')

    def measurement_setter(t:float):
        '''Set all measurements at time.'''
        for arg in args: arg.at_time(t)

    return measurement_setter


def make_measurement_setter_with_index_as_argument(*args):
    '''Make a measurement setter function for expressions.

    Parameters
    ----------
    args : MeasurementExpressionBase
        Measurement expressions.

    Returns
    -------
    measurement_setter : function(i:int)
        Measurement setting function.

    '''

    if not all(isinstance(arg, MeasurementExpressionBase) for arg in args):
        raise TypeError('`args` must have base type `MeasurementExpressionBase`.')

    def measurement_setter(i:int):
        '''Set all measurements at index.'''
        for arg in args: arg.at_index(i)

    return measurement_setter


def measurement_expression(f_msr, t_msr=None, degree=None):
    '''Return a suitable measurement expression for the type of parameters.

    Parameters
    ----------
    f_msr : sequence of dolfin.Function's or numpy.ndarray's
        Sequence of measurement snapshots.
    t_msr : a sequence of ascending values or a single value (optional)
        Measurement times. Could be a sequence of values for the measurement
        snapshots, or a sequence of two values for the first time and the
        last time of the snapshots, or a single value for the last time.

    '''

    if not isinstance(f_msr, (list, tuple, np.ndarray)):
        raise TypeError('Expecting parameter `f_msr` to be a '
                        'sequence of measurement snapshots.')

    if all(isinstance(f_msr_i, Function) for f_msr_i in f_msr):
        return MeasurementExpressionFromFunctions(f_msr, t_msr, degree= \
            f_msr[0].ufl_element().degree() if degree is None else degree)

    elif all(isinstance(f_msr_i, np.ndarray) for f_msr_i in f_msr):
        return MeasurementExpressionFromArrays(f_msr, t_msr, degree=0)

    elif all(isinstance(f_msr_i, (float, int)) for f_msr_i in f_msr):
        return MeasurementExpressionFromScalars(f_msr, t_msr, degree=0)

    else:
        raise TypeError('Expected parameter `f_msr` to be a sequence of '
                        'either `dolfin.Function`s or `numpy.ndarray`s.')


class MeasurementExpressionBase(UserExpression):

    _msr_rtol = 1e-14

    def __new__(cls, *args, **kwargs):
        '''Must be extended by deriving class.'''

        if 'degree' not in kwargs and 'element' not in kwargs:
            raise TypeError('Require `degree` or `element` as keyword argument.')

        self = super().__new__(cls)
        self._ufl_shape = None

        return self

    def __init__(self, f_msr, t_msr=None, **kwargs):
        '''Must be extended by deriving class.

        Parameters
        ----------
        f_msr : sequence of object's
            Sequence of measurement snapshots.
        t_msr : a sequence of ascending values or a single value (optional)
            Measurement times. Could be a sequence of values for the measurement
            snapshots, or a sequence of two values for the first time and the
            last time of the snapshots, or a single value for the last time.

        Keyword Parameters
        ------------------
        degree : int
            The `degree` must be given if no `element` is given.
        element : dolfin.Element (optional)
            The `element` must be given if no `degree` is given.

        '''

        # Must initialize base class
        super().__init__(**kwargs)

        n_msr = len(f_msr)

        if t_msr is None:
            t_msr = tuple(np.linspace(0, 1, n_msr, dtype=float))

        elif not hasattr(t_msr, '__getitem__'):
            t_msr = tuple(np.linspace(0, t_msr, n_msr, dtype=float))

        else:

            if not all(t_i < t_j for t_i, t_j in zip(t_msr[:-1], t_msr[1:])):
                raise TypeError('Parameter `t_msr` must be an ascending sequence.')

            if len(t_msr) == 2:
                t_msr = tuple(np.linspace(t_msr[0], t_msr[1], n_msr, dtype=float))

            elif len(t_msr) != n_msr:
                raise TypeError('Parameter `t_msr` is incompatible with `f_msr`.')

        if len(t_msr) > 1:
            self._msr_atol = self._msr_rtol * (t_msr[-1] - t_msr[0])
        else:
            self._msr_atol = self._msr_rtol

        self._msr_f_msr = f_msr if isinstance(f_msr, tuple) else tuple(f_msr)
        self._msr_t_msr = t_msr if isinstance(t_msr, tuple) else tuple(t_msr)
        self._msr_n_msr = n_msr

        self._msr_f_cur = None
        self._msr_t_cur = t_msr[0]
        self._msr_i_cur = 0

    def __repr__(self):
        return f'<{self.__class__.__name__} at {hex(id(self))}>'

    def _msr_index_from_time(self, t, i_start=0):
        '''Find the index `i` that corresponds to the left of (or at) time `t`.
        `i_start` can be specified to start the search around index `i_start`,
        otherwise `i_start=0` and so the search starts from begining.'''

        while i_start < 0:
            i_start += self._msr_n_msr

        if t >= self._msr_t_msr[i_start]:
            # search to the right of `i_start`

            if t >= self._msr_t_msr[-2]: # edge case
                return self._msr_n_msr-2

            # NOTE: t >= self._msr_t_msr[i_start] and t < self._msr_t_msr[-2]
            # hence, first lesser between `i_start+1` and `end-1`

            return next(i for i, t_j in enumerate(
                self._msr_t_msr[i_start+1:-1], i_start) if t < t_j)

        else: # t < self._msr_t_msr[i_start]:
            # search to the left of `i_start`

            if t <= self._msr_t_msr[1]: # edge case
                return 0

            # NOTE: t < self._msr_t_msr[i_start] and t > self._msr_t_msr[1]
            # hence, first greater between `i_start-1` and `0`

            return next(i_start-i for i, t_j in enumerate(
                self._msr_t_msr[i_start-1:0:-1], start=1) if t > t_j)

    def _msr_index_and_weight_from_time(self, t):
        '''Index and weight of the adjacent left measurement for time `t`.'''

        if (t < self._msr_t_msr[0]-self._msr_atol or
            t > self._msr_t_msr[-1]+self._msr_atol):
            raise ValueError('Measurement time `t` out of range.')

        i = self._msr_index_from_time(t, self._msr_i_cur)
        assert (0 <= i < self._msr_n_msr-1), f'i = {i}'

        w = (self._msr_t_msr[i+1]-t)/(self._msr_t_msr[i+1]-self._msr_t_msr[i])
        assert (-self._msr_rtol < w < 1.0 + self._msr_rtol), f'w = {w}'

        return i, w

    @property
    def n_msr(self):
        '''Number of measurements.'''
        return self._msr_n_msr

    @property
    def t_msr(self):
        '''All measurement times.'''
        return self._msr_t_msr

    @property
    def f_msr(self):
        '''All measurement values.'''
        return self._msr_f_msr

    def at_index(self, i):
        '''Set measurement at index.'''
        raise NotImplementedError

    def at_time(self, t):
        '''Set measurement at time.'''
        raise NotImplementedError

    def get_index(self):
        '''Current measurement index.'''
        return self._msr_i_cur

    def get_time(self):
        '''Current measurement time.'''
        return self._msr_t_cur

    def get_value(self, copy=True):
        '''Current measurement value.'''
        return NotImplementedError

    def eval(self, value, x):
        raise NotImplementedError

    def value_shape(self):
        return self._ufl_shape


class MeasurementExpressionFromFunctions(MeasurementExpressionBase):

    def __new__(cls, f_msr, *args, **kwargs):
        self = super().__new__(cls, **kwargs)

        if not isinstance(f_msr, SEQUENCE_TYPES) or \
           not all(isinstance(f, Function) for f in f_msr):
            raise TypeError('Parameter `f_msr` must be a '
                            'sequence of `dolfin.Function`s.')

        self._ufl_shape = f_msr[0].ufl_shape

        return self

    def __init__(self, f_msr, t_msr=None, **kwargs):
        '''

        Parameters
        ----------
        f_msr : sequence of dolfin.Function's.
            Sequence of measurement snapshots.
        t_msr : a sequence of ascending values or a single value (optional)
            Measurement times. Could be a sequence of values for the measurement
            snapshots, or a sequence of two values for the first time and the
            last time of the snapshots, or a single value for the last time.

        Keyword Parameters
        ------------------
        degree : int
            The `degree` must be given if no `element` is given.
        element : dolfin.Element (optional)
            The `element` must be given if no `degree` is given.

        '''

        super().__init__(f_msr, t_msr, **kwargs)
        self._msr_f_cur = Function.copy(f_msr[0], deepcopy=True)

    def at_index(self, i):
        '''Set measurement at index `i`.'''

        if i < 0:
            i += self._msr_n_msr

        try:
            self._msr_f_cur.vector()[:] = self._msr_f_msr[i].vector()
            self._msr_i_cur, self._msr_t_cur = i, self._msr_t_msr[i]
        except IndexError:
            raise IndexError('Measurement index `i` out of range.')

        return self

    def at_time(self, t):
        '''Set measurement at time `t`.'''

        # Adjacent left measurement index and weight
        i, w = self._msr_index_and_weight_from_time(t)

        self._msr_f_cur.vector()[:] = self._msr_f_msr[i].vector()*w \
                                    + self._msr_f_msr[i+1].vector()*(1.0-w)
        self._msr_t_cur = t
        self._msr_i_cur = i

        return self

    def get_value(self, copy=True):
        '''Current measurement value.'''
        return self._msr_f_cur.copy(True) if copy else self._msr_f_cur

    def eval(self, value, x):
        self._msr_f_cur.eval(value, x)


class MeasurementExpressionFromArrays(MeasurementExpressionBase):

    def __new__(cls, f_msr, *args, **kwargs):
        self = super().__new__(cls, **kwargs)

        if not hasattr(f_msr, '__getitem__') or \
           not all(isinstance(f, np.ndarray) for f in f_msr):
            raise TypeError('Parameter `f_msr` must be a '
                            'sequence of `numpy.ndarray`s.')

        self._ufl_shape = f_msr[0].shape

        return self

    def __init__(self, f_msr, t_msr=None, **kwargs):
        '''

        Parameters
        ----------
        f_msr : sequence of numpy.ndarray's.
            Sequence of measurement snapshots.
        t_msr : a sequence of ascending values or a single value (optional)
            Measurement times. Could be a sequence of values for the measurement
            snapshots, or a sequence of two values for the first time and the
            last time of the snapshots, or a single value for the last time.

        Keyword Parameters
        ------------------
        degree : int
            The `degree` must be given if no `element` is given.
        element : dolfin.Element (optional)
            The `element` must be given if no `degree` is given.

        '''

        super().__init__(f_msr, t_msr, **kwargs)
        self._msr_f_cur = np.array(f_msr[0], float)

    def at_index(self, i):
        '''Set measurement at index `i`.'''

        if i < 0:
            i += self._msr_n_msr

        try:
            self._msr_f_cur[:] = self._msr_f_msr[i]
            self._msr_t_cur = self._msr_t_msr[i]
            self._msr_i_cur = i
        except IndexError:
            raise IndexError('Measurement index `i` out of range.')

        return self

    def at_time(self, t):
        '''Set measurement at time `t` by linear interpolation.'''

        # Adjacent left measurement index and weight
        i, w = self._msr_index_and_weight_from_time(t)

        self._msr_f_cur[:] = self._msr_f_msr[i]*w \
                           + self._msr_f_msr[i+1]*(1.0-w)

        self._msr_t_cur = t
        self._msr_i_cur = i

        return self

    def get_value(self, copy=True):
        '''Current measurement value.'''
        return self._msr_f_cur.copy() if copy else self._msr_f_cur

    def eval(self, value, x):
        value[:] = self._msr_f_cur


class MeasurementExpressionFromScalars(MeasurementExpressionBase):

    def __new__(cls, f_msr, *args, **kwargs):
        self = super().__new__(cls, **kwargs)

        if not hasattr(f_msr, '__getitem__') or \
           not all(isinstance(f, (float, int)) for f in f_msr):
            raise TypeError('Parameter `f_msr` must be a '
                            'sequence of float\'s or int\'s.')

        self._ufl_shape = ()

        return self

    def __init__(self, f_msr, t_msr=None, **kwargs):
        '''

        Parameters
        ----------
        f_msr : sequence of reals.
            Sequence of measurement snapshots.
        t_msr : a sequence of ascending values or a single value (optional)
            Measurement times. Could be a sequence of values for the measurement
            snapshots, or a sequence of two values for the first time and the
            last time of the snapshots, or a single value for the last time.

        Keyword Parameters
        ------------------
        degree : int
            The `degree` must be given if no `element` is given.
        element : dolfin.Element (optional)
            The `element` must be given if no `degree` is given.

        '''

        super().__init__(f_msr, t_msr, **kwargs)
        self._msr_f_cur = f_msr[0]

    def at_index(self, i):
        '''Set measurement at index `i`.'''

        if i < 0:
            i += self._msr_n_msr

        try:
            self._msr_f_cur = self._msr_f_msr[i]
            self._msr_t_cur = self._msr_t_msr[i]
            self._msr_i_cur = i
        except IndexError:
            raise IndexError('Measurement index `i` out of range.')

        return self

    def at_time(self, t):
        '''Set measurement at time `t` by linear interpolation.'''

        # Adjacent left measurement index and weight
        i, w = self._msr_index_and_weight_from_time(t)

        self._msr_f_cur = self._msr_f_msr[i]*w \
                        + self._msr_f_msr[i+1]*(1.0-w)

        self._msr_t_cur = t
        self._msr_i_cur = i

        return self

    def get_value(self, copy=True):
        '''Current measurement value.'''
        return self._msr_f_cur

    def eval(self, value, x):
        value[:] = self._msr_f_cur


# class MeasurementExpressionFromScatters(MeasurementExpressionFromArrays):
#
#     def __new__(cls, x_msr, f_msr, *args, **kwargs):
#         self = super().__new__(cls, f_msr, *args, **kwargs)
#
#         if any(f_i.ndim != 2 for f_i in f_msr):
#             raise TypeError('Parameter `f_msr` must contain 2D `numpy.ndarray`s.')
#
#         self._ufl_shape = f_msr[0][0].shape
#
#         return self
#
#     def __init__(self, x_msr, f_msr, t_msr=None, **kwargs):
#         '''
#
#         Parameters
#         ----------
#         x_msr : numpy.ndarray (2D)
#             Coordinates of measurement points.
#         f_msr : sequence of numpy.ndarray's.
#             Sequence of measurement snapshots.
#         t_msr : a sequence of ascending values or a single value (optional)
#             Measurement times. Could be a sequence of values for the measurement
#             snapshots, or a sequence of two values for the first time and the
#             last time of the snapshots, or a single value for the last time.
#
#         Keyword Parameters
#         ------------------
#         degree : int
#             The `degree` must be given if no `element` is given.
#         element : dolfin.Element (optional)
#             The `element` must be given if no `degree` is given.
#
#         '''
#
#         super().__init__(f_msr, t_msr, **kwargs)
#         tri = Triangulation(x_msr[:,0], x_msr[:,1])
#         self._msr_z_cur = np.empty((len(f_msr[0]),), float)
#         self.interp = LinearTriInterpolator(tri, self._msr_z_cur)
#
#
#     def __repr__(self):
#         return f'<{self.__class__.__name__} at {hex(id(self))}>'
#
#     def eval(self, value, x):
#         for i in range(self._ufl_shape[0]):
#             self._msr_z_cur[:] = self._msr_f_cur[:,i]
#             value[i] = self.interp(*x).data
#
#
#     def LinearInterpolator2D(self, xk, fk):
#
#         self._xk = np.array(xk, order='C')
#         self._fk = np.array(fk, order='F')
#         self._zk = np.empty((len(fk),))
#
#         tri = Triangulation(self._xk[:,0], self._xk[:,1])
#         self.interpolator = LinearTriInterpolator(tri, self._zk)
#
#         mesh = dolfin.Mesh()
#         editor = dolfin.MeshEditor()
#
#         editor.open(mesh, 'triangle', tdim=xk.shape[1], gdim=xk.shape[1])
#         editor.init_vertices(len(self._xk))
#         editor.init_cells(len(tri.triangles))
#
#         for i, v_i in enumerate(self._xk):
#             editor.add_vertex(i, v_i.tolist())
#
#         for i, c_i in enumerate(tri.triangles):
#             editor.add_cell(i, c_i.tolist())
#
#         editor.close()
