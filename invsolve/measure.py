'''
invsolve/measure.py

'''

import numpy
import dolfin


class MeasurementExpression(dolfin.UserExpression):
    '''Make a function-like object that can be evaluated at an arbitrary
    coordinate `x` in space and time `t` (or at index `i`).

    With respect to the evaluation at time `t`, the value is linearly
    interpolated between two measurement snapshots evaluated at coordinate `x`.
    '''

    _msr_rtol = 1e-9 # for finding the time index `i_t` given a time `t`

    def __new__(cls, f_msr, t_msr=None, **kwargs):

        if 'degree' not in kwargs and 'element' not in kwargs:
            raise TypeError('Require `degree` or `element` as keyword argument.')

        if all(isinstance(f, dolfin.Function) for f in f_msr):

            self = super().__new__(cls)
            self._ufl_shape = f_msr[0].ufl_shape

            self._msr_is_dolfin_function_type = True
            self._msr_is_numpy_ndarray_type = False

        elif all(isinstance(f, numpy.ndarray) for f in f_msr):

            self = super().__new__(cls)
            self._ufl_shape = f_msr[0].shape

            self._msr_is_dolfin_function_type = False
            self._msr_is_numpy_ndarray_type = True

        else:
            raise TypeError('Expected parameter `f_msr` to be a sequence of '
                '`dolfin.Function`s or `numpy.ndarray`s.')

        return self

    def __init__(self, f_msr, t_msr=None, **kwargs):
        '''

        Parameters
        ----------
        f_msr : list or tuple of dolfin.Function's
            Vector-valued functions corresponding to the measurement snapshots.
        t_msr : list or tuple or numpy.ndarray (1D), optional
            Measurement times.

        Keyword Parameters
        ------------------
        degree : integer (optional)
            The `degree` must be given if no `element` is given.
        element : dolfin.Element (optional)
            The `element` must be given if no `degree` is given.

        '''

        # Init. dolfin.UserExpression
        super().__init__(**kwargs)

        n_msr = len(f_msr)

        if t_msr is None:
            t_msr = tuple(range(n_msr))

        elif hasattr(t_msr, '__iter__'):
            t_msr = tuple(t_msr)

            if len(t_msr) != n_msr:
                raise TypeError('Lengths of `f_msr` and `t_msr` are not the same.')

            elif not any(t_i <= t_j for t_i, t_j in zip(t_msr[:-1], t_msr[1:])):
                raise TypeError('`t_msr` must be in a sorted (ascending) order.')

        else:
            t_msr = tuple(numpy.linspace(0, t_msr, n_msr))

        if n_msr > 1:
            self._msr_atol = self._msr_rtol * (t_msr[-1]-t_msr[0])
        else:
            self._msr_atol = self._msr_rtol

        self._msr_f_msr = f_msr
        self._msr_t_msr = t_msr
        self._msr_n_msr = n_msr

        if self._msr_is_dolfin_function_type:
            self._msr_f_now = dolfin.Function.copy(f_msr[0], deepcopy=True)

        elif self._msr_is_numpy_ndarray_type:
           self._msr_f_now = numpy.array(f_msr[0], dtype=float, copy=True)

        else:
           raise TypeError

        self._msr_t_now = t_msr[0]
        self._msr_i_now = 0

    def set_measurement_time(self, t):
        '''Setting function at time. The function is then
        linearly interpolated between adjacent time points.'''

        if (t < self._msr_t_msr[0]-self._msr_atol or
            t > self._msr_t_msr[-1]+self._msr_atol):
            raise ValueError('Input time `t` out of range')

        if abs(t-self._msr_t_now) > self._msr_atol:

            i = self.get_measurement_index_from_time(t, self._msr_i_now)
            assert (0 <= i < self._msr_n_msr-1), print(i)

            w = (t-self._msr_t_msr[i])/(self._msr_t_msr[i+1]-self._msr_t_msr[i])
            assert (-self._msr_rtol < w < 1.0 + self._msr_rtol), print(w)

            self._msr_t_now = t
            self._msr_i_now = i

            f0 = self._msr_f_msr[i]
            f1 = self._msr_f_msr[i+1]

            if self._msr_is_dolfin_function_type:
                self._msr_f_now.vector()[:] = f0.vector()*(1.0-w) + f1.vector()*w

            elif self._msr_is_numpy_ndarray_type:
               self._msr_f_now[:] = f0*(1.0-w) + f1*w

            else:
               raise TypeError

    def set_measurement_index(self, i):
        if i < 0: i += self._msr_n_msr
        if 0 <= i < self._msr_n_msr:

            if self._msr_is_dolfin_function_type:
                self._msr_f_now.vector()[:] = self._msr_f_msr[i].vector()

            elif self._msr_is_numpy_ndarray_type:
               self._msr_f_now[:] = self._msr_f_msr[i]

            else:
               raise TypeError

            self._msr_t_now = self._msr_t_msr[i]
            self._msr_i_now = i

        else:
            raise IndexError('Input index `i` out of range')

    def get_measurement_index_from_time(self, t, i_start=0):
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

    def get_measurement_time(self):
        return self._msr_t_now

    def get_measurement_index(self):
        return self._msr_i_now

    def get_measurement_times(self):
        return self._msr_t_msr

    def get_measurement(self, copy=True):

        if self._msr_is_dolfin_function_type:
            if copy: return self._msr_f_now.copy(True)
            else: return self._msr_f_now

        elif self._msr_is_numpy_ndarray_type:
            if copy: return self._msr_f_now.copy()
            else: return self._msr_f_now

        else:
           raise TypeError

    def get_measurement_at_index(self, i):

        i_now = self._msr_i_now
        self.set_measurement_index(i)

        f_msr_i = self.get_measurement(copy=True)
        self.set_measurement_index(i_now)

        return f_msr_i

    def get_measurement_at_time(self, t):

        t_now = self._msr_t_now
        self.set_measurement_time(t)

        f_msr_t = self.get_measurement(copy=True)
        self.set_measurement_time(t_now)

        return f_msr_t

    def value_shape(self):
        # NOTE: Must be defined if deriving from `dolfin.UserExpression`
        return self._ufl_shape

    def eval(self, value, x):
        if self._msr_is_dolfin_function_type:
            self._msr_f_now.eval(value, x)
        else: # self._msr_is_numpy_ndarray_type
            value[:] = self._msr_f_now
