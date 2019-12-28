'''
Core module that defines the inverse solver.

'''

import dolfin
import logging
import numpy as np
import scipy.linalg as linalg

from math   import sqrt
from copy   import deepcopy
from dolfin import Constant
from dolfin import Function
from dolfin import Measure
from dolfin import action
from dolfin import assemble
from dolfin import assemble_system
from dolfin import derivative
from dolfin import diff

from ufl.form import Form as ufl_form_t
from ufl.core.expr import Expr as ufl_expr_t
from ufl.indexed import Indexed as ufl_indexed_t
from ufl.constantvalue import zero as ufl_zero

from . import config
from . import utility

SEQUENCE_TYPES = (tuple, list)
NUMERIC_TYPES = (int, float)

logger = config.logger


class InverseSolverBasic:
    '''Basic inverse solver for model parameter estimation.'''

    SENSITIVITY_METHOD_DIRECT = 'direct'
    SENSITIVITY_METHOD_ADJOINT = 'adjoint'
    INVERSE_SOLVER_METHOD_NEWTON = 'newton'
    INVERSE_SOLVER_METHOD_GRADIENT = 'gradient'

    def __init__(self, J, F, u, bcs, model_parameters,
                 observation_times, measurement_setter,
                 parameters_inverse_solver=None):
        '''
        Initialize basic inverse solver.

        Parameters
        ----------

        J : ufl.Form
            The cost functional to be minimized.

        F : ufl.Form
            Variational form.

        u : dolfin.Function
            The primary field, e.g. the displacement field.

        bcs : (sequence of) dolfin.DirichletBC('s)
            Dirichlet boundary conditions.

        model_parameters : contains dolfin.Constant's
            The model parameters to be optimized.

        observation_times : iterable
            Discrete times for which to consider the value of the model cost.

        measurement_setter : function-like that takes time as single argument
            Function that sets all measurements for a given observation time.

        '''

        if not isinstance(J, ufl_form_t):
            raise TypeError('Parameter `J`')

        if not isinstance(F, ufl_form_t):
            raise TypeError('Parameter `F`')

        if not isinstance(u, Function):
            raise TypeError('Parameter `u`')

        if not (isinstance(bcs, dolfin.DirichletBC) or (
                isinstance(bcs, SEQUENCE_TYPES) and all(
                isinstance(bc, dolfin.DirichletBC) for bc in bcs))):
            raise TypeError('Parameter `bcs`')

        self._model_parameters = utility.replicate_tree_structure(
            iterable=model_parameters, value_types=Constant)

        self._m = tuple(utility.list_values_from_iterable(
            iterable=model_parameters, value_types=Constant))

        self._n = len(self._m)

        self._J = J
        self._F = F
        self._u = u

        self._V = u.function_space()
        self._z = Function(self._V)

        self._dudm = tuple(Function(self._V) for i in range(self._n))
        self._d2udm2 = tuple(tuple(Function(self._V) for j in range(i, self._n))
            for i in range(self._n)) # NOTE: Storing only the upper triangle

        self._dJdu, self._dJdm, self._d2Jdu2, self._d2Jdudm, self._d2Jdm2 \
            = self._partial_derivatives_of_form(self._J)

        self._dFdu, self._dFdm, self._d2Fdu2, self._d2Fdudm, self._d2Fdm2 \
            = self._partial_derivatives_of_form(self._F)

        try:    action(self._d2Jdu2, self._u)
        except: self._is_actionable_d2Jdu2 = False
        else:   self._is_actionable_d2Jdu2 = True

        try:    action(self._d2Fdu2, self._u)
        except: self._is_actionable_d2Fdu2 = False
        else:   self._is_actionable_d2Fdu2 = True

        ### Dirichlet BC's

        self._bcs_zro = []
        self._bcs_dof = []

        for bc in bcs:
            self._bcs_dof.extend(bc.get_boundary_values().keys())
            bc_zro = dolfin.DirichletBC(bc); bc_zro.homogenize()
            self._bcs_zro.append(bc_zro)

        self._bcs_dof = np.array(self._bcs_dof, dtype=np.uint)

        ### Solvers

        self._dFdu_mtx = dolfin.PETScMatrix()
        self._linear_solver = dolfin.LUSolver()

        self._nonlinear_solver = dolfin.NonlinearVariationalSolver(dolfin \
            .NonlinearVariationalProblem(self._F, self._u, bcs, self._dFdu))

        ### Solver parameters

        self.parameters_linear_solver = self._linear_solver.parameters
        self.parameters_nonlinear_solver = self._nonlinear_solver.parameters
        self.parameters_inverse_solver = config.parameters_inverse_solver.copy()

        self.set_parameters_linear_solver(config.parameters_linear_solver)
        self.set_parameters_nonlinear_solver(config.parameters_nonlinear_solver)

        if parameters_inverse_solver is not None:
            self.set_parameters_inverse_solver(parameters_inverse_solver)

        ### Properties to be shared with a derived class

        # NOTE: Using a `dict` because this allows the values to be mutated.
        # For example, these values inside an instance of the derived class
        # `InverseSolver` can be updated and the changes will be reflected
        # in the object of the `InverseSolverBasic` class that was originally used
        # to construct the aforementioned object of the `InverseSolver` class.

        self._property = {
            'observation_time': None,
            'observation_times': None,
            'measurement_setter': None,
            'cumsum_DJDm': None,
            'cumsum_D2JDm2': None,
            'is_converged': False,
            'is_missing_z': True,
            'is_missing_dudm': True,
            'is_missing_d2udm2': True,
            'is_missing_dFdu_mtx': True,
            }

        self.assign_observation_times(observation_times)
        self.assign_measurement_setter(measurement_setter)


    def _partial_derivatives_of_form(self, f):
        '''Partial derivatives of form.

        Parameters
        ----------
        f : ufl.Form
            A form that is differentiable with respect to the primary field
            `self._u` (`dolfin.Function`) and model parameters `self._m`
            (`tuple` of `dolfin.Constant`s).

        Notes
        -----
        The variation form could have some zero partial derivatives. In the
        FEniCS implementation, zero-value partial derivatives obtained via the
        `diff` function can result in the form being empty. Since an empty form
        can not be assembled, the solution is to set empty `ufl.Form`s to zero.

        '''

        dfdu    = derivative(f, self._u)
        d2fdu2  = derivative(dfdu, self._u)
        dfdm    = list(diff(f, m_i) for m_i in self._m)
        d2fdudm = tuple(diff(dfdu, m_i) for m_i in self._m)
        d2fdm2  = list(list(diff(dfdm_i, m_j) for m_j in self._m[i:])
            for i, dfdm_i in enumerate(dfdm)) # only upper triangular part

        dx = dolfin.dx(self._V.mesh())
        ufl_zero_dx = ufl_zero()*dx

        for i, dfdm_i in enumerate(dfdm):
            if dfdm_i.empty():
                dfdm[i] = ufl_zero_dx

        dfdm = tuple(dfdm)

        for d2fdm2_i in d2fdm2:
            for j, d2fdm2_ij in enumerate(d2fdm2_i):
                if d2fdm2_ij.empty():
                    d2fdm2_i[j] = ufl_zero_dx

        d2fdm2 = tuple(tuple(d2fdm2_i) for d2fdm2_i in d2fdm2)

        return dfdu, dfdm, d2fdu2, d2fdudm, d2fdm2


    def _reset_inverse_solution(self):
        self._property['cumsum_DJDm'] = None
        self._property['cumsum_D2JDm2'] = None
        self._property['is_converged'] = False


    def _reset_nonlinear_solution(self):
        self._property['is_missing_z'] = True
        self._property['is_missing_dudm'] = True
        self._property['is_missing_d2udm2'] = True
        self._property['is_missing_dFdu_mtx'] = True


    def _set_nonlinear_solution_time(self, t):
        self._property['measurement_setter'](t)
        self._property['observation_time'] = t


    def _reset_measurement_time(self):
        '''Reset the measurement time to the current time. If the current time
        is `None` then invalidate/null the current nonlinear solution instead.

        Notes
        -----
        This function is intended to be called for resetting the measurements
        at the current time in case the measurement time has been overriden.

        '''

        if self._property['observation_time'] is not None:
            self._override_measurement_time(self._property['observation_time'])
        else:
            # Trying to set measurements for time `None` is too risky.
            # It is safer to invalidate the current solution instead.
            self._reset_nonlinear_solution()


    def _override_measurement_time(self, t):
        '''Try to set measurements at time `t` without changing `observation_time`.
        If an exception is raised, try to reset measurements at `observation_time`.
        If this also fails then invalidate the current nonlinear solution.

        Notes
        -----
        This function is intended to be used for the evaluation of measurements
        at arbitrary observation times without changing the nonlinear solution.
        This means the measurements must finally be reset at the original time.

        '''

        try:
            self._property['measurement_setter'](t)
        except:
            # Could not set measurements at time `t`.
            # Try to reset the previous measurements.
            if self._property['observation_time'] == t:
                # No point in attempting to reset time.
                # Just invalidate the current soluton.
                self._reset_nonlinear_solution()
            else:
                try:
                    self._property['measurement_setter'](
                        self._property['observation_time'])
                except:
                    # Since the measurement could not be reset,
                    # invalidate the current nonlinear solution.
                    self._reset_nonlinear_solution()

            raise ValueError('Could not override measurement '
                             f'time for argument `t = {t}`.')


    def _std_observation_times(self, observation_times):
        '''Return `observation_times` as a tuple of numerical values.'''

        if not isinstance(observation_times, tuple):
            try: observation_times = tuple(observation_times)
            except: observation_times = (observation_times,)

        if not all(isinstance(t, NUMERIC_TYPES) for t in observation_times):
            for numeric_type in NUMERIC_TYPES:

                tmp = tuple(numeric_type(t) for t in observation_times)

                if tmp == observation_times:
                    observation_times = tmp
                    break

            else:
                raise TypeError('Parameter `observation_times` must be '
                                'a (sequence of) numerical value(s).')

        return observation_times


    def _assemble_dFdu_and_rhs_sym(self, rhs_form):
        '''Assemble tangent system. Assembled `dFdu` will be symmetric.'''

        if self._property['is_missing_dFdu_mtx']:
            _, rhs = assemble_system(self._dFdu, rhs_form,
                self._bcs_zro, A_tensor=self._dFdu_mtx)
            self._property['is_missing_dFdu_mtx'] = False
        else:
            rhs = assemble(rhs_form)
            rhs[self._bcs_dof] = 0.0

        return self._dFdu_mtx, rhs


    def _assemble_dFdu_and_rhs(self, rhs_form):
        '''Assemble tangent system. Assembled `dFdu` will be asymmetric.'''

        if self._property['is_missing_dFdu_mtx']:
            assemble(self._dFdu, tensor=self._dFdu_mtx)
            self._property['is_missing_dFdu_mtx'] = False
            for bc in self._bcs_zro: bc.apply(self._dFdu_mtx)

        rhs = assemble(rhs_form)
        rhs[self._bcs_dof] = 0.0

        return self._dFdu_mtx, rhs


    def _compute_z(self):
        '''Solve the adjoint problem for the adjoint variable `z` that is
        required for computing cost sensitivities wrt model parameters.

        Notes
        -----
        Since the variational form `self._F` was derived from the potential
        energy functional `Pi`, the adjoint of the tangent stiffness is itself.
        In other words, `adjoint(self._dFdu)` is equivalent to `self._dFdu`.

        Warning
        -------
        There seems to be a problem assembling the bilinear form `dFdu` and the
        linear form `dJdu` using the function `dolfin.assemble_system` when the
        integration subdomains in `dFdu` and `dJdu` are different. In this case,
        FEniCS throws a warning and goes on to assume the integration domain for
        `dJdu` to be the same as that of `dFdu`, which is generally wrong. The
        way around this problem is to assemble `dFdu` and `dJdu` separately by
        calling the `dolfin.assemble` function. Unfortunately, the assembled
        `dFdu` becomes unsymmetric after imposition of the boundary conditions.
        The unsymmetric matrix could be a problem for certain types of solvers.

        '''

        lhs, rhs = self._assemble_dFdu_and_rhs(-self._dJdu)
        lhs = dolfin.PETScMatrix(lhs.mat().copy().transpose())
        self._linear_solver.solve(lhs, self._z.vector(), rhs)

        self._z.vector()[self._bcs_dof] = 0.0
        self._property['is_missing_z'] = False


    def _compute_dudv(self, dFdv, dudv):
        '''Compute primary field derivatives.'''

        lhs, rhs = self._assemble_dFdu_and_rhs_sym(-dFdv[0])
        self._linear_solver.solve(lhs, dudv[0].vector(), rhs)

        for dudv_i, dFdv_i in zip(dudv[1:], dFdv[1:]):
            rhs = assemble(-dFdv_i); rhs[self._bcs_dof] = 0.0
            self._linear_solver.solve(lhs, dudv_i.vector(), rhs)


    def _compute_dudm(self):
        '''Compute primary field sensitivities.'''
        self._compute_dudv(self._dFdm, self._dudm)
        self._property['is_missing_dudm'] = False


    def _compute_d2udm2(self):
        '''Compute second order primary field sensitivities. Note, since
        `d2udm2` is symmetric, only the upper triangular part is stored.
        '''

        if self._property['is_missing_dudm'] or \
           self._property['is_missing_dFdu_mtx']:
            self._compute_dudm()

        for i, dudm_i in enumerate(self._dudm, start=0):
            for j, dudm_j in enumerate(self._dudm[i:], start=i):

                if self._is_actionable_d2Fdu2:
                    action_action_d2Fdu2_dudm_j_dudm_i = \
                        action(action(self._d2Fdu2, dudm_j), dudm_i)
                else:
                    action_action_d2Fdu2_dudm_j_dudm_i = 0

                rhs = - (
                    assemble(self._d2Fdm2[i][j-i]
                        + action_action_d2Fdu2_dudm_j_dudm_i)
                    + assemble(self._d2Fdudm[j])*(dudm_i.vector())
                    + assemble(self._d2Fdudm[i])*(dudm_j.vector()))

                rhs[self._bcs_dof] = 0.0

                self._linear_solver.solve(
                    self._dFdu_mtx, self._d2udm2[i][j-i].vector(), rhs)

        self._property['is_missing_d2udm2'] = False


    def _factory_compute_dudv(self, dFdv):
        '''Factory for computing the first order derivatives
        of the primary field field with respect to variables.
        '''

        dudv = tuple(Function(self._V) for _ in dFdv)

        def compute_dudv():
            '''Compute the primary field derivatives.'''
            self._compute_dudv(dFdv, dudv)

        return compute_dudv, dudv


    def _factory_compute_dudv_d2udmdv(self, dFdv):
        '''Factory for computing first and second order mixed derivatives of
        the primary field with respect to the model parameters and variables.
        '''

        dudv = tuple(Function(self._V) for _ in dFdv)
        d2udmdv = tuple(tuple(Function(self._V)
            for j in range(self._n)) for _ in dFdv)

        d2Fdudv = tuple(derivative(dFdv_i, self._u) for dFdv_i in dFdv)
        d2Fdmdv = tuple(tuple(diff(dFdv_i, m_j) for m_j in self._m)
            for dFdv_i in dFdv)

        is_actionable_d2Fdu2 = self._is_actionable_d2Fdu2

        def compute_dudv_d2udmdv():
            '''Compute the first and second order mixed derivatives of the
            primary field with respect to the model parameters and variables.
            '''

            self._compute_dudv(dFdv, dudv)

            if self._property['is_missing_dudm']:
                self._compute_dudm()

            for i, dudv_i in enumerate(dudv):
                for j, dudm_j in enumerate(self._dudm):

                    if is_actionable_d2Fdu2:
                        action_action_d2Fdu2_dudv_i_dudm_j = \
                            action(action(self._d2Fdu2, dudv_i), dudm_j)
                    else:
                        action_action_d2Fdu2_dudv_i_dudm_j = 0

                    rhs = - (
                        assemble(d2Fdmdv[i][j]
                            + action_action_d2Fdu2_dudv_i_dudm_j)
                        + assemble(self._d2Fdudm[j])*(dudv_i.vector())
                        + assemble(d2Fdudv[i])*(dudm_j.vector()))

                    rhs[self._bcs_dof] = 0.0

                    self._linear_solver.solve(
                        self._dFdu_mtx, d2udmdv[i][j].vector(), rhs)

        return compute_dudv_d2udmdv, dudv, d2udmdv


    def _compute_DJDm_method_adjoint(self):
        '''Compute the derivatives of the model cost `J` with respect to
        model parameters `m` using the adjoint method: `dJ/dm + dF/dm dJ/dF`.

        Notes
        -----
        The adjoint method requires one solve for any number of model parameters:
            solve(adjoint(self._dFdu) == -self._dJdu, self._z, bcs=self._bcs_zro)

        '''

        DJDm = np.zeros((self._n,), float)

        if self._property['is_missing_z']:
            self._compute_z()

        for i, (dJdm_i, dFdm_i) in enumerate(zip(self._dJdm, self._dFdm)):
            DJDm[i] += assemble(dJdm_i) + assemble(dFdm_i).inner(self._z.vector())

        return DJDm


    def _compute_DJDm_method_direct(self):
        '''Compute the derivatives of the model cost `J` with respect to
        model parameters `m` using a direct method: `dJ/dm + du/dm dJ/du`.

        Notes
        -----
        The direct method requires as many solves as there model parameters:
            solve(self._dFdu == -self._dFdm[i], self._dudm[i], bcs=self._bcs_zro)

        '''

        DJDm = np.zeros((self._n,), float)
        assembled_dJdu = assemble(self._dJdu)

        if self._property['is_missing_dudm']:
            self._compute_dudm()

        for i, (dJdm_i, dudm_i) in enumerate(zip(self._dJdm, self._dudm)):
            DJDm[i] += assemble(dJdm_i) + assembled_dJdu.inner(dudm_i.vector())

        return DJDm


    def _compute_DJDm_D2JDm2_method_adjoint(self):
        '''Compute first order and second order derivatives of cost with respect
        to model parameters. This computation relies on the adjoint method.'''

        DJDm = np.zeros((self._n,), float)
        D2JDm2 = np.zeros((self._n, self._n), float)

        assembled_dJdu = assemble(self._dJdu)
        assembled_d2Jdu2 = assemble(self._d2Jdu2)

        if self._property['is_missing_dudm']:
            self._compute_dudm()

        if self._property['is_missing_z']:
            self._compute_z()

        is_actionable_d2Fdu2 = self._is_actionable_d2Fdu2

        for i, (d2Jdudm_i, d2Fdudm_i, dJdm_i, dFdm_i, dudm_i) in enumerate(
          zip(self._d2Jdudm, self._d2Fdudm, self._dJdm, self._dFdm, self._dudm)):

            # DJDm[i] = assemble(dJdm_i + action(self._dJdu, dudm_i)) # Can fail
            DJDm[i] = assemble(dJdm_i) + assembled_dJdu.inner(dudm_i.vector())

            for j, (d2Jdm2_ij, d2Fdm2_ij, d2Jdudm_j, d2Fdudm_j, dudm_j) in \
              enumerate(zip(self._d2Jdm2[i], self._d2Fdm2[i], self._d2Jdudm[i:],
              self._d2Fdudm[i:], self._dudm[i:]), start=i):

                D2JDm2[i,j] = (
                    assemble(d2Jdm2_ij)
                    + (assembled_d2Jdu2*dudm_j.vector()).inner(dudm_i.vector())
                    + assemble(d2Jdudm_j).inner(dudm_i.vector())
                    + assemble(d2Jdudm_i).inner(dudm_j.vector())
                    )

                if is_actionable_d2Fdu2:
                    action_action_d2Fdu2_dudm_j_dudm_i = \
                        action(action(self._d2Fdu2, dudm_j), dudm_i)
                else:
                    action_action_d2Fdu2_dudm_j_dudm_i = 0

                D2JDm2[i,j] += (
                    assemble(d2Fdm2_ij
                        + action_action_d2Fdu2_dudm_j_dudm_i)
                    + assemble(d2Fdudm_j)*(dudm_i.vector())
                    + assemble(d2Fdudm_i)*(dudm_j.vector())
                    ).inner(self._z.vector())

            for j in range(i+1, self._n):
                D2JDm2[j,i] = D2JDm2[i,j]

        return DJDm, D2JDm2


    def _compute_DJDm_D2JDm2_method_direct(self):
        '''Compute first order and second order derivatives of cost with respect
        to model parameters. This computation relies on the direct method.'''

        DJDm = np.zeros((self._n,), float)
        D2JDm2 = np.zeros((self._n, self._n), float)

        assembled_dJdu = assemble(self._dJdu)
        assembled_d2Jdu2 = assemble(self._d2Jdu2)

        if self._property['is_missing_dudm']:
            self._compute_dudm()

        if self._property['is_missing_d2udm2']:
            self._compute_d2udm2()

        for i, (d2Jdudm_i, dJdm_i, dudm_i) in enumerate(
          zip(self._d2Jdudm, self._dJdm, self._dudm)):

            # DJDm[i] = assemble(dJdm_i + action(self._dJdu, dudm_i)) # Can fail
            DJDm[i] = assemble(dJdm_i) + assembled_dJdu.inner(dudm_i.vector())

            for j, (d2Jdm2_ij, d2udm2_ij, d2Jdudm_j, dudm_j) in enumerate(
              zip(self._d2Jdm2[i], self._d2udm2[i], self._d2Jdudm[i:], self._dudm[i:]),
              start=i):

                D2JDm2[i,j] = (
                    assemble(d2Jdm2_ij)
                    + (assembled_d2Jdu2*dudm_j.vector()).inner(dudm_i.vector())
                    + assemble(d2Jdudm_j).inner(dudm_i.vector())
                    + assemble(d2Jdudm_i).inner(dudm_j.vector()))

                D2JDm2[i,j] += assembled_dJdu.inner(d2udm2_ij.vector())

            for j in range(i+1, self._n):
                D2JDm2[j,i] = D2JDm2[i,j]

        return DJDm, D2JDm2


    def _factory_compute_DJDm(self, sensitivity_method):
        '''Factory for computing model cost sensitivities.'''

        if sensitivity_method == self.SENSITIVITY_METHOD_ADJOINT:
            compute_DJDm = self._compute_DJDm_method_adjoint

        elif sensitivity_method == self.SENSITIVITY_METHOD_DIRECT:
            compute_DJDm = self._compute_DJDm_method_direct

        else:
            raise ValueError('Invalid sensitivity method '
                             f'\"{sensitivity_method:s}\"')

        return compute_DJDm


    def _factory_compute_DJDm_D2JDm2(self, sensitivity_method):
        '''Factory for computing model cost sensitivities.'''

        if sensitivity_method == self.SENSITIVITY_METHOD_ADJOINT:
            return self._compute_DJDm_D2JDm2_method_adjoint

        elif sensitivity_method == self.SENSITIVITY_METHOD_DIRECT:
            return self._compute_DJDm_D2JDm2_method_direct

        else:
            raise ValueError('Invalid sensitivity method '
                             f'\"{sensitivity_method:s}\"')


    def _factory_compute_dm(self):
        '''Factory for computing the change in model parameters.'''

        solver_method = self.parameters_inverse_solver['solver_method']

        if solver_method == self.INVERSE_SOLVER_METHOD_NEWTON:

            def compute_dm(DJDm, D2JDm2):
                try:
                    dm = self._compute_dm_method_newton(DJDm, D2JDm2)
                except linalg.LinAlgError:
                    logger.warning('Could not compute `dm` using '
                        'Newton method; trying gradient descent.')
                    dm = self._compute_dm_method_gradient(DJDm, D2JDm2)

                return dm

        elif solver_method == self.INVERSE_SOLVER_METHOD_GRADIENT:
            compute_dm = self._compute_dm_method_gradient

        else:
            raise ValueError(f'Invalid solver method \"{solver_method:s}\"')

        return compute_dm


    def _factory_constrain_dm(self):
        '''Factory for constraining the change in the model parameters.'''

        absval_min = self.parameters_inverse_solver['absolute_tolerance']
        rdelta_max = self.parameters_inverse_solver['maximum_relative_change']

        if rdelta_max is not None:
            if not (0.0 < rdelta_max < np.inf):
                raise ValueError('Require positive `maximum_relative_change`')

            def constrain_dm(dm, m):
                '''Constrain the maximum change in the model parameters.'''

                # rdelta = linalg.norm(dm) / linalg.norm(m)
                # if rdelta > rdelta_max: dm *= rdelta_max/rdelta

                absval = np.abs(m)
                rdelta = np.abs(dm) / (absval + absval_min)
                mask = (rdelta > rdelta_max) & (absval > absval_min)

                if mask.any():
                    ind = np.flatnonzero(mask)
                    dm[ind] *= rdelta_max / rdelta[ind]

                return dm

        else:
            def constrain_dm(dm, m):
                '''Do not constrain the change.'''
                return dm

        return constrain_dm


    def assign_model_parameters(self, model_parameters, value_types=NUMERIC_TYPES):
        '''Assign model parameters from a (nested) iterable. `model_parameters`
        is effectively flattened and the values assigned from left to right.'''

        self._reset_inverse_solution()
        self._reset_nonlinear_solution()

        if isinstance(model_parameters, np.ndarray):
            model_parameters = model_parameters.tolist()

        m = utility.list_values_from_iterable(model_parameters, value_types)

        if self._n != len(m):
            raise TypeError('Expected `model_parameters` to contain exactly '
                f'{self._n} model parameter(s) but instead found {len(m)}.')

        for self_m_i, m_i in zip(self._m, m):
            self_m_i.assign(float(m_i))


    def assign_observation_times(self, observation_times):
        '''Assign observation times and reset inverse solution.'''

        self._reset_inverse_solution()

        if observation_times is None:
            self._property['observation_times'] = (None,)
        else:
            self._property['observation_times'] = \
                self._std_observation_times(observation_times)


    def assign_measurement_setter(self, measurement_setter):
        '''Assign measurement setter and reset inverse solution.

        Parameters
        ----------
        measurement_setter : function-like that takes time as the argument
            Function that sets all measurements at a given observation time.

        '''

        self._reset_inverse_solution()

        if measurement_setter is None:
            measurement_setter = lambda t: None

        elif not callable(measurement_setter):
            raise TypeError('Parameter `measurement_setter` must be '
                            'a callable with time as the argument.')

        self._property['measurement_setter'] = measurement_setter


    def solve_nonlinear_problem(self, t=None):
        '''Solve the nonlinear problem for time `t`.'''

        self._reset_nonlinear_solution()

        if t is not None:
            self._set_nonlinear_solution_time(t)

        n, b = self._nonlinear_solver.solve()

        if not b:
            logger.error(f'Nonlinear solver failed to converge '
                         f'for time t={t} after n={n} iterations')

        return n, b


    def solve_forward_problem(self):
        '''Solve nonlinear problem at each observation time and add up the model
        cost sensitivities, which are to be used in updating model parameters.

        Returns
        -------
        cumsum_DJDm : numpy.ndarray (1D) of float's
            Cumulative gradient of the model cost over the observation times.
        cumsum_D2JDm2 : numpy.ndarray (2D) of float's
            Cumulative hessian of the model cost over the observation times.

        '''

        cumsum_DJDm = np.zeros((self._n,), float)
        cumsum_D2JDm2 = np.zeros((self._n, self._n), float)

        compute_DJDm_D2JDm2 = self._factory_compute_DJDm_D2JDm2(
            self.parameters_inverse_solver['sensitivity_method'])

        for t in self.observation_times:

            self.solve_nonlinear_problem(t)
            DJDm, D2JDm2 = compute_DJDm_D2JDm2()

            if not np.isfinite(D2JDm2).all():
                raise RuntimeError('Model cost sensitivities at time '
                                   f'`t = {t}` have non-finite values.')

            cumsum_DJDm += DJDm
            cumsum_D2JDm2 += D2JDm2

        self._property['cumsum_DJDm'] = cumsum_DJDm.copy()
        self._property['cumsum_D2JDm2'] = cumsum_D2JDm2.copy()

        return cumsum_DJDm, cumsum_D2JDm2


    def solve_inverse_problem(self, atol=None, rtol=None):
        '''Minimize model cost with respect to model parameters.

        Returns
        -------
        num_iterations : int
            Number of iterations.
        is_converged : bool
            Convergence status.

        '''

        self._property['is_converged'] = False

        logging_level = logger.getEffectiveLevel()
        parameters = self.parameters_inverse_solver

        max_iterations = parameters['maximum_iterations']
        max_divergences = parameters['maximum_divergences']

        if atol is None: atol = parameters['absolute_tolerance']
        if rtol is None: rtol = parameters['relative_tolerance']

        dm_old = None
        DJDm_old = None
        D2JDm2_old = None
        norm_DJDm_old = np.inf

        num_divergences = 0
        is_converged = False

        DJDm, D2JDm2 = self.solve_forward_problem()
        m = np.array(self.view_model_parameter_values())

        compute_dm = self._factory_compute_dm()
        constrain_dm = self._factory_constrain_dm()

        for num_iterations in range(1, max_iterations+1):

            if logging_level <= logging.INFO:
                print(f'\n*** Iteration #{num_iterations:d} ***\n')

            dm = compute_dm(DJDm, D2JDm2)
            dm = constrain_dm(dm, m)

            norm_dm = sqrt(dm.dot(dm))
            norm_DJDm = sqrt(DJDm.dot(DJDm))

            dJ = DJDm.dot(dm)
            d2J = D2JDm2.dot(dm).dot(dm)

            DJDm_dm = dJ/norm_dm
            D2JDm2_dm = d2J/norm_dm**2

            is_decreasing = DJDm_dm < 0.0
            is_pathconvex = D2JDm2_dm > 0.0
            is_converging = norm_DJDm < norm_DJDm_old

            if logging_level <= logging.INFO:
                print('\n'
                    f'  norm(DJDm_old)     :{norm_DJDm_old: g}\n'
                    f'  norm(DJDm)         :{norm_DJDm: g}\n\n'
                    f'  DJDm[dm]           :{DJDm_dm: g}\n'
                    f'  D2JDm2[dm]         :{D2JDm2_dm: g}\n\n'
                    f'  model param.,  m   : {m}\n'
                    f'  delta param., dm   : {dm}\n\n'
                    f'  is path convex     : {is_pathconvex}\n'
                    f'  is cost decreasing : {is_decreasing}\n'
                    f'  is cost converging : {is_converging}\n'
                    , flush=True)

            if logging_level <= logging.DEBUG and dm_old is not None:

                # Check if the directional second derivative of `J` can estimate the
                # change in gradient of `J` between previous and current iterations.

                dDJDm_exact = DJDm - DJDm_old
                dDJDm_estim = (D2JDm2_old + D2JDm2).dot(dm_old) * 0.5
                # err_dDJDm = linalg.norm(dDJDm_estim-dDJDm_exact)/linalg.norm(dDJDm_exact)

                print(f'  {"estimated change in DJDm, i.e. D2JDm2[dm_old]":s} : {dDJDm_estim}\n'
                      f'  {"actual change in DJDm, i.e. DJDm_new-DJDm_old":s} : {dDJDm_exact}\n'
                      , flush=True)

            if np.all(np.abs(dm) < atol):
                logger.info('Iterations converged in absolute tolerance')
                is_converged = True
                break

            if np.all(np.abs(dm) < np.abs(m)*rtol):
                logger.info('Iterations converged in relative tolerance')
                is_converged = True
                break

            if norm_DJDm_old < norm_DJDm:
                num_divergences += 1

                logger.warning('Model cost diverged '
                    f'({num_divergences}/{max_divergences})')

                if num_divergences > max_divergences:
                    logger.error('Model cost diverged maximum number of times')

                    m -= dm_old

                    for m_i, m_i_new in zip(self._m, m):
                        m_i.assign(m_i_new)

                    self.solve_nonlinear_problem()

                    self._property['cumsum_DJDm'] = DJDm_old
                    self._property['cumsum_D2JDm2'] = D2JDm2_old

                    break

            m += dm

            for m_i, m_i_new in zip(self._m, m):
                m_i.assign(m_i_new)

            dm_old = dm
            DJDm_old = DJDm
            D2JDm2_old = D2JDm2
            norm_DJDm_old = norm_DJDm

            DJDm, D2JDm2 = self.solve_forward_problem()

        if not is_converged:
            logger.error('Inverse solver did not converge after '
                         f'{num_iterations} iterations.')

        self._property['is_converged'] = is_converged

        return num_iterations, is_converged


    def require_nonlinear_solution(self, t=None):
        '''Check if the nonlinear problem needs to be solved for time `t`.'''
        return (t is not None and t != self._property['observation_time']) \
               or self._property['observation_time'] is None


    def observe_J(self, t=None):
        '''Model cost at time `t`.'''

        if self.require_nonlinear_solution(t):
            self.solve_nonlinear_problem(t)

        return assemble(self._J)


    def observe_DJDm(self, t=None):
        '''Model cost gradient at time `t`.'''

        if self.require_nonlinear_solution(t):
            self.solve_nonlinear_problem(t)

        compute_DJDm = self._factory_compute_DJDm(
            self.parameters_inverse_solver['sensitivity_method'])

        return compute_DJDm()


    def observe_u(self, t=None, copy=True):
        '''Primary field solution at time `t`.'''

        if self.require_nonlinear_solution(t):
            self.solve_nonlinear_problem(t)

        return self._u.copy(True) if copy else self._u


    def observe_dudm(self, t=None, copy=True):
        '''Partial derivatives of the primary field at time `t`.'''

        if self.require_nonlinear_solution(t):
            self.solve_nonlinear_problem(t)

        if self._property['is_missing_dudm']:
            self._compute_dudm()

        return [dudm_i.copy(True) for dudm_i in self._dudm] \
               if copy else self._dudm


    def observe_d2udm2(self, t=None, copy=True):
        '''Compute partial derivatives of the primary field with respect to
        the model parameters at observation time `t`. Note, since the matrix
        d2u/dm2 is symmetric, only the upper triangular part is returned.'''

        if self.require_nonlinear_solution(t):
            self.solve_nonlinear_problem(t)

        if self._property['is_missing_d2udm2']:
            self._compute_d2udm2()

        if copy:
            return [[dudm_ij.copy(True)
                    for dudm_ij in dudm_i]
                    for dudm_i in self._d2udm2]
        else:
            return self._d2udm2


    def set_parameters_inverse_solver(self, rhs):
        '''Update parameter values with those of a nested dict-like `rhs`.'''
        utility.update_existing_keyvalues(self.parameters_inverse_solver, rhs)

    def set_parameters_nonlinear_solver(self, rhs):
        '''Update parameter values with those of a nested dict-like `rhs`.'''
        utility.update_existing_keyvalues(self.parameters_nonlinear_solver, rhs)

    def set_parameters_linear_solver(self, rhs):
        '''Update parameter values with those of a nested dict-like `rhs`.'''
        utility.update_existing_keyvalues(self.parameters_linear_solver, rhs)


    def view_cumsum_DJDm(self):
        if self._property['cumsum_DJDm'] is not None:
            return self._property['cumsum_DJDm'].copy()
        else:
            logger.info('Solving forward problem')
            return self.solve_forward_problem()[0]


    def view_cumsum_D2JDm2(self):
        if self._property['cumsum_D2JDm2'] is not None:
            return self._property['cumsum_D2JDm2'].copy()
        else:
            logger.info('Solving forward problem')
            return self.solve_forward_problem()[1]


    def view_model_parameters(self):
        '''Return a view of the `model_parameters` where `dolfin.Constant`s are
        replaced with their `float` values. `model_parameters` can be nested.'''

        def extract(model_parameters):

            if hasattr(model_parameters, 'keys'):
                return {k : extract(model_parameters[k])
                    for k in model_parameters.keys()}

            elif isinstance(model_parameters, SEQUENCE_TYPES):
                return [extract(m) for m in model_parameters]

            else:
                if isinstance(model_parameters, Constant):
                    return float(model_parameters.values())
                else:
                    return model_parameters

        return extract(self._model_parameters)


    def view_model_parameter_values(self):
        return tuple(float(m_i) for m_i in self._m)

    @property
    def m(self):
        '''Tuple of model parameters.'''
        return self._m

    @property
    def n(self):
        '''Number of model parameters.'''
        return self._n

    @property
    def model_parameters(self):
        return utility.replicate_tree_structure(
            self._model_parameters, Constant)

    @property
    def model_parameters_listed(self):
        return list(self._m)

    @property
    def num_model_parameters(self):
        return self._n

    @property
    def observation_time(self):
        return self._property['observation_time']

    @property
    def observation_times(self):
        return self._property['observation_times']

    @property
    def num_observation_times(self):
        return len(self._property['observation_times'])

    @property
    def measurement_setter(self):
        return self._property['measurement_setter']

    @property
    def is_converged(self):
        return self._property['is_converged']


    @staticmethod
    def _compute_dm_method_gradient(DJDm, D2JDm2):
        '''Compute model parameter change using gradient-descent with line-search.'''
        dm = -DJDm
        d2J = D2JDm2.dot(dm).dot(dm)
        if d2J > 0:
            dm *= (dm.dot(dm)/d2J)
        return dm

    @staticmethod
    def _compute_dm_method_newton(DJDm, D2JDm2):
        '''Compute model parameter change using the Newton method.'''
        return linalg.solve(D2JDm2, -DJDm, assume_a='sym')


class InverseSolver(InverseSolverBasic):

    def __new__(cls, inverse_solver_basic, *args):

        if not isinstance(inverse_solver_basic, InverseSolverBasic):
            raise TypeError('Parameter `inverse_solver_basic` must '
                            'be an instance of `InverseSolverBasic`.')

        # Create an empty instance
        self = object.__new__(cls)

        # Reference the fields from `inverse_solver_basic`
        self.__dict__.update(inverse_solver_basic.__dict__)

        return self


    def __init__(self, inverse_solver_basic,
        u_obs, u_msr, dx_msr, T_obs, T_msr, ds_msr):
        '''Initialize inverse solver.

        Parameters
        ----------
        inverse_solver_basic : InverseSolverBasic
            Basic inverse solver.

        u_obs : dolfin.Function
            The displacement field function. Note, in case the primary field
            `u` is mixed, `u_obs` can be extracted by the splitting method;
            specifically, `u_obs, *_ = u.split(deepcopy=False)`. By ensuring
            that `deepcopy==False`, the degrees of freedom will be shared.

        u_msr : dolfin expression-like object
            An expression of the displacement field measurements.

        dx_msr : dolfin.Measure
            Integration domain measure for the measured displacement field.

        T_obs : a single or a sequence of dolfin expression-like objects
            An expression of the tractions observed on the measurement boundary.

        T_msr : a single or a sequence of dolfin expression-like objects
            An expression of the tractions measured on the measurement boundary.

        ds_msr : dolfin.Measure
            Integration boundary measure for the measured/observed tractions.

        '''

        self._u_obs, self._u_msr, self._dx_msr = \
            self._std_init_args_u_obs_u_msr_dx_msr(u_obs, u_msr, dx_msr)

        self._T_obs, self._T_msr, self._ds_msr = \
            self._std_init_args_T_obs_T_msr_ds_msr(T_obs, T_msr, ds_msr)

        self._int_dx_msr = tuple(assemble(1*dx_i) for dx_i in self._dx_msr)
        self._int_ds_msr = tuple(assemble(1*sum(ds_i[1:], ds_i[0]))
                                 for ds_i in self._ds_msr)

        self._f_obs = tuple(tuple(sum(T[j_dim]*ds
            for T, ds in zip(T_obs_i, ds_msr_i))
            for j_dim in range(len(T_obs_i[0])))
            for T_obs_i, ds_msr_i in zip(self._T_obs, self._ds_msr))

        self._f_msr = tuple(tuple(sum(T[j_dim]*ds
            for T, ds in zip(T_msr_i, ds_msr_i))
            for j_dim in range(len(T_msr_i[0])))
            for T_msr_i, ds_msr_i in zip(self._T_msr, self._ds_msr))

        assert all(len(f_obs_i) == len(f_msr_i)
            for f_obs_i, f_msr_i in zip(self._f_obs, self._f_msr))

        self._dfdm = tuple(tuple(
            tuple(diff(fj_obs_i, m_k) for m_k in self._m)
            for fj_obs_i in f_obs_i) for f_obs_i in self._f_obs)

        self._dfdu = tuple(tuple(derivative(fj_obs_i, self._u)
            for fj_obs_i in f_obs_i) for f_obs_i in self._f_obs)

        # Model parameter sensitivities wrt themselves
        self.observe_dmdm = self.apply_observation_caching(
                            self.factory_observe_dmdm())

        def raise_method_undefined_error(*args, **kwargs):
            raise RuntimeError('Method `observe_dmdu_msr` has not been defined; '
                               'define it by calling `init_observe_dmdu_msr`.')

        # Sensitivities wrt displacement field measurements
        self.observe_dmdu_msr = raise_method_undefined_error

        def raise_method_undefined_error(*args, **kwargs):
            raise RuntimeError('Method `observe_dmdT_msr` has not been defined; '
                               'define it by calling `init_observe_dmdT_msr`.')

        # Sensitivities wrt reaction (traction) measurements
        self.observe_dmdT_msr = raise_method_undefined_error

        self._du_msr_dummy = ()
        self._dT_msr_dummy = ()

        self._cached_u_dofs = {} # Cached displacements
        self._cached_u_auth = () # Cache authentication

        # self._is_sequence_u_msr = True if \
        #     isinstance(dx_msr, SEQUENCE_TYPES) else False

        # self._is_sequence_f_msr = True if \
        #     isinstance(ds_msr, SEQUENCE_TYPES) else False


    def assign_measurement_setter(self, measurement_setter):
        '''Assign measurement setter and reset inverse solution.'''

        super().assign_measurement_setter(measurement_setter)
        self._cached_u_dofs.clear(); self._cached_u_auth = ()


    def fit_model_forall_times(self, observation_times=None):
        '''Minimize model cost for all observation times.'''

        logger.info('Begin model fitting for all observation times')

        if observation_times is not None and \
           observation_times is not self.observation_times:
            self.assign_observation_times(observation_times)

        num_iterations, is_converged = self.solve_inverse_problem()
        model_parameters = self.view_model_parameter_values()

        if not is_converged and \
           self.parameters_inverse_solver['error_on_nonconvergence']:
            raise RuntimeError('Inverse solver did not converge')

        return model_parameters, num_iterations, is_converged


    def fit_model_foreach_time(self, observation_times=None):
        '''Minimize model cost for each observation time.'''

        logger.info('Begin model fitting for each observation time')

        if observation_times is None: observation_times = self.observation_times
        else: observation_times = self._std_observation_times(observation_times)

        is_converged = []
        num_iterations = []
        model_parameters = []

        try:
            for t in observation_times:

                self.assign_observation_times((t,))
                n, b = self.solve_inverse_problem()
                m = self.view_model_parameter_values()

                is_converged.append(b)
                num_iterations.append(n)
                model_parameters.append(m)

                if not b:
                    logger.error('Inverse solver did not converge '
                                 f'for observation time {t}')

                    if self.parameters_inverse_solver['error_on_nonconvergence']:
                        raise RuntimeError('Inverse solver did not converge')

        finally:
            self.assign_observation_times(observation_times)

        return model_parameters, num_iterations, is_converged


    def update_nonlinear_solution(self, t=None):
        '''Update the nonlinear solution for the current time. If the solution
        is not already in cache, solve the nonlinear problem and cache the
        solution for the current time and model parameter values.

        Returns
        -------
        n : int
            Number of iterations.
        b : bool
            Convergence status.

        '''

        if t is None:
            t = self.observation_time

            if t is None:
                return self.solve_nonlinear_problem(t)

        auth = self.view_model_parameter_values()

        if self._cached_u_auth != auth:
            self._cached_u_auth = auth
            self._cached_u_dofs.clear()

        u_dofs = self._cached_u_dofs.get(t)

        if u_dofs:

            self._reset_nonlinear_solution()
            self._set_nonlinear_solution_time(t)
            self._u.vector()[:] = u_dofs
            n, b = 0, True

        else:

            n, b = self.solve_nonlinear_problem(t)
            self._cached_u_dofs[t] = self._u.vector().copy()

        return n, b


    def assess_model_cost(self, observation_times=None, compute_gradients=True):
        '''Compute the model cost `J` and the model cost derivatives `DJDm`.

        Returns
        -------
        cost_values : list of float's
            Values of the model cost at observation times.
        cost_gradients : list of numpy.ndarray's (1D)
            Gradients of the model cost at observation times.

        '''

        if observation_times is None: observation_times = self.observation_times
        else: observation_times = self._std_observation_times(observation_times)

        cost_values = []
        cost_gradients = []

        if compute_gradients:
            compute_DJDm = self._factory_compute_DJDm(
                self.parameters_inverse_solver['sensitivity_method'])

        for t in observation_times:

            self.update_nonlinear_solution(t)
            cost_values.append(assemble(self._J))

            if compute_gradients:
                cost_gradients.append(compute_DJDm())

        return cost_values, cost_gradients


    def assess_cost_sensitivity(self, constraint_vectors=None):
        '''Compute the principal (eigenvalue-like) model cost sensitivities
        with respect to the model parameters taking into account any constraint
        vectors for the model parameter change direction.

        Notes
        -----
        The constraint vectors will typically be for the reaction forces. For
        example, suppose you wish to compute the sensitivities of the cost
        subject to the constness of an observed reaction force on a measurement
        boundary. The reaction force variation (i.e. a constraint vector) can
        be obtained useing `observe_dfdm(t)[i_msr][i_dim]` where `i_msr` denotes
        a particular measurement and `i_dim` -- the observed force component.

        Important
        ---------
        The number of constraint vectors must be fewer than the number of
        model parameters with respect to which the cost sensitivity will be
        computed.

        Returns
        -------
        d2J_eig : list of float's
            Model cost sensitivities with respect to the principal model
            parameter change directions `dm_eig`.
        dm_eig : list of numpy.ndarray's (1D)
            Principal model parameter change directions that are conjugate
            with respect to `D2JD2m` and orthogonal to any constraint vectors.
            Note that `dm_eig` are generally not mutually orthogonal.

        '''

        EPS = 1e-12

        D2JDm2 = self.view_cumsum_D2JDm2()

        if (hasattr(constraint_vectors, '__len__') \
            and len(constraint_vectors) != 0):
            C = constraint_vectors
        else:
            C = None

        if C is not None:

            C = self._std_model_parameter_constraints(C)
            R = self._compute_orthogonalizing_operator(C)

            # Principal directions of curvature of `D2JDm2`
            _, eigvec = linalg.eigh(R.T.dot(D2JDm2.dot(R)))
            dm_eig = eigvec.T.dot(R) # (as row vectors)

            # NOTE: `dm_eig` are conjugate wrt `D2JDm2`, but
            #       generally not orthogonal wrt each other.

            dm_eig = [v_i / (sqrt(v_i.dot(v_i)) + EPS) for v_i in dm_eig]
            d2J_eig = [D2JDm2.dot(v_i).dot(v_i) for v_i in dm_eig]

        else:

            d2J_eig, eigvec = linalg.eigh(D2JDm2)
            dm_eig = eigvec.T # (as row vectors)

        ind = np.argsort(np.abs(d2J_eig))[::-1]
        d2J_eig = [d2J_eig[i] for i in ind]
        dm_eig = [dm_eig[i] for i in ind]

        return d2J_eig, dm_eig


    def assess_misfit_displacements(self, observation_times=None, subdims=None):
        '''Compute the relative misfit in the displacements.

        Notes
        -----
        The measure of misfit is the L2-norm:
            $\sqrt{(u_{obs} - u_{msr})**2 dx_{msr} / u_{msr}**2 dx_{msr}}$

        Parameters
        ----------
        observation_times : (sequence of) real(s) (optional)
            Observation times at which to compute the displacement errors. If
            `None`, the errors are computed for the current observation times.
        subdims : (sequence of (sequences of)) int(s) (optional)
            The subdimension indices into the displacement field measurements.
            If `None`, the error is computed considering all subdimension.

        Returns
        -------
        errors : list of list's of 1D numpy.ndarray's
            The relative misfit between the observed and measured displacements.
            Value at `errors[I][J]` refers to `I`th measurement at `J`th time.

        '''

        EPS = 1e-12
        EPSEPS = EPS*EPS

        if observation_times is None: observation_times = self.observation_times
        else: observation_times = self._std_observation_times(observation_times)

        size_msr, dim_max = len(self._u_msr), len(self._u_msr[0])
        subdims = self._std_subdims_v_msr(subdims, size_msr, dim_max)

        numers = [sum((self._u_obs[dim_j]-u_msr_i[dim_j])**2*dx_msr_i for dim_j in subdims_msr_i)
                  for u_msr_i, dx_msr_i, subdims_msr_i in zip(self._u_msr, self._dx_msr, subdims)]

        denoms = [sum(u_msr_i[dim_j]**2*dx_msr_i for dim_j in subdims_msr_i)
                  for u_msr_i, dx_msr_i, subdims_msr_i in zip(self._u_msr, self._dx_msr, subdims)]

        errors = []

        def compute_error(numer_i, denom_i):
            return sqrt(assemble(numer_i)/(assemble(denom_i) + EPSEPS))

        for t in observation_times:
            self.update_nonlinear_solution(t)
            errors.append([compute_error(numer_i, denom_i)
                for numer_i, denom_i in zip(numers, denoms)])

        # Current indexing: observation time, measurement subdomain
        # Reorder indexing: measurement subdomain, observation time

        errors = [[error_t[i_msr] for error_t in errors]
                  for i_msr in range(size_msr)]

        return errors


    def assess_misfit_reaction_forces(self, observation_times=None, subdims=None):
        '''Compute the relative misfit in the reaction forces.

        Notes
        -----
        The measure of misfit is the L2-norm:
            $\sqrt{(f_{obs} - f_{msr})**2 / f_{msr}**2}$

        Parameters
        ----------
        observation_times : (sequence of) real(s) (optional)
            Observation times at which to compute the displacement errors. If
            `None`, the errors are computed for the current observation times.
        subdims : (sequence of (sequences of)) int(s) (optional)
            The subdimension indices into the displacement field measurements.
            If `None`, the error is computed considering all subdimension.

        Returns
        -------
        errors : list of list's of 1D numpy.ndarray's
            The relative misfit between the model and measured reaction forces.
            The value at `errors[I][J][K]` refers to the `I`th measurement, the
            `J`th time, and the `K`th reaction force component.

        '''

        EPS = 1e-12
        EPSEPS = EPS*EPS

        if observation_times is None: observation_times = self.observation_times
        else: observation_times = self._std_observation_times(observation_times)

        size_msr, dim_max = len(self._f_msr), len(self._f_msr[0])
        subdims = self._std_subdims_v_msr(subdims, size_msr, dim_max)

        f_obs = [[f_obs_i[j] for j in subdims_i]
            for f_obs_i, subdims_i in zip(self._f_obs, subdims)]

        f_msr = [[f_msr_i[j] for j in subdims_i]
            for f_msr_i, subdims_i in zip(self._f_msr, subdims)]

        errors = []

        def compute_error(f_obs_i, f_msr_i):

            f_obs_i = [assemble(fj) for fj in f_obs_i]
            f_msr_i = [assemble(fj) for fj in f_msr_i]

            denom_i = sum(fj**2 for fj in f_msr_i) + EPSEPS
            numer_i = sum((fj_obs_i - fj_msr_i)**2
                for fj_obs_i, fj_msr_i in zip(f_obs_i, f_msr_i))

            return sqrt(numer_i/denom_i)

        for t in observation_times:
            self.update_nonlinear_solution(t)
            errors.append([compute_error(f_obs_i, f_msr_i)
                for f_obs_i, f_msr_i in zip(f_obs, f_msr)])

        # Current indexing: time, measurement subdomain, force component
        # Reorder indexing: measurement subdomain, time, force component

        errors = [[error_t[i_msr] for error_t in errors]
                  for i_msr in range(size_msr)]

        return errors


    def apply_observation_caching(self, function):
        '''Observations produced by `function` will be cached.

        Parameters
        ----------
        function : callable(t)
            Function-like with a single argument for the observation time.

        '''

        cache = {}

        cache_auth_1 = self.view_model_parameter_values()
        cache_auth_2 = id(self.observation_times)
        cache_auth_3 = id(self.measurement_setter)

        def function_with_caching(t=None, copy=True):
            # To replace `__doc__` with `function.__doc__`

            nonlocal cache_auth_1
            nonlocal cache_auth_2
            nonlocal cache_auth_3

            if t is None:
                t = self.observation_time

                if t is None:
                    return function(t)

            auth_1 = self.view_model_parameter_values()
            auth_2 = id(self.observation_times)
            auth_3 = id(self.measurement_setter)

            if cache_auth_1 != auth_1 or \
               cache_auth_2 != auth_2 or \
               cache_auth_3 != auth_3:
                cache_auth_1 = auth_1
                cache_auth_2 = auth_2
                cache_auth_3 = auth_3
                cache.clear()

            value = cache.get(t)

            if value is None:

                if t != self.observation_time:
                    self.update_nonlinear_solution(t)

                value = function(t)
                cache[t] = value

            return deepcopy(value) if copy else value

        function_with_caching.__doc__ = function.__doc__
        function_with_caching.cache = cache

        return function_with_caching


    def init_observe_dmdu_msr(self, v, ignore_dFdv, ignore_dJdv=False):
        '''Define the method for computing the model parameter sensitivities
        with respect to the displacement field measurement perturbations.

        Parameters
        ----------
        v : (sequence of) dolfin.Function('s) or dolfin.Constant('s)
            Dummy arguments with respect to which to compute the sensitivities.

        '''

        if not isinstance(v, SEQUENCE_TYPES): v = (v,)

        if len(v) != len(self._u_msr):
            raise TypeError('Expected parameter `v` to contain '
                            f'{len(self._u_msr)} coefficient(s)')

        v = utility.list_values_from_iterable(v, (Function, Constant))

        self.observe_dmdu_msr = self.apply_observation_caching(
            self.factory_observe_dmdv(v, ignore_dFdv, ignore_dJdv))

        self._du_msr_dummy = tuple(v)

        return self.observe_dmdu_msr


    def init_observe_dmdT_msr(self, v, ignore_dFdv, ignore_dJdv=False):
        '''Define the method for computing the model parameter sensitivities
        with respect to the reaction (traction) measurement perturbations.

        Parameters
        ----------
        v : (sequence of) dolfin.Function('s) or dolfin.Constant('s)
            Dummy arguments with respect to which to compute the sensitivities.

        '''

        if not isinstance(v, SEQUENCE_TYPES): v = (v,)

        if len(v) != len(self._T_msr):
            raise TypeError('Expected parameter `v` to contain '
                            f'{len(self._T_msr)} coefficient(s).')

        v = utility.list_values_from_iterable(v, (Function, Constant))

        self.observe_dmdT_msr = self.apply_observation_caching(
            self.factory_observe_dmdv(v, ignore_dFdv, ignore_dJdv))

        self._dT_msr_dummy = tuple(v)

        return self.observe_dmdT_msr


    def observe_dmdf_msr(self, t=None):
        '''Compute model parameter sensitivities with respect to the reaction
        force measurements at observation time `t`.

        Returns
        -------
        dmdf_msr : list of numpy.ndarray's (2D)
            Model parameter sensitivities with respect to the reaction force
            measurements at observation time `t`. The value at `dmdf_msr[I][J,K]`
            corresponds to the `I`th measurement, and the `J`th model parameter
            sensitivity with respect to the `K`th measurement degree of freedom.

        '''
        return [dmdT_msr_i / ds_msr_i for dmdT_msr_i, ds_msr_i in \
            zip(self.observe_dmdT_msr(t, copy=False), self._int_ds_msr)]


    def factory_observe_dmdm(self):
        '''Factory for computing model parameter self-sensitivities.

        Notes
        -----
        This function is useful for testing the self-correction of the inverse
        solver. In general, given a perturbation in the model parameters the
        resultant cumulative model parameter sensitivities should exactly be in
        the opposite direction of the perturbation. Specifically, the cumulative
        sensitivities should be equal to the negative of the identity matrix.

        Returns
        -------
        observe_dmdm : function(t=None)
            The function that computes the model parameter self-sensitivities
            for the observation time `t`. Note, if `t` is `None`, assume the
            current solution time.

        '''

        _observe_dmdm = self.factory_observe_dmdv(
            self._m, ignore_dFdv=False, ignore_dJdv=False)

        # NOTE: The function `_observe_dmdm` returns a list of arrays.
        #       As this is not convenient, `observe_dmdm` modifies the
        #       function so that a square matrix is returned instead.

        def observe_dmdm(t=None):
            '''Compute model parameter self-sensitivities at time `t`.

            Returns
            -------
            dmdm : numpy.ndarray (2D)
                Model parameter self-sensitivities for time `t`.

            '''
            return np.concatenate(_observe_dmdm(t), 1)

        return observe_dmdm


    def factory_observe_dmdv(self, v, ignore_dFdv=False, ignore_dJdv=False):
        '''Return a function that computes the model parameter sensitivities.

        Parameters
        ----------
        v : (sequence of) either dolfin.Constant(s) or dolfin.Function(s)
            The arguments with respect to wich to compute the sensitivities.

        ignore_dFdv : bool
            Whether the argumens `v` are not in the weak form.

        ignore_dJdv : bool
            Whether the arguments `v` are not in the model cost.

        Returns
        -------
        observe_dmdv : function(t=None)
            The function that computes the sensitivities.

        Notes
        -----
        The sensitivities are local quantities. More precisely, they represent
        the sensitivities due to local changes in the variables at a given time.
        The total (or cumulative) sensitivities can be obtained by summing all
        local sensitivities over the observation times.

        '''

        if ignore_dFdv and ignore_dJdv:
            raise ValueError('Parameters `ignore_dFdv` and `ignore_dJdv` must '
                             'not both be `True`; at least one must be `False`.')

        not_ignore_dFdv = not ignore_dFdv; del ignore_dFdv
        not_ignore_dJdv = not ignore_dJdv; del ignore_dJdv

        if not isinstance(v, SEQUENCE_TYPES): v = (v,)

        is_type_constant = all(isinstance(v_i, Constant) for v_i in v)
        is_type_function = all(isinstance(v_i, Function) for v_i in v) \
                           if not is_type_constant else False

        if is_type_constant:

            if not_ignore_dFdv: dFdv = []
            if not_ignore_dJdv: dJdv = []

            slices = []
            posnxt = 0

            for v_i in v:

                shape = v_i.ufl_shape

                if len(shape) > 1:
                    raise TypeError('Parameter `v` can only contain `dolfin.Constant`s '
                                    'that are either scalar-valued or vector-valued.')

                dim = shape[0] if shape else 1
                poscur, posnxt = posnxt, posnxt+dim
                slices.append(slice(poscur, posnxt))

                if dim > 1:
                    for j in range(dim):

                        dv_i = [0.0]*dim
                        dv_i[j] = 1.0
                        dv_i = Constant(dv_i)

                        if not_ignore_dFdv: dFdv.append(derivative(self._F, v_i, dv_i))
                        if not_ignore_dJdv: dJdv.append(derivative(self._J, v_i, dv_i))

                else:
                    if not_ignore_dFdv: dFdv.append(diff(self._F, v_i))
                    if not_ignore_dJdv: dJdv.append(diff(self._J, v_i))

            if not_ignore_dFdv:
                compute_dudv_d2udmdv, dudv, d2udmdv = \
                    self._factory_compute_dudv_d2udmdv(dFdv)

            if not_ignore_dJdv:
                d2Jdudv = [derivative(dJdv_i, self._u) for dJdv_i in dJdv]
                d2Jdmdv = [[diff(dJdv_i, m_j) for m_j in self._m] for dJdv_i in dJdv]


            def compute_dDJDmdv(i, assembled_dJdu, assembled_d2Jdu2, assemble_d2Jdudm):

                slc = slices[i]; dim = slc.stop - slc.start
                dDJDmdv_i = np.zeros((self._n, dim), float)

                if not_ignore_dJdv:

                    for j, dudm_j in enumerate(self._dudm):
                        for k, (d2Jdudv_ik, d2Jdmdv_ik) in \
                          enumerate(zip(d2Jdudv[slc], d2Jdmdv[slc])):

                            dDJDmdv_i[j,k] += assemble(d2Jdmdv_ik[j]) + \
                                              assemble(d2Jdudv_ik).inner(dudm_j.vector())

                if not_ignore_dFdv:

                    for j, dudm_j in enumerate(self._dudm):
                        for k, (dudv_ik, d2udmdv_ik) in \
                          enumerate(zip(dudv[slc], d2udmdv[slc])):

                            dDJDmdv_i[j,k] += assembled_dJdu.inner(d2udmdv_ik[j].vector()) + \
                                              assemble_d2Jdudm[j].inner(dudv_ik.vector()) + \
                                              (assembled_d2Jdu2*dudv_ik.vector()).inner(dudm_j.vector())

                return dDJDmdv_i # -> 2D array

        elif is_type_function:

            if not_ignore_dFdv:
                raise RuntimeError('Not possible to compute sensitivities with respect '
                                   'to `dolfin.Function`(s). The sensitivities can only '
                                   'be computed if the paramter `ignore_dFdv` is `True`.')

            assert not_ignore_dJdv

            dJdv    = [derivative(self._J, v_i) for v_i in v]
            d2Jdudv = [derivative(dJdv_i, self._u) for dJdv_i in dJdv]
            d2Jdmdv = [[diff(dJdv_i, m_j) for m_j in self._m] for dJdv_i in dJdv]

            def compute_dDJDmdv(i, _1, _2, _3):

                dDJDmdv_i = np.zeros((self._n, v[i].vector().size()), float)

                for dDJDmdv_ij, d2Jdmdv_ij, dudm_j in zip(dDJDmdv_i, d2Jdmdv[i], self._dudm):
                    dDJDmdv_ij += assemble(d2Jdmdv_ij) + assemble(d2Jdudv[i])*dudm_j.vector()

                return dDJDmdv_i # -> 2D array

        else:
            raise TypeError('Parameter `v` must contain exclusively either '
                            '`dolfin.Constant`s or `dolfin.Function`s.')

        D2JDm2 = None # to be alias for current `self._property['cumsum_D2JDm2']`
        inv_D2JDm2 = None # to compute inverse of `D2JDm2` when `D2JDm2` changes

        def observe_dmdv(t=None):
            '''Compute the local model parameter sensitivities at time `t`.

            Parameters
            ----------
            t : float or int or None (optional)
                Time at which to compute the model parameter sensitivities.
                If `t` is `None` then `t` defaults to the current time.

            Returns
            -------
            dmdv : list of numpy.ndarray's (2D)
                The model parameter sensitivities. The value at `dmdv[I][J,K]`
                corresponds to the `I`th argument, the `J`th model parameter
                sensitivity with respect to the `K`th DOF of the argument.

            '''

            nonlocal D2JDm2
            nonlocal inv_D2JDm2

            if t is None:
                t = self._property['observation_time']

                if t is None and self._property['cumsum_D2JDm2'] is None:
                    raise RuntimeError('Parameter `t` can not be `None`')

            if not self._property['is_converged']:
                logger.warning('Inverse solver is not converged')

            if self._property['cumsum_D2JDm2'] is None:
                logger.info('Solving forward problem')
                self.solve_forward_problem()

            if D2JDm2 is not self._property['cumsum_D2JDm2']:
                D2JDm2 = self._property['cumsum_D2JDm2']

                try: # Could be ill-conditioned
                    inv_D2JDm2 = linalg.inv(-D2JDm2)
                except linalg.LinAlgError:
                    inv_D2JDm2 = linalg.pinv(-D2JDm2)

            if self.require_nonlinear_solution(t):
                self.update_nonlinear_solution(t)

            if self._property['is_missing_dudm']:
                self._compute_dudm()

            if not_ignore_dFdv:
                compute_dudv_d2udmdv()
                assembled_dJdu = assemble(self._dJdu)
                assembled_d2Jdu2 = assemble(self._d2Jdu2)
                assembled_d2Jdudm = [assemble(self._d2Jdudm[i_m])
                                     for i_m in range(self._n)]
            else:
                assembled_dJdu = None
                assembled_d2Jdu2 = None
                assembled_d2Jdudm = None

            dmdv = []

            for i in range(len(v)):

                D2JDmDv_i = compute_dDJDmdv(i, assembled_dJdu,
                    assembled_d2Jdu2, assembled_d2Jdudm) # -> 2D array

                dmdv.append(inv_D2JDm2.dot(D2JDmDv_i))

            return dmdv

        return observe_dmdv


    def observe_dmdu(self, t=None, constraint_vectors=None):
        '''Estimate the model parameter sensitivities with respect to an
        arbitrary perturbation of the degrees of freedom of the primary field.

        Method
        ------
        The method is based on the solution to the following problem:

            $\frac{du}{dm} dm_{perturb} = du_{perturb}$,

        whose variational form can be defined as

            $\frac{du}{dm_I} (\frac{du}{dm_J} {dm_{perturb}}_J - du_{perturb}) = 0
                \quad \forall I$

        Notes
        -----
        The constraint vectors will typically be for the reaction forces. For
        example, suppose you wish to compute the sensitivities of the cost
        subject to the constness of an observed reaction force on a measurement
        boundary. The reaction force variation (i.e. a constraint vector) can
        be obtained useing `observe_dfdm(t)[i_msr][i_dim]` where `i_msr` denotes
        a particular measurement and `i_dim` -- the observed force component.

        Important
        ---------
        The number of constraint vectors must be fewer than the number of
        model parameters with respect to which the cost sensitivity will be
        computed.

        Todo
        ----
        The model parameter sensitivities are obtained with respect to a
        variation in the primary field, which is generally not the same as the
        displacement field because the primary field may be a mixed function
        space. It would be more useful to compute the sensitivities just with
        respect to the variations in the model displacement field. To do this,
        the displacement field degrees of freedom need to be extracted from the
        primary field.

        Returns
        -------
        dmdu : numpy.ndarray (2D)
            Model parameter sensitivities for observation time `t` with respect
            to the perturbations of the degrees of freedom of the primary field.

        '''

        MAXCOND = 1e9

        dx = dolfin.dx
        dot = dolfin.dot

        du = dolfin.TestFunction(self._V)
        dudm = self.observe_dudm(t, False)

        if (hasattr(constraint_vectors, '__len__')
            and len(constraint_vectors) != 0):
            C = constraint_vectors
        else:
            C = None

        if C is not None:

            C = self._std_model_parameter_constraints(C)
            R = self._compute_orthogonalizing_operator(C)

            eigval, eigvec = np.linalg.eigh(R)
            abs_eigval = np.abs(eigval)

            eigvec = eigvec[:, abs_eigval > abs_eigval.max() / MAXCOND]

            dudm = tuple(sum(dudm_i*dm_i
                for dudm_i, dm_i in zip(dudm, dm_hat))
                for dm_hat in eigvec.T)

            local_to_global_transform = eigvec

        else:
            local_to_global_transform = np.identity(self._n)

        n = len(dudm)
        A = np.zeros((n, n))
        B = np.zeros((n, self._V.dim()))

        for i, dudm_i in enumerate(dudm):
            A[i,i] = assemble(dot(dudm_i, dudm_i)*dx)
            B[i,:] = assemble(dot(dudm_i, du)*dx).get_local()
            for j, dudm_j in enumerate(dudm[i+1:], start=i+1):
                A[i,j] = assemble(dot(dudm_i, dudm_j)*dx)
                A[j,i] = A[i,j]

        ind = np.flatnonzero(A.any(1))

        if len(ind):

            A = A[ind,:][:,ind]
            B = B[ind,:]

            x = linalg.inv(A).dot(B) # Local axes values
            dmdu = local_to_global_transform[:,ind].dot(x)

        else:
            dmdu = np.zeros((n, self._V.dim()))

        return dmdu


    def observe_f_obs(self, ts=None):
        '''Compute the model reaction forces.

        Parameters
        ----------
        ts : numeric value or a sequence of numeric values (optional)
            The observation time(s). If `ts` is an iterable, compute the model
            reaction forces for each time in `ts`. If `ts` is `None`, compute
            the forces for the current time.

        Returns
        -------
        f_obs : list of list's of numpy.ndarray's (1D)
            The observed (model) reaction forces. The value at `f_obs[I][J][K]`
            corresponds to the `I`th measurement, `J`th observation time, and
            the `K`th reaction force component.

        '''

        f_obs = []

        if ts is None:
            ts = self.observation_time

        for t in self._std_observation_times(ts):
            self.update_nonlinear_solution(t)
            f_obs.append([np.array([assemble(fj_obs_i)
                          for fj_obs_i in f_obs_i])
                          for f_obs_i in self._f_obs])

        # Current indexing: time, observed force, force component
        # Reorder indexing: observed force, time, force component

        if hasattr(ts, '__len__') or len(f_obs) > 1:
            f_obs = [[np.array(f_obs_t[i_obs]) for f_obs_t in f_obs]
                     for i_obs in range(len(self._f_obs))]
        else:
            f_obs = [np.array(f_obs[0][i_obs])
                     for i_obs in range(len(self._f_obs))]

        return f_obs


    def observe_f_msr(self, ts=None):
        '''Retrieve the measured reaction forces.

        ts : numeric value or a sequence of numeric values (optional)
            The observation time(s). If `ts` is an iterable, compute the forces
            for each value of time in `ts`. If `ts` is `None`, compute the model
            reaction forces for the current time.

        Returns
        -------
        f_msr : list of list's of numpy.ndarray's (1D)
            The measured (experimental) forces. The value at `f_msr[I][J][K]`
            corresponds to the `I`th measurement, `J`th observation time, and
            the `K`th reaction force component.

        '''

        f_msr = []

        t_cur = self.observation_time

        if ts is None:
            ts = t_cur

        for t in self._std_observation_times(ts):
            self._override_measurement_time(t)
            f_msr.append([np.array([assemble(fj_msr_i)
                          for fj_msr_i in f_msr_i])
                          for f_msr_i in self._f_msr])

        if t != t_cur:
            self._reset_measurement_time()

        # Current indexing: time, measured force, force component
        # Reorder indexing: measured force, time, force component

        if hasattr(ts, '__len__') or len(f_msr) > 1:
            f_msr = [[np.array(f_msr_t[i_msr]) for f_msr_t in f_msr]
                     for i_msr in range(len(self._f_msr))]
        else:
            f_msr = [np.array(f_msr[0][i_msr])
                     for i_msr in range(len(self._f_msr))]

        return f_msr


    def observe_dfdm(self, ts=None):
        '''Compute the sensitivities of the model reaction forces.

        Parameters
        ----------
        ts : numeric value or a sequence of numeric values (optional)
            The observation time(s). If `ts` is an iterable, compute the model
            reaction force sensitivities for each time in `ts`. If `ts` is
            `None`, compute the sensitivities for the current time.

        Returns
        -------
        dfdm : list of list's of numpy.ndarray's (2D)
            Sensitivities of the model reaction forces. The value at `dfdm[I][J]
            [K,L]` corresponds to the `I`th observed force, `J`th observation
            time, `K`th force component, and `L`th model parameter.

        '''

        dfdm = []

        if ts is None:
            ts = self.observation_time

        for t in self._std_observation_times(ts):
            dudm = self.observe_dudm(t, copy=False)

            dfdm.append([np.array([[assemble(dfjdmk_obs_i)
                + assemble(dfjdu_obs_i).inner(dudm_k.vector())
                for dfjdmk_obs_i, dudm_k in zip(dfjdm_obs_i, dudm)]
                for dfjdm_obs_i, dfjdu_obs_i in zip(dfdm_obs_i, dfdu_obs_i)])
                for dfdm_obs_i, dfdu_obs_i in zip(self._dfdm, self._dfdu)])

        # Current indexing: time, observed force, force component, model parameter
        # Reorder indexing: observed force, time, force component, model parameter

        if hasattr(ts, '__len__') or len(dfdm) > 1:
            dfdm = [[np.array(dfdm_t[i_obs]) for dfdm_t in dfdm]
                    for i_obs in range(len(self._f_obs))]
        else:
            dfdm = [np.array(dfdm[0][i_obs])
                    for i_obs in range(len(self._f_obs))]

        return dfdm


    def observe_dfdm_dm(self, dm, ts=None):
        '''Compute the directional sensitivities of the model reaction forces.

        Parameters
        ----------
        dm : sequence of scalar values
            Directional change in the model parameters.

        Returns
        -------
        dfdm_dm : list of (list's of) numpy.ndarray's (1D)
            Directional derivative of the observed force. The value at `dfdm[I]
            [J][K]` corresponds to the `I`th observed force, `J`th observation
            time, `K`th force component. If `ts` is not a sequence, the index
            `J` is squeezed out.

        '''

        if not hasattr(dm, '__len__') or len(dm) != self._n:
            raise TypeError('Parameter `dm` must be sequence-like whose length is '
                            f'equal to the number of model parameters ({self._n}).')

        if ts is None:
            ts = self.observation_time

        if not isinstance(dm, np.ndarray):
            dm = np.array([float(dm_i) for dm_i in dm])

        dfdm = [[dfdm_obs_it.dot(dm) for dfdm_obs_it in dfdm_obs_i] for dfdm_obs_i
                in self.observe_dfdm(ts if hasattr(ts, '__getitem__') else (ts,))]

        if not hasattr(ts, '__getitem__'): # Collapse time
            dfdm = [dfdm_obs_i[0] for dfdm_obs_i in dfdm]

        return dfdm


    def observe_u(self, t=None, copy=True):
        '''Nonlinear solution at time `t`.'''

        if self.require_nonlinear_solution(t):
            self.update_nonlinear_solution(t)

        return self._u.copy(True) if copy else self._u


    def observe_dudm(self, t=None, copy=True):
        '''Partial derivatives of the primary field at time `t`.'''

        if self.require_nonlinear_solution(t):
            self.update_nonlinear_solution(t)

        return super().observe_dudm(t, copy)


    def observe_d2udm2(self, t=None, copy=True):
        '''Compute partial derivatives of the primary field with respect to
        the model parameters at observation time `t`. Note, since the matrix
        d2u/dm2 is symmetric, only the upper triangular part is returned.'''

        if self.require_nonlinear_solution(t):
            self.update_nonlinear_solution(t)

        return super().observe_d2udm2(t, copy)


    def observe_dudm_dm(self, dm, t=None):
        '''Compute the directional sensitivity of the primary field.

        Parameters
        ----------
        dm : sequence of scalar values
            Directional change in the model parameters.

        Returns
        -------
        dudm_dm : dolfin.Function
            Directional derivative of the primary field.

        '''

        if not hasattr(dm, '__len__') or len(dm) != self._n:
            raise TypeError('Parameter `dm` must be sequence-like whose length is '
                            f'equal to the number of model parameters ({self._n}).')

        dudm_dm = Function(self._V)

        if any(dm):
            dudm_dm.assign(sum(dudm_i*dm_i for dudm_i, dm_i
                in zip(self.observe_dudm(t, copy=False), dm)))

        return dudm_dm


    def test_model_parameter_sensitivity_dmdm(self):
        '''Compute the predicted and the expected model parameter self-sensitivities.

        Notes
        -----
        The predicted sensitivities must be close to the negative of the identity
        matrix, which means that if the model parameters were to be perturbed in
        some direction then there would be a corresponding restoration in the
        opposite direction.

        '''

        dmdm_predicted = sum(self.observe_dmdm(t, copy=False)
                             for t in self.observation_times)

        dmdm_expected = -np.identity(len(dmdm_predicted), float)

        return dmdm_predicted, dmdm_expected


    def test_model_parameter_sensitivity_dmdT_msr(self, h=1e-2):
        '''Finite-difference test of the model parameter sensitivities with respect to
        arguments. The measurement perturbations are generated from a random uniform
        distribution. The perturbation magnitudes are normalized to the value of `h`.

        Returns
        -------
        dm_predicted: numpy.ndarray (1D)
            Predicted change in model parameter values.

        dm_expected: numpy.ndarray (1D)
            Expected change in model parameter values.

        '''
        return self.test_model_parameter_sensitivity_dmdv(
            self.observe_dmdT_msr, self._dT_msr_dummy, h)


    def test_model_parameter_sensitivity_dmdu_msr(self, h=1e-2):
        '''Finite-difference test of the model parameter sensitivities with respect to
        arguments. The measurement perturbations are generated from a random uniform
        distribution. The perturbation magnitudes are normalized to the value of `h`.

        Returns
        -------
        dm_predicted: numpy.ndarray (1D)
            Predicted change in model parameter values.

        dm_expected: numpy.ndarray (1D)
            Expected change in model parameter values.

        '''
        return self.test_model_parameter_sensitivity_dmdv(
            self.observe_dmdu_msr, self._du_msr_dummy, h)


    def test_model_parameter_sensitivity_dmdv(self, observe_dmdv, v, h):
        '''Finite-difference test of the model parameter sensitivities with respect to
        arguments. The measurement perturbations are generated from a random uniform
        distribution. The perturbation magnitudes are normalized to the value of `h`.

        Parameters
        ----------
        self : invsolve.InverseSolver
            Instance of an `InverseSolver`.
        v : (sequence of) either dolfin.Function's or dolfin.Constant's
            Arguments.
        h : float
            Stepsize.

        Returns
        -------
        dm_predicted: numpy.ndarray (1D)
            Predicted change in model parameter values.

        dm_expected: numpy.ndarray (1D)
            Expected change in model parameter values.

        Limitations
        -----------
        The same perturbation direction is assumed for all observation times.
        This is simpler to implement but not as general as allowing for arbitrary
        perturbation directions at different observation times.

        '''

        GENERATE_PERTURBATION_DIRECTION = np.random.rand

        if not self._property['is_converged']:
            logger.warning('Results may be inaccurate because '
                           'the solver is not converged.')

        if self.parameters_inverse_solver['absolute_tolerance'] > 1e-6:
            logger.warning('Results may be inaccurate because the solver '
                           'convergence parameter "absolute_tolerance" '
                           'may be insufficiently small.')

        if self.parameters_inverse_solver['relative_tolerance'] > 1e-6:
            logger.warning('Results may be inaccurate because the solver '
                           'convergence parameter "relative_tolerance" '
                           'may be insufficiently small.')

        if self.parameters_inverse_solver['maximum_relative_change']:
            logger.warning('Results may be inaccurate because the solver '
                           'parameter "maximum_relative_change" has '
                           'been prescribed.')

        vs = v if isinstance(v, SEQUENCE_TYPES) else (v,)

        is_type_function = all(isinstance(vi, Function) for vi in vs)
        is_type_constant = all(isinstance(vi, Constant) for vi in vs) \
                               if not is_type_function else False

        if not (is_type_function or is_type_constant):
            raise TypeError('Expected parameter `v` to contain either '
                            '`dolfin.Function`(s) or `dolfin.Constant`(s).')

        # Backup current values of the arguments `v`

        if is_type_function:
            vs_arr = [vi.vector().get_local() for vi in vs]
        else: # is_type_constant
            vs_arr = [vi.values() for vi in vs]

        # Compute perturbation values

        dvs_arr = []

        for vi_arr in vs_arr:
            dvi_arr = GENERATE_PERTURBATION_DIRECTION(len(vi_arr))
            dvi_arr *= h / sqrt(dvi_arr.dot(dvi_arr))
            dvs_arr.append(dvi_arr)

        if __debug__:

            if is_type_function:
                if any(dmdvi.shape[-1] != len(vi) for dmdvi, vi
                       in zip(observe_dmdv(copy=False), vs_arr)):
                    raise TypeError('Parameter `v`')

            else: # is_type_constant
                if any(dmdvi.shape[-1] != len(vi) for dmdvi, vi
                       in zip(observe_dmdv(copy=False), vs_arr)):
                    raise TypeError('Parameter `v`')

        # Back up current model parameter values

        m0 = np.array(self.view_model_parameter_values())

        # Predict model parameter change

        dm_predicted = \
            sum(sum(dmdvi.dot(dvi) for dmdvi, dvi
            in zip(observe_dmdv(t, False), dvs_arr))
            for t in self.observation_times)

        # Perturb arguments using perturbation values

        if is_type_function:
            for vi, vi_arr, dvi_arr in zip(vs, vs_arr, dvs_arr):
                vi.vector()[:] = vi_arr + dvi_arr

        else: # is_type_constant
            for vi, vi_arr, dvi_arr in zip(vs, vs_arr, dvs_arr):
                if len(vi.ufl_shape):
                    vi.assign(Constant(vi_arr + dvi_arr)) # Vector-valued
                else: # vi.ufl_shape == ()
                    vi.assign(vi_arr[0] + dvi_arr[0]) # Scalar-valued

        try:

            # Compute perturbed model parameter values

            if not self.solve_inverse_problem()[1]:
                raise RuntimeError('Inverse solver did not converge')

            m1 = np.array(self.view_model_parameter_values())

        finally:

            # Restore original model parameter and argument values

            self.assign_model_parameters(m0)

            if is_type_function:
                for vi, vi_arr in zip(vs, vs_arr):
                    vi.vector()[:] = vi_arr

            else: # is_type_constant
                for vi, vi_arr in zip(vs, vs_arr):
                    if len(vi.ufl_shape):
                        vi.assign(Constant(vi_arr)) # Vector-valued
                    else: # vi.ufl_shape == ()
                        vi.assign(vi_arr[0]) # Scalar-valued

        # Resolve the inverse problem for the original values

        if not self.solve_inverse_problem()[1]:
            raise RuntimeError('Inverse solver did not converge')

        return dm_predicted, m1-m0


    @property
    def num_u_msr(self):
        return len(self._dx_msr)

    @property
    def num_T_msr(self):
        return len(self._ds_msr)

    @property
    def num_f_msr(self):
        return len(self._ds_msr)


    @staticmethod
    def _std_init_args_u_obs_u_msr_dx_msr(u_obs, u_msr, dx_msr):
        '''Return standardized initialization arguments.

        Parameters
        ----------
        u_obs : vector dolfin.Function
            Observed displacement field.
        u_msr : (sequence of) vector-like ufl expression(s).
            Measured displacement fields.
        dx_msr : (sequence of) dolfin.Measure(s).
            Integration measures.

        '''

        if not isinstance( u_msr, SEQUENCE_TYPES): u_msr  = ( u_msr,)
        if not isinstance(dx_msr, SEQUENCE_TYPES): dx_msr = (dx_msr,)

        if len(u_msr) != len(dx_msr):
            raise TypeError('Parameters `u_msr` and `dx_msr`')

        if not (isinstance(u_obs, ufl_expr_t) and len(u_obs.ufl_shape) == 1):
            raise TypeError('Parameter `u_obs`')

        if not all(isinstance(u_msr_i, ufl_expr_t) and \
           u_msr_i.ufl_shape == u_obs.ufl_shape for u_msr_i in u_msr):
            raise TypeError('Parameters `u_msr` and `u_obs`')

        if not all(isinstance(dx_msr_i, Measure) for dx_msr_i in dx_msr):
            raise TypeError('Parameter `dx_msr`')

        if not isinstance( u_msr, tuple): u_msr  = tuple( u_msr)
        if not isinstance(dx_msr, tuple): dx_msr = tuple(dx_msr)

        return u_obs, u_msr, dx_msr


    @staticmethod
    def _std_init_args_T_obs_T_msr_ds_msr(T_obs, T_msr, ds_msr):
        '''Return standardized initialization arguments.

        Parameters
        ----------
        T_obs : (sequence of (sequences of)) vector ufl expression(s)
            Observed tractions.
        T_msr : (sequence of (sequences of)) vector ufl expression(s)
            Measured tractions.
        ds_msr : (sequence of (sequences of)) dolfin.Measure(s)
            Integration measures.

        '''

        if not isinstance(ds_msr, SEQUENCE_TYPES):
            T_obs, T_msr, ds_msr = (T_obs,), (T_msr,), (ds_msr,)

        else:

            if not isinstance(T_obs, SEQUENCE_TYPES): T_obs = (T_obs,)
            if not isinstance(T_msr, SEQUENCE_TYPES): T_msr = (T_msr,)

            if len(T_obs) != len(ds_msr):
                raise TypeError('Parameters `T_obs` and `ds_msr`')

            if len(T_msr) != len(ds_msr):
                raise TypeError('Parameters `T_msr` and `ds_msr`')

        for i, (T_obs_i, T_msr_i, ds_msr_i) in enumerate(zip(T_obs, T_msr, ds_msr)):

            if not isinstance( T_obs_i, SEQUENCE_TYPES): T_obs_i  = ( T_obs_i,)
            if not isinstance( T_msr_i, SEQUENCE_TYPES): T_msr_i  = ( T_msr_i,)
            if not isinstance(ds_msr_i, SEQUENCE_TYPES): ds_msr_i = (ds_msr_i,)

            if not all(isinstance(T_obs_ij, ufl_expr_t) for T_obs_ij in T_obs_i):
                raise TypeError('Parameter `T_obs`')

            if not all(isinstance(T_msr_ij, ufl_expr_t) for T_msr_ij in T_msr_i):
                raise TypeError('Parameter `T_msr`')

            if not all(isinstance(ds_msr_ij, Measure) for ds_msr_ij in ds_msr_i):
                raise TypeError('Parameter `ds_msr`')

            ufl_shape_obs = T_obs_i[0].ufl_shape
            ufl_shape_msr = T_msr_i[0].ufl_shape

            if len(ufl_shape_obs) != 1:
                raise TypeError('Parameter `T_obs`')

            if len(ufl_shape_msr) != 1:
                raise TypeError('Parameters `T_msr`')

            if any(T_obs_ij.ufl_shape != ufl_shape_obs for T_obs_ij in T_obs_i[1:]):
                raise TypeError('Parameter `T_obs`')

            if any(T_msr_ij.ufl_shape != ufl_shape_msr for T_msr_ij in T_msr_i[1:]):
                raise TypeError('Parameter `T_msr`')

            if ufl_shape_obs != ufl_shape_msr:
                raise TypeError('Parameters `T_obs` and `T_msr`')

            if len(ds_msr_i) > 1:
                if len(T_obs_i) == 1:
                    if len(T_msr_i) == 1:
                        ds_msr_i = (sum(ds_msr_i[1:], ds_msr_i[0]),)
                    else:
                        T_obs_i = (T_obs_i[0],) * len(ds_msr_i)
                elif len(T_msr_i) == 1:
                    T_msr_i = (T_msr_i[0],) * len(ds_msr_i)

            if not (len(T_obs_i) == len(T_msr_i) == len(ds_msr_i)):
                raise RuntimeError('Parameters `T_obs`, `T_msr` and `ds_msr`')

            if not isinstance( T_obs_i, tuple): T_obs_i  = tuple( T_obs_i)
            if not isinstance( T_msr_i, tuple): T_msr_i  = tuple( T_msr_i)
            if not isinstance(ds_msr_i, tuple): ds_msr_i = tuple(ds_msr_i)

            if T_obs_i is not T_obs[i]:
                if not isinstance(T_obs, list): T_obs = list(T_obs)
                T_obs[i] = T_obs_i if isinstance(T_obs_i, tuple) else tuple(T_obs_i)

            if T_msr_i is not T_msr[i]:
                if not isinstance(T_msr, list): T_msr = list(T_msr)
                T_msr[i] = T_msr_i if isinstance(T_msr_i, tuple) else tuple(T_msr_i)

            if ds_msr_i is not ds_msr[i]:
                if not isinstance(ds_msr, list): ds_msr = list(ds_msr)
                ds_msr[i] = ds_msr_i if isinstance(ds_msr_i, tuple) else tuple(ds_msr_i)

        if any(T_obs_i[0].ufl_shape != ufl_shape_obs for T_obs_i in T_obs[:-1]):
            raise TypeError('Parameter `T_obs` (and `T_msr`)')

        if not isinstance( T_obs, tuple): T_obs  = tuple( T_obs)
        if not isinstance( T_msr, tuple): T_msr  = tuple( T_msr)
        if not isinstance(ds_msr, tuple): ds_msr = tuple(ds_msr)

        return T_obs, T_msr, ds_msr


    @staticmethod
    def _std_subdims_v_msr(subdims, size_msr, dim_max):
        '''Standardized indices into subdimensions of `u_msr` or `f_msr`.

        Parameters
        ----------
        subdims : integer, integer sequence, or sequence of integer sequences
            Generic form of subdimension indices into measurements.

        Returns
        ----------
        subdims : sequence of integer sequence(s)
            Standardized subdimension indices into measurements.

        '''

        if subdims is None:
            subdims = (range(dim_max),) * size_msr
        else:
            if hasattr(subdims, '__getitem__'):
                if all(hasattr(sub_i, '__getitem__') for sub_i in subdims):

                    if not all(isinstance(dim_j, int)
                               for sub_i in subdims for dim_j in sub_i):
                        raise TypeError('Parameter `subdims` (sequence of sequences) '
                                        'contains non-integer subdimension indices')

                    if not all(-dim_max <= dim_j < dim_max \
                               for sub_i in subdims for dim_j in sub_i):
                        raise ValueError('Parameter `subdims` (sequence of sequences) '
                                         'contains out-of-bounds subdimension indices')

                    if len(subdims) != size_msr:
                        if len(subdims) == 0:
                            subdims = (subdims,) * size_msr
                        elif len(subdims) == 1:
                            subdims = (subdims[0],) * size_msr
                        else:
                            raise IndexError('Parameter `subdims` (sequence of sequences) must '
                                             'have length equal to the number of measurements')

                elif all(isinstance(dim_i, int) for dim_i in subdims):

                    if not all(-dim_max <= dim_i < dim_max for dim_i in subdims):
                        raise ValueError('Parameter `subdims` (sequence of integers) '
                                         'contains out-of-bounds subdimension indices')

                    subdims = (subdims,) * size_msr

                else:
                    raise TypeError

            elif isinstance(subdims, int):

                if not (-dim_max <= subdims < dim_max):
                    raise ValueError('Parameter `subdims` (integer) is '
                                     'out-of-bounds as a subdimension index')

                subdims = ((subdims,),) * size_msr

            else:
                raise TypeError('Parameter `subdims`')

        return subdims


    def _std_model_parameter_constraints(self, constraint_vectors):

        if not isinstance(constraint_vectors, np.ndarray):
            constraint_vectors = np.array(constraint_vectors, ndmin=2)

        elif constraint_vectors.ndim == 1:
            constraint_vectors = constraint_vectors[np.newaxis,:]

        if constraint_vectors.shape[0] >= self._n or \
           constraint_vectors.shape[1] != self._n:
            raise TypeError('Each constraint vector must have length equal '
                            f'to the number of model parameters ({self._n}) '
                            'and the number of such vectors should be fewer '
                            'than the number of model parameters.')

        return constraint_vectors


    @staticmethod
    def _compute_orthogonalizing_operator(sequence_of_vectors):
        C = np.array(sequence_of_vectors, dtype=float, copy=False, ndmin=2)
        return np.identity(len(C.T)) - C.T.dot(linalg.inv(C.dot(C.T)).dot(C))
