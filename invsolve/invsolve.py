
'''Define the inverse solver class `InverseSolver` used for solving inverse
problems of model parameter estimation.'''

import ufl
from dolfin import *
import dolfin
import logging
import importlib

import numpy as np
import scipy.linalg as linalg

from dolfin import (
    Constant,
    Function,
    action,
    assemble,
    assemble_system,
    derivative,
    diff,
    )

from . import config
logger = logging.getLogger()


class InverseSolver:
    '''
    Notes
    -----
    This class does not take measurements as arguments. The measurements are
    assumed to be provided in the defintion of the cost functional. Setting of
    measurements and of the boundary conditions is deligated to the measurement
    setter function.

    '''

    ITER_METHOD_NEWTON = 'newton'
    ITER_METHOD_GRADIENT = 'gradient'
    COMPUTE_SENS_METHOD_DIRECT = 'direct'
    COMPUTE_SENS_METHOD_ADJOINT = 'adjoint'

    def __init__(self, cost, model, model_parameters, cost_u, cost_f,
        measurement_setter=None, observation_times=None):
        '''
        Initialize inverse solver.

        Parameters
        ----------

        model : has attributes
            u : dolfin.Function
                Function for the displacement field.
            Pi : ufl.Form
                Potential energy of the hyperelastic material.
            bcs : list or tuple of dolfin.DirichletBC's
                Dirichlet (displacement) boundary conditions.

        cost : ufl.Form
            Cost functional to be minimized

        model_parameters : contains dolfin.Constant's
            The model parameters to be optimized.

        measurement_setter : function-like that takes time as single argument
            Function that sets all measurements for a given observation time.

        observation_times : iterable
            Discrete times for which to consider the value of the model cost.

        '''

        if not (hasattr(model,'u') and hasattr(model,'Pi') and hasattr(model,'bcs')):
            # print (model)
            raise TypeError('`model` must have attributes: `u`, `Pi` and `bcs`.')

        if not isinstance(model.u, Function):
            raise TypeError('Parameter `model.u` must be a `dolfin.Function`.')

        if not isinstance(model.Pi, ufl.Form):
            raise TypeError('Parameter `model.Pi` must be a `ufl.Form`.')

        if not (isinstance(model.bcs, dolfin.DirichletBC) or (
                isinstance(model.bcs, (list,tuple)) and
                all(isinstance(bc, dolfin.DirichletBC) for bc in model.bcs))):
            raise TypeError('Parameter `model.bcs` must contain `dolfin.DirichletBC`s.')

        if not isinstance(cost, ufl.Form):
            raise TypeError('Parameter `cost` must be a `ufl.Form`.')

        self.model_parameters = model_parameters

        self._m = tuple(list_variables_from_iterable(
            iterable=model_parameters, valid_types=Constant))

        self._n = len(self._m)

        if not all(isinstance(m_i, Constant) for m_i in self._m):
            raise TypeError('`model_parameters` must contain `dolfin.Constant`s.')

        self.assign_measurement_setter(measurement_setter)
        self.assign_observation_times(observation_times)

        self._J = cost
        if cost_u is not None:
           self._Ju = cost_u
        if cost_f is not None:
           self._Jf = cost_f

        self._t = None
        self._u = model.u
        self._Pi = model.Pi
        self._bcs = model.bcs

        self._V = self._u.function_space()
        self._F = derivative(self._Pi, self._u)
        self._z = Function(self._V) # as self._u

        self._dudm = tuple(Function(self._V) for i in range(self._n))

        self._d2udm2 = tuple(
            tuple(Function(self._V) for j in range(i, self._n))
            for i in range(self._n)) # store upper triangular part

        self._dJdu, self._dJdm, self._d2Jdu2, self._d2Jdudm, self._d2Jdm2 \
            = self._partial_derivatives_of_form(self._J)

        self._dFdu, self._dFdm, self._d2Fdu2, self._d2Fdudm, self._d2Fdm2 \
            = self._partial_derivatives_of_form(self._F)

        # For assembled LHS of tanget system
        self._dFdu_mtx = dolfin.PETScMatrix()

        # Initialize solution checkpoints

        self._reset_checkpoints_nonlinear_solve()
        self._reset_checkpoints_inverse_solve()

        # Dirichlet BC's

        self._bcs_zro = [] # homogenized bcs
        self._bcs_dof = [] # Dirichlet dofs

        for bc in self._bcs: # assuming dofs will remain the same
            self._bcs_dof.extend(bc.get_boundary_values().keys())
            bc_zro = dolfin.DirichletBC(bc); bc_zro.homogenize()
            self._bcs_zro.append(bc_zro)

        self._bcs_dof = np.array(self._bcs_dof, dtype=np.uint)

        # Nonlinear problem

        nonlinear_problem = dolfin.NonlinearVariationalProblem(
            self._F, self._u, bcs=self._bcs, J=self._dFdu)

        self._nonlinear_solver = dolfin.NonlinearVariationalSolver(nonlinear_problem)
        self._nonlinear_solver.parameters.update(config.parameters_nonlinear_solver)
        self.parameters_nonlinear_solver = self._nonlinear_solver.parameters
        # self._nonlinear_solver.parameters.nonzero_initial_guess = True

        # Linear adjoint problem

        adjoint_problem = dolfin.LinearVariationalProblem(
            dolfin.adjoint(self._dFdu), -self._dJdu, self._z, bcs=self._bcs_zro)

        self._adjoint_solver = dolfin.LinearVariationalSolver(adjoint_problem)
        self.parameters_adjoint_solver = self._adjoint_solver.parameters

        for k in set(self.parameters_adjoint_solver) & set(config.parameters_linear_solver):
            self.parameters_adjoint_solver[k] = config.parameters_linear_solver[k]

        # Linear solver (for tangent problems)

        self._linear_solver = dolfin.LUSolver()
        self.parameters_linear_solver = self._linear_solver.parameters

        for k in set(self.parameters_linear_solver) & set(config.parameters_linear_solver):
            self.parameters_linear_solver[k] = config.parameters_linear_solver[k]

        # Inverse solver parameters

        self.parameters_inverse_solver = config.parameters_inverse_solver.copy()


    def _reset_checkpoints_nonlinear_solve(self):
        self._is_missing_z = True
        self._is_missing_dudm = True
        self._is_missing_d2udm2 = True
        self._is_missing_dFdu_mtx = True

    def _reset_checkpoints_inverse_solve(self):
        self._cumsum_DJDm = None
        self._cumsum_D2JDm2 = None
        self._is_converged = False


    def _observation_times_getdefault(self, observation_times=None):
        '''Return parameter `observation_times` as a tuple; however, if
        `observation_times` is `None`, return `self._observation_times`.'''

        if observation_times is None:
            return self._observation_times

        elif hasattr(observation_times, '__iter__'):
            return tuple(observation_times)

        elif isinstance(observation_times, (float,int)):
            return (observation_times,)

        else:
            raise TypeError('Not clear how to cast the value of '
                'parameter `observation_times` into a `tuple`.')


    def assign_observation_times(self, observation_times):
        '''Assign new observation times and reset solution checkpoints.'''

        self._reset_checkpoints_inverse_solve()

        if hasattr(observation_times, '__iter__'):
            self._observation_times = tuple(observation_times)

        elif isinstance(observation_times, (float, int, type(None))):
            self._observation_times = (observation_times,)

        else:
            raise TypeError('Not clear how to cast the value of '
                'parameter `observation_times` into a `tuple`.')


    def assign_measurement_setter(self, measurement_setter):
        '''Assign measurement setter.

        Parameters
        ----------
        measurement_setter : function-like that takes time as single argument
            Function that sets all measurements for a given observation time.

        '''
        if callable(measurement_setter):
            self._measurement_setter = measurement_setter
        elif measurement_setter is None:
            self._measurement_setter = lambda t: None # dummy
        else:
            raise TypeError('Require parameter `measurement_setter` to be '
                'a callable (e.g. a function of time) or to be a `None`.')


    def assign_model_parameters(self, model_parameters,
        valid_types = (float, int, np.float64, np.int64, Constant)):
        '''Assign model parameters from a (nested) iterable. `model_parameters`
        will be flattened and then the values will be correspondingly assigned.'''

        self._reset_checkpoints_nonlinear_solve()
        self._reset_checkpoints_inverse_solve()

        m = list_variables_from_iterable(model_parameters, valid_types)

        if self._n != len(m):
            raise TypeError('Expected `model_parameters` to contain exactly '
                '{self._n} model parameter(s) but instead found {len(m)}.')

        for self_m_i, m_i in zip(self._m, m):
            self_m_i.assign(float(m_i))


    def _partial_derivatives_of_form(self, f):
        '''Partial derivatives of a form `f`.

        Parameters
        ----------
        f : ufl.Form
            A form that is differentiable with respect to the displacement
            field `self._u` (`dolfin.Function`) and model parameters `self._m`
            (`tuple` of `dolfin.Constant`s).

        '''

        dfdu    = derivative(f, self._u)
        d2fdu2  = derivative(dfdu, self._u)
        dfdm    = tuple(diff(f, m_i) for m_i in self._m)
        d2fdudm = tuple(diff(dfdu, m_i) for m_i in self._m)
        d2fdm2  = tuple(tuple(diff(dfdm_i, m_j) for m_j in self._m[i:])
            for i, dfdm_i in enumerate(dfdm)) # only upper triangular part

        return dfdu, dfdm, d2fdu2, d2fdudm, d2fdm2


    def _assemble_tangent_system(self, rhs):
        '''Assemble tangent stiffness and right hand side forms.'''

        if self._is_missing_dFdu_mtx:
            lhs, rhs = assemble_system(self._dFdu, rhs,
                self._bcs_zro, A_tensor=self._dFdu_mtx)
            self._is_missing_dFdu_mtx = False
        else:
            lhs = self._dFdu_mtx
            rhs = assemble(rhs)
            rhs[self._bcs_dof] = 0.0

        return lhs, rhs


    def _compute_z(self, rhs):
        '''Compute adjoint variable.'''
        lhs, rhs = self._assemble_tangent_system(rhs)
        self._linear_solver.solve(lhs, self._z.vector(), rhs)
        self._is_missing_z = False


    def _compute_dudv(self, dFdv, dudv):
        '''Compute displacement derivatives with respect to variables.'''

        lhs, rhs = self._assemble_tangent_system(-dFdv[0])
        self._linear_solver.solve(lhs, dudv[0].vector(), rhs)

        for dudv_i, dFdv_i in zip(dudv[1:], dFdv[1:]):

            rhs = assemble(-dFdv_i); rhs[self._bcs_dof] = 0.0
            self._linear_solver.solve(lhs, dudv_i.vector(), rhs)


    def _compute_dudm(self):
        '''Compute displacement derivatives with respect to model parameters.'''
        self._compute_dudv(self._dFdm, self._dudm)
        self._is_missing_dudm = False


    def _compute_d2udm2(self):
        '''Compute second order displacement derivatives with respect to model
        parameters. Note, since `d2udm2` is symmetric, only the upper triangular
        part is stored.'''

        if self._is_missing_dudm or self._is_missing_dFdu_mtx:
            self._compute_dudm() # -> self._dudm, self._dFdu_mtx

        for i, dudm_i in enumerate(self._dudm, start=0):
            for j, dudm_j in enumerate(self._dudm[i:], start=i):

                rhs = - (
                    assemble(self._d2Fdm2[i][j-i]
                        + action(action(self._d2Fdu2, dudm_j), dudm_i))
                    + assemble(self._d2Fdudm[j])*(dudm_i.vector())
                    + assemble(self._d2Fdudm[i])*(dudm_j.vector()))

                rhs[self._bcs_dof] = 0.0

                self._linear_solver.solve(self._dFdu_mtx,
                    self._d2udm2[i][j-i].vector(), rhs)

        self._is_missing_d2udm2 = False


    def _factory_compute_dudv(self, dFdv):
        '''Factory for computing the first order derivatives of the
        displacement field with respect to variables.'''

        dudv = tuple(Function(self._V) for _ in dFdv)

        def compute_dudv():
            '''Compute derivatives of displacement field'''
            self._compute_dudv(dFdv, dudv)

        return compute_dudv, dudv


    def _factory_compute_dudv_d2udmdv(self, dFdv):
        '''Factory for computing first and second order mixed derivatives of
        displacement field with respect to model parameters and variables.'''

        dudv = tuple(Function(self._V) for _ in dFdv)
        d2udmdv = tuple(tuple(Function(self._V)
            for j in range(self._n)) for _ in dFdv)

        d2Fdudv = tuple(derivative(dFdv_i, self._u) for dFdv_i in dFdv)
        d2Fdmdv = tuple(tuple(diff(dFdv_i, m_j) for m_j in self._m)
            for dFdv_i in dFdv)

        def compute_dudv_d2udmdv():
            '''Compute first and second order mixed derivatives of displacement
            field with respect to model parameters and variables.'''

            self._compute_dudv(dFdv, dudv)

            if self._is_missing_dudm:
                self._compute_dudm()

            for i, dudv_i in enumerate(dudv):
                for j, dudm_j in enumerate(self._dudm):

                    rhs = - (
                        assemble(d2Fdmdv[i][j]
                            + action(action(self._d2Fdu2, dudv_i), dudm_j))
                        + assemble(self._d2Fdudm[j])*(dudv_i.vector())
                        + assemble(d2Fdudv[i])*(dudm_j.vector()))

                    rhs[self._bcs_dof] = 0.0

                    self._linear_solver.solve(self._dFdu_mtx,
                        d2udmdv[i][j].vector(), rhs)

        return compute_dudv_d2udmdv, dudv, d2udmdv


    def solve_nonlinear_problem(self, t=None):
        '''Solve non-linear problem at observation time.

        Parameters
        ----------
        t : float or int or None (optional)
            The time of the observation. Note, the `self._measurement_setter` is
            assumed to be responsible for interpreting the value of `t`; however,
            if the value is `None`, `self._measurement_setter` will not called.

        '''

        self._reset_checkpoints_nonlinear_solve()
        print(self.view_model_parameters_as_array())
        print([float(m_i.values()) for m_i in self._m])
        # print(list(self._m))

        if t is not None:
            self._measurement_setter(t)
            self._t = t
        print(t)

        n, b = self._nonlinear_solver.solve()

        if not b:
            logger.error('Nonlinear solver failed to converge')

        return n, b


    def solve_adjoint_problem(self):
        '''Solve adjoint problem at current time.'''
        # self._compute_z(-self._dJdu)
        self._adjoint_solver.solve()
        self._is_missing_z = False


    def factory_compute_DJDm(self, sensitivity_method='default'):
        '''Factory for computing model cost sensitivities.'''

        if sensitivity_method == 'default':
            sensitivity_method = self.parameters_inverse_solver['sensitivity_method']

        if sensitivity_method == self.COMPUTE_SENS_METHOD_ADJOINT:
            compute_DJDm = self.compute_DJDm_method_adjoint

        elif sensitivity_method == self.COMPUTE_SENS_METHOD_DIRECT:
            compute_DJDm = self.compute_DJDm_method_direct

        else:
            raise KeyError("parameters_inverse_solver['sensitivity_method'] "
                " = {self.parameters_inverse_solver['sensitivity_method']} "
                "is not one of the available methods.")

        return compute_DJDm


    def compute_DJDm_method_adjoint(self):
        '''Compute the derivatives of the model cost `J` with respect to
        model parameters `m` using the adjoint method: `dJ/dm + dF/dm dJ/dF`.

        Notes
        -----
        The adjoint method requires one solve for any number of model parameters:
            solve(adjoint(self._dFdu) == -self._dJdu, self._z, bcs=self._bcs_zro)

        '''

        DJDm = np.zeros((self._n,), float)

        if self._is_missing_z:
            self.solve_adjoint_problem()

        for i, (dJdm_i, dFdm_i) in enumerate(zip(self._dJdm, self._dFdm)):
            DJDm[i] += assemble(dJdm_i) + assemble(dFdm_i).inner(self._z.vector())

        return DJDm


    def compute_DJDm_method_direct(self):
        '''Compute the derivatives of the model cost `J` with respect to
        model parameters `m` using a direct method: `dJ/dm + du/dm dJ/du`.

        Notes
        -----
        The direct method requires as many solves as there model parameters:
            solve(self._dFdu == -self._dFdm[i], self._dudm[i], bcs=self._bcs_zro)

        '''

        DJDm = np.zeros((self._n,), float)
        assembled_dJdu = assemble(self._dJdu)

        if self._is_missing_dudm:
            self._compute_dudm()

        for i, (dJdm_i, dudm_i) in enumerate(zip(self._dJdm, self._dudm)):
            DJDm[i] += assemble(dJdm_i) + assembled_dJdu.inner(dudm_i.vector())

        return DJDm


    def factory_compute_DJDm_D2JDm2(
        self, sensitivity_method='default', approximate_D2JDm2='default'):
        '''Factory for computing model cost sensitivities.'''

        if sensitivity_method == 'default':
            sensitivity_method = self.parameters_inverse_solver['sensitivity_method']

        if approximate_D2JDm2 == 'default':
            approximate_D2JDm2 = self.parameters_inverse_solver['approximate_D2JDm2']

        if sensitivity_method == self.COMPUTE_SENS_METHOD_ADJOINT:

            def compute_DJDm_D2JDm2():
                return self.compute_DJDm_D2JDm2_method_adjoint(approximate_D2JDm2)

        elif sensitivity_method == self.COMPUTE_SENS_METHOD_DIRECT:

            def compute_DJDm_D2JDm2():
                return self.compute_DJDm_D2JDm2_method_direct(approximate_D2JDm2)

        else:
            raise RuntimeError('parameters_inverse_solver[\'sensitivity_method\']: '
                "\"{self.parameters_inverse_solver['sensitivity_method']}\" "
                'is not a valid parameter.')

        return compute_DJDm_D2JDm2


    def compute_DJDm_D2JDm2_method_adjoint(self, ignore_action_dJdu_d2udm2=False):
        '''Compute first order and second order derivatives of cost with respect
        to model parameters. This computation relies on the adjoint method.'''

        DJDm = np.zeros((self._n,), float)
        D2JDm2 = np.zeros((self._n,self._n), float)

        assembled_dJdu = assemble(self._dJdu)
        assembled_d2Jdu2 = assemble(self._d2Jdu2)

        if self._is_missing_dudm:
            self._compute_dudm()

        if self._is_missing_z and not ignore_action_dJdu_d2udm2:
            self.solve_adjoint_problem()

        for i, (d2Jdudm_i, d2Fdudm_i, dJdm_i, dFdm_i, dudm_i) in enumerate(
          zip(self._d2Jdudm, self._d2Fdudm, self._dJdm, self._dFdm, self._dudm)):

            # DJDm[i] = assemble(dJdm_i + action(self._dJdu, dudm_i)) # can fail
            DJDm[i] = assemble(dJdm_i) + assembled_dJdu.inner(dudm_i.vector())
            print("computing DJDm(i) for i= ", i)

            for j, (d2Jdm2_ij, d2Fdm2_ij, d2Jdudm_j, d2Fdudm_j, dudm_j) in \
              enumerate(zip(self._d2Jdm2[i], self._d2Fdm2[i], self._d2Jdudm[i:],
              self._d2Fdudm[i:], self._dudm[i:]), start=i):

                D2JDm2[i,j] = (
                    assemble(d2Jdm2_ij)
                    + (assembled_d2Jdu2*dudm_j.vector()).inner(dudm_i.vector())
                    + assemble(d2Jdudm_j).inner(dudm_i.vector())
                    + assemble(d2Jdudm_i).inner(dudm_j.vector())
                    )

                if not ignore_action_dJdu_d2udm2:

                    D2JDm2[i,j] += (
                        assemble(d2Fdm2_ij
                         + action(action(self._d2Fdu2, dudm_j), dudm_i))
                        + assemble(d2Fdudm_j)*(dudm_i.vector())
                        + assemble(d2Fdudm_i)*(dudm_j.vector())
                        ).inner(self._z.vector())

            for j in range(i+1,self._n):
                D2JDm2[j,i] = D2JDm2[i,j]

        return DJDm, D2JDm2


    def compute_DJDm_D2JDm2_method_direct(self, ignore_action_dJdu_d2udm2=False):
        '''Compute first order and second order derivatives of cost with respect
        to model parameters. This computation relies on the direct method.'''

        DJDm = np.zeros((self._n,), float)
        D2JDm2 = np.zeros((self._n,self._n), float)

        assembled_dJdu = assemble(self._dJdu)
        assembled_d2Jdu2 = assemble(self._d2Jdu2)

        if self._is_missing_dudm:
            self._compute_dudm()

        if self._is_missing_d2udm2 and not ignore_action_dJdu_d2udm2:
            self._compute_d2udm2()

        for i, (d2Jdudm_i, dJdm_i, dudm_i) in enumerate(
          zip(self._d2Jdudm, self._dJdm, self._dudm)):

            # DJDm[i] = assemble(dJdm_i + action(self._dJdu, dudm_i)) # can fail
            DJDm[i] = assemble(dJdm_i) + assembled_dJdu.inner(dudm_i.vector())
            print("computing DJDm(i) for i= ", i)

            for j, (d2Jdm2_ij, d2udm2_ij, d2Jdudm_j, dudm_j) in enumerate(
              zip(self._d2Jdm2[i], self._d2udm2[i], self._d2Jdudm[i:], self._dudm[i:]),
              start=i):

                D2JDm2[i,j] = (
                    assemble(d2Jdm2_ij)
                    + (assembled_d2Jdu2*dudm_j.vector()).inner(dudm_i.vector())
                    + assemble(d2Jdudm_j).inner(dudm_i.vector())
                    + assemble(d2Jdudm_i).inner(dudm_j.vector()))

                if not ignore_action_dJdu_d2udm2:
                    D2JDm2[i,j] += assembled_dJdu.inner(d2udm2_ij.vector())

            for j in range(i+1,self._n):
                D2JDm2[j,i] = D2JDm2[i,j]

        return DJDm, D2JDm2


    @staticmethod
    def _compute_dm_method_newton(DJDm, D2JDm2):
        '''Compute model parameter change using the Newton method.'''
        dm = linalg.solve(D2JDm2, -DJDm, sym_pos=False)
        return dm

    @staticmethod
    def _compute_dm_method_gradient(DJDm, D2JDm2):
        '''Compute model parameter change using gradient-descent and line-search.'''
        dm = -DJDm
        d2J = D2JDm2.dot(dm).dot(dm)
        if d2J > 0:
            dm *= (dm.dot(dm)/d2J)
        return dm


    def _factory_compute_dm(self):
        '''Factory for computing the change in model parameters.'''

        if self.parameters_inverse_solver['solver_method'] == self.ITER_METHOD_NEWTON:

            def compute_dm(DJDm, D2JDm2):
                try:
                    dm = self._compute_dm_method_newton(DJDm, D2JDm2)
                    if D2JDm2.dot(dm).dot(dm) < 0.0:
                        logger.warning('`D2JDm2[dm]` is not positive, i.e. '
                            'model cost is not convex in direction of `dm`. '
                            '(Could be okey for a saddle-point problem.)')

                except linalg.LinAlgError:
                    logger.warning('Could not compute `dm` using '
                        'Newton method; trying gradient descent.')
                    dm = self._compute_dm_method_gradient(DJDm, D2JDm2)

                return dm

        elif self.parameters_inverse_solver['solver_method'] == self.ITER_METHOD_GRADIENT:
            compute_dm = self._compute_dm_method_gradient

        else:
            raise RuntimeError('parameters_inverse_solver[\'solver_method\']: '
                "\"{self.parameters_inverse_solver['solver_method']}\" "
                'is not a valid parameter.')

        return compute_dm


    def _factory_constrain_dm(self):
        '''Factory for constraining the change in the model parameters.'''

        delta_max = self.parameters_inverse_solver['model_parameter_delta_max']
        delta_nrm = self.parameters_inverse_solver['model_parameter_delta_nrm']

        if delta_max:
            if not (0.0 < delta_max < np.inf):
                raise ValueError('Expected inverse solver parameter: '
                    '`model_parameter_delta_max` to be positive')

            if delta_nrm == 'L2':
                def constrain_dm(dm, m):
                    '''Constrain the change (max L2 change).'''
                    delta = linalg.norm(dm)/linalg.norm(m)
                    if delta > delta_max:
                        dm *= delta_max/delta
                    return dm

            elif delta_nrm == 'L1':
                def constrain_dm(dm, m):
                    '''Constrain the change (max L1 change).'''
                    delta = linalg.norm(dm,1)/linalg.norm(m,1)
                    if delta > delta_max:
                        dm *= delta_max/delta
                    return dm

        else:
            def constrain_dm(dm, m):
                '''Do not constrain the change.'''
                return dm

        return constrain_dm


    def minimize_cost_forall(self, observation_times=None,
        sensitivity_method='default', approximate_D2JDm2='default'):
        '''Minimize model cost (on average) over the observation times.

        Returns
        -------
        m : numpy.ndarray (1D) of float's
            Optimized model parameters over all observation times.

        '''

        if observation_times is not None:
            self.assign_observation_times(observation_times)

        logging_level = logger.getEffectiveLevel()
        params = self.parameters_inverse_solver

        atol = params['absolute_tolerance']
        rtol = params['relative_tolerance']

        max_iterations = params['maximum_iterations']
        max_diverged = params['maximum_diverged_iterations']

        if sensitivity_method == 'default':
            sensitivity_method = params['sensitivity_method']

        if approximate_D2JDm2 == 'default':
            approximate_D2JDm2 = params['approximate_D2JDm2']

        forward_solver = self.factory_forward_solver(
            None, sensitivity_method, approximate_D2JDm2)

        compute_dm = self._factory_compute_dm()
        constrain_dm = self._factory_constrain_dm()
        m = self.view_model_parameters_as_array()

        dm_old = None
        DJDm_old = None
        D2JDm2_old = None
        norm_DJDm_old = np.inf

        num_diverged = 0
        is_converged = False

        cost_u_for_iteration = []
        cost_f_for_iteration = []
        cost_total_for_iteration = []

        cost_for_each_iteration = [cost_u_for_iteration, cost_f_for_iteration, cost_total_for_iteration]

        param_for_each_iteration = [0 for i in range(self._n)]

        for i in range(self._n):
            param_for_each_iteration[i] = [m[i],]

        for num_iterations in range(max_iterations):

            try:
                DJDm, D2JDm2 = forward_solver()
            except:
                print("No convergence in nonlinear FEM solver")
                for i in cost_for_each_iteration:
                    i.append(0) # The cost in the last iteration is set NULL
                return m, (num_iterations + 1, is_converged,\
                            cost_for_each_iteration, param_for_each_iteration)

            dm = compute_dm(DJDm, D2JDm2)
            dm = constrain_dm(dm, m)

            norm_dm = linalg.norm(dm)
            norm_DJDm = linalg.norm(DJDm)

            dJ = DJDm.dot(dm)
            d2J = D2JDm2.dot(dm).dot(dm)

            DJDm_dm = dJ/norm_dm
            D2JDm2_dm = d2J/norm_dm**2

            dircos_dm = -DJDm_dm/norm_DJDm
            norm_dm_res = -DJDm_dm/D2JDm2_dm

            dm_param = dm[range(self._n)]
            m_param = m[range(self._n)]

            is_decreasing = (DJDm_dm < 0.0)
            is_pathconvex = (D2JDm2_dm > 0.0)
            is_converging = (norm_DJDm < norm_DJDm_old)

            cost_for_each_iteration [0].append(assemble(self._Ju))
            cost_for_each_iteration [1].append(assemble(self._Jf))
            cost_for_each_iteration [2].append(assemble(self._J))

            if logging_level <= logging.INFO:
                print('\n\n*** Summary of Iteration \n\n')
                print(num_iterations, '/', max_iterations)
                print('  "norm(DJDm_old)"     :', norm_DJDm_old)
                print('  "norm(DJDm)"	        :', norm_DJDm, '\n')
                print('  "DJDm[dm]"           :', DJDm_dm, '\n')
                print('  "D2JDm2[dm]"         :', D2JDm2_dm, '\n')
                print('  "model param.,  m"	:', m, '\n')
                print('  "model param.,  dm"	:', dm, '\n')
                print('  "actual model param.,  m"	:', m+dm, '\n')
                # m_sum = m + dm
                # m_sum[0:4] = np.abs(m_sum[0:4])
                # print('  "actual model param., sum (m)"	:', m_sum, '\n')
                print('  "direction cosine"   :', dircos_dm, '\n')
                print('  "residual est., dm"  :', norm_dm_res, '\n')
                print('  "is cost convex"     :', is_pathconvex, '\n')
                print('  "is cost decreasing" :', is_decreasing, '\n')
                print('  "is cost converging" :', is_converging, '\n')
                print('  "the cost of the last observation time" :', cost_for_each_iteration [2][-1], '\n')
					# '  {"det(F)"}				:', assemble(det(self._F)))
                # print(
					# num_iterations, '/', max_iterations, '\n'

					# '
					 # )

            if logging_level <= logging.DEBUG and dm_old is not None:

                # Check if the directional second derivative of `J` can estimate the
                # change in gradient of `J` between previous and current iterations.

                dDJDm_exa = DJDm - DJDm_old
                dDJDm_est = (D2JDm2_old + D2JDm2).dot(dm_old) * 0.5
                # err_dDJDm = linalg.norm(dDJDm_est-dDJDm_exa)/linalg.norm(dDJDm_exa)

                print('  {"estimated change in DJDm, i.e. D2JDm2[dm_old]":s} : {dDJDm_est}\n'
                      '  {"actual change in DJDm, i.e. DJDm_new-DJDm_old":s} : {dDJDm_exa}\n'
                      , flush=True)

            if np.all(np.abs(dm) < np.abs(m)*rtol + atol):
            # if np.all(np.abs(dm_param) < np.abs(m_param)*rtol + atol):
                is_converged = True
                break

            if norm_DJDm > norm_DJDm_old:

                num_diverged += 1

                if num_diverged < max_diverged:
                    logger.warning('Model cost diverged '
                        '{num_diverged} time(s).')
                else:
                    logger.error('Model cost diverged maximum '
                        'number of times ({max_diverged}).')
                    break

                # if approximate_D2JDm2 and num_iterations < max_iterations:
                    # logger.warning('Continue without approximating `D2JDm2 = False`.')

                    # approximate_D2JDm2 = True
                    # num_diverged = 0 # reset
                    # dm = -dm_old # backtrace

                    # forward_solver = self.factory_forward_solver(
                        # None, sensitivity_method, approximate_D2JDm2)

            m += dm

            for i in range(self._n):
                param_for_each_iteration[i].append(m[i])
            # m[0:4] = np.abs(m[0:4]) #if m's elements are negative, energy density is negtive as well and so the PDE can't be solved
            # for i in range(4):
                # j = 0
                # while ((m[i] + dm[i]*10**(-j)) < 0): j += 1
                # m[i] += dm[i]*10**(-j)

            for m_i, m_i_new in zip(self._m, m):
                m_i.assign(m_i_new)

            dm_old = dm
            DJDm_old = DJDm
            D2JDm2_old = D2JDm2
            norm_DJDm_old = norm_DJDm

        else:
            logger.warning('Inverse solver failed to converge '
                'after {max_iterations+1} iterations.')

        self._cumsum_DJDm = DJDm
        self._cumsum_D2JDm2 = D2JDm2
        self._is_converged = is_converged

        if not is_converged and params['error_on_nonconvergence']:
            input('Inverse solver did not converge. Do you like to continue ?')

        return m, (num_iterations+1, is_converged, cost_for_each_iteration, param_for_each_iteration)


    def minimize_cost_foreach(self, observation_times=None,
        sensitivity_method='default', approximate_D2JDm2='default'):
        '''Minimize cost for each of the observation times.

        Returns
        -------
        m : list of numpy.ndarray's (1D) of float's
            Optimized model parameters for each observation time.

        '''

        observation_times = \
            self._observation_times_getdefault(observation_times)

        m = []

        for t in observation_times:
            m_t, (n, b) = self.minimize_cost_forall(
                t, sensitivity_method, approximate_D2JDm2)
            m.append(m_t)

        if m: # usually more convenient to use first index for parameter
            m = [np.ascontiguousarray(m_i) for m_i in np.stack(m, 0).T]

        # Assign old observation times and reset checkpoints
        self.assign_observation_times(observation_times)

        return m, (n, b)


    def factory_forward_solver(self, observation_times=None,
        sensitivity_method='default', approximate_D2JDm2='default'):
        '''Factory for solving the forward problem.'''

        observation_times = \
            self._observation_times_getdefault(observation_times)

        compute_DJDm_D2JDm2 = \
            self.factory_compute_DJDm_D2JDm2(sensitivity_method, approximate_D2JDm2)

        def forward_solver():
            '''Solve forward problem.

            Returns
            -------
            DJDm : numpy.ndarray (1D) of float's
                Cumulative gradient of the model cost over the observation times.
            D2JDm2 : numpy.ndarray (2D) of float's
                Cumulative hessian of the model cost over the observation times.

            '''
            print(self._u([0, 20]))
            DJDm = 0.0
            D2JDm2 = 0.0

            self._u.vector()[:] = 0.0
            for t in observation_times:

                n, b = self.solve_nonlinear_problem(t)

                dDJDm, dD2JDm2 = compute_DJDm_D2JDm2()

                if not b and not np.isfinite(dD2JDm2).all():
                    raise RuntimeError('Model cost sensitivities '
                        'at time t={t} have non-finite values.')

                DJDm += dDJDm
                D2JDm2 += dD2JDm2
            # self._u.vector()[:] = 0.0

            return DJDm, D2JDm2



        return forward_solver


    def solve_forward_problem(self, observation_times=None,
        sensitivity_method='default', approximate_D2JDm2='default'):
        '''Solve forward problem.

        Returns
        -------
        DJDm : numpy.ndarray (1D) of float's
            Cumulative gradient of the model cost over the observation times.
        D2JDm2 : numpy.ndarray (2D) of float's
            Cumulative hessian of the model cost over the observation times.

        '''

        DJDm, D2JDm2 = self.factory_forward_solver(
            observation_times, sensitivity_method, approximate_D2JDm2)()

        if observation_times is None:
            self._cumsum_DJDm = DJDm.copy()
            self._cumsum_D2JDm2 = D2JDm2.copy()

        return DJDm, D2JDm2


    @staticmethod
    def _sum_actions(dfdv, dv):
        '''Try to compute the sum of actions of pairs dfdv_i and dv_i. If an action
        can not be computed due to `IndexError` exception, assume the action is zero.
        '''
        dfdv_dv = 0
        for dfdv_i, dv_i in zip(dfdv, dv):
            try:
                dfdv_dv_i = action(dfdv_i, dv_i)
            except IndexError:
                pass # assume `dfdv_dv_i` is zero
            else:
                dfdv_dv += dfdv_dv_i
        return dfdv_dv


    def factory_observe_DmDv(self, vars, dv=None, ignore_dFdv=False, ignore_dJdv=False):
        '''Factory for computing model parameter sensitivities with respect to
        variabes for different observation times.

        Important
        ---------
        Note that the sensitivities are local quantities, i.e. they represent the
        sensitivities due to local changes in the variables at a given time. The
        total (or cumulative) sensitivities can be obtained by summing all local
        sensitivities over the observation times.

        Parameters
        ----------
        vars : sequence of dolfin.Constant's or dolfin.Function's
            Obtain model parameter sensitivities with respect to each of the
            variables in `vars`.

        dv : sequence of float's, int's, dolfin.Constant's, or dolfin.Function's
            Compute the directional (Getaux) sensitivity in the direction of `dv`.

            Note, in case `vars` are functions only the directional sensitivity
            can be computed. The one exception when non-directional sensitvities
            can be computed is when `vars` consists of a single function and the
            parameter `ignore_dFdv` can be assumed to be `True`.

            Note, if `dv` consists of `dolfin.Constant`s or `dolfin.Function`s,
            then such type objects can be updated externally at any time.

        ignore_dFdv : bool
            Whether to ignore the partial interaction of `vars` with the weak form.

        ignore_dJdv : bool
            Whether to ignore the partial interaction of `vars` with the model cost.

        Returns
        -------
        observe_DmDv() : function
            Compute local model parameter sensitivities with respect to variables.

        '''

        if ignore_dFdv and ignore_dJdv:
            logger.warning('Parameters `ignore_dFdv` and `ignore_dJdv` can not '
                           'both be `True`; at least one should be `False`.')

        not_ignore_dFdv = not ignore_dFdv
        not_ignore_dJdv = not ignore_dJdv

        vars = list_variables_from_iterable(vars, (Constant,Function))

        if all(isinstance(v_i, Constant) for v_i in vars):

            if not_ignore_dFdv: dFdv = tuple(diff(self._F, v_i) for v_i in vars)
            if not_ignore_dJdv: dJdv = tuple(diff(self._J, v_i) for v_i in vars)

            if dv is not None:

                if isinstance(dv, np.ndarray):
                    dv = dv.tolist()

                dv = list_variables_from_iterable(dv, (float,int,Constant))

                if len(dv) != len(vars):
                    raise TypeError('Expected parameter `dv` to contain {len(vars)} '
                                    '`dolfin.Constant`(s) or `float`(s) but instead the '
                                    'number of such types found in `dv` was {len(dv)}.')

                if not_ignore_dFdv: dFdv = (sum(dv_i*dFdv_i for dFdv_i,dv_i in zip(dFdv,dv)),)
                if not_ignore_dJdv: dJdv = (sum(dv_i*dJdv_i for dJdv_i,dv_i in zip(dJdv,dv)),)

        elif all(isinstance(v_i, Function) for v_i in vars):

            if dv is None:

                if not_ignore_dFdv:
                    raise RuntimeError('Unable to compute the sensitivities with respect '
                                       'to a `dolfin.Function`. The sensitivities can only '
                                       'be computed provided the paramter `ignore_dFdv` is '
                                       '`True`. Alternativelly, the directional sensitivity '
                                       'can be computed by providing parameter `dv`.')

                if len(vars) != 1:
                    raise RuntimeError('Parameter `vars` should contain a single '
                                       '`dolfin.Function` if no `dv` is provided.')

            if not_ignore_dFdv: dFdv = tuple(derivative(self._F, v_i) for v_i in vars)
            if not_ignore_dJdv: dJdv = tuple(derivative(self._J, v_i) for v_i in vars)

            if dv is not None:

                dv = list_variables_from_iterable(dv, Function)

                if len(dv) != len(vars):
                    raise TypeError('Expected parameter `dv` to contain {len(vars)} '
                                    '`dolfin.Function`(s) but instead the number of '
                                    'such type objects found in `dv` was {len(dv)}.')

                if not_ignore_dFdv:
                    dFdv_dv = self._sum_actions(dFdv, dv)
                    if dFdv_dv != 0: dFdv = (dFdv_dv,)
                    else: not_ignore_dFdv = False

                if not_ignore_dJdv:
                    dJdv_dv = self._sum_actions(dJdv, dv)
                    if dJdv_dv != 0: dJdv = (dJdv_dv,)
                    else: not_ignore_dJdv = False

        else:
            raise TypeError('Parameter `vars` must exclusively contain either '
                            '`dolfin.Constant`(s) or `dolfin.Function`(s).')

        if not_ignore_dFdv:
            compute_dudv_d2udmdv, dudv, d2udmdv = \
                self._factory_compute_dudv_d2udmdv(dFdv)

        if not_ignore_dJdv:
            d2Jdudv = tuple(derivative(dJdv_i, self._u) for dJdv_i in dJdv)
            d2Jdmdv = tuple(tuple(diff(dJdv_i, m_j) for m_j in self._m) for dJdv_i in dJdv)

        if not_ignore_dJdv and (not_ignore_dFdv is False) \
            and isinstance(vars[0], Function) and dv is None:

            def compute_D2JDmDv(i, j):
                J = assemble(d2Jdmdv[i][j]) \
                    + assemble(d2Jdudv[i])*self._dudm[j].vector()
                return J # numpy.ndarray of float's

        else:

            def compute_D2JDmDv(i, j):
                J = 0
                if not_ignore_dJdv:
                    J += assemble(d2Jdmdv[i][j])
                    J += assemble(d2Jdudv[i]).inner(self._dudm[j].vector())
                if not_ignore_dFdv:
                    J += assemble(self._dJdu).inner(d2udmdv[i][j].vector())
                    J += assemble(self._d2Jdudm[j]).inner(dudv[i].vector())
                    J +=(assemble(self._d2Jdu2)*dudv[i].vector()).inner(self._dudm[j].vector())
                return J # float

        if not (not_ignore_dFdv or not_ignore_dJdv):
            logger.warning('All derivatives will be zero.')

        D2JDm2 = None # to be the alias for the current `self._cumsum_D2JDm2`
        inv_D2JDm2 = None # to compute `linalg.inv(D2JDm2)` if `D2JDm2` changes

        def observe_DmDv(t=None):
            '''Compute the local model parameter sensitivities with respect to
            variables at a given observation time.

            Parameters
            ----------
            t : float or int or None (optional)
                Time at which to compute the model parameter sensitivities.
                If `t` is `None`, `t` defaults to current time.

            Returns
            -------
            DmDv : numpy.ndarray (1D or 2D) of float's
                Local model parameter sensitivities with respect to variables at
                observation time `t`.

            Notes
            -----
            If `vars` contains `dolfin.Constant`s and `dv` is `None`, the array
            `DmDv` will have shape `(NUM_MODEL_PARAMETERS, NUM_VARIABLES)`. If
            `vars` contains `dolfin.Function`s (or `dolfin.Constant`s) and `dv`
            contains `dolfin.Function`s (or `dolfin.Constant`s), array `DmDv`
            will have shape `(NUM_MODEL_PARAMETERS,)`. If `vars` has a single
            `dolfin.Function` and `dv` is `None`, array `DmDv` will have shape
            `(NUM_MODEL_PARAMETERS, NUM_BASIS_FUNCTIONS)`.

            '''

            nonlocal D2JDm2
            nonlocal inv_D2JDm2

            if t is None:
                if self._t is None:
                    raise RuntimeError('Can not specify time parameter '
                                       '`t = None`. Require at least once '
                                       'to solve for an explicit time `t`.')
                t = self._t

            if not self._is_converged:
                logger.warning('Inverse solver is not converged.')

            if self._cumsum_D2JDm2 is None:
                logger.info('Solving forward problem for `D2JDm2`.')
                self.solve_forward_problem() # -> self._cumsum_D2JDm2

            if D2JDm2 is not self._cumsum_D2JDm2:
                D2JDm2 = self._cumsum_D2JDm2

                try: # could be ill-conditioned
                    inv_D2JDm2 = linalg.inv(-D2JDm2)
                except linalg.LinAlgError:
                    inv_D2JDm2 = linalg.pinv(-D2JDm2)

            if t != self._t:
                self.solve_nonlinear_problem(t)

            if self._is_missing_dudm:
                self._compute_dudm()

            if not_ignore_dFdv:
                compute_dudv_d2udmdv()

            DmDv = []

            for i in range(len(vars) if dv is None else 1):
                D2JDmDv_i = [compute_D2JDmDv(i,j) for j in range(self._n)]
                DmDv.append(inv_D2JDm2.dot(np.array(D2JDmDv_i, dtype=float)))

            # Reorder indexing into `DmDv`:
            # current indexing: DmDv[i_dv][j_dm]
            # new indexing: DmDv[i_dm][j_dv]

            DmDv = np.array([[DmDv[j_dv][i_dm]
                for j_dv in range(len(DmDv))]
                for i_dm in range(self._n)])

            if dv is not None or DmDv.ndim > 2:
                DmDv = DmDv.squeeze(axis=1)

            # # Model parameter residual
            # dm_residual = inv_D2JDm2.dot(DJDm).tolist()
            # logger.info('dm_residual: ' + repr(dm_residual))

            return DmDv

        return observe_DmDv


    def factory_observe_DuDv(self, vars, DmDv, dv=None, ignore_dFdv=False):
        '''Factory for computing the displacement sensitivities with respect
        to variables for different observation times.

        Notes
        -----
        The derivative `DuDv` at time `t` is computed according to this rule:
            Du/Dv_i = du/dv_i + Dm_j/Dv_i du/dm_j

        Note, `DmDv` are the cumulative sensitivities that are obtained by
        summing the local sensitivities over all observation times.

        Parameters
        ----------
        vars : sequence of dolfin.Constant('s) or `dolfin.Function(`s)
            Obtain displacement sensitivities with respect to each of the
            variables in `vars`.

        DmDv : numpy.ndarray (1D or 2D) of float's
            The cumulative sensitivities of the model parameters with respect to
            variables. The length of `DmDv` in the first dimension must always
            equal the number of model parameters. The length of `DmDv` in its
            second dimension must equal the length of `vars` if `dv` is `None`;
            otherwise, if `dv` is not `None`, then either the length of `DmDv`
            in its second dimension must be `1` or `DmDv` must be a 1D array.

        dv : sequence of float's, int's, dolfin.Constant's, or dolfin.Function's
            Compute the directional (Getaux) sensitivity in the direction of `dv`.

            Note, in case `vars` are functions only the directional sensitivity
            can be computed. The one exception when non-directional sensitvities
            can be computed is when `vars` consists of a single function and the
            parameter `ignore_dFdv` can be assumed to be `True`.

            Note, if `dv` consists of `dolfin.Constant`s or `dolfin.Function`s,
            then such type objects can be updated externally at any time.

        ignore_dFdv : bool
            Whether to ignore the partial interaction of `vars` with the weak form.

        Returns
        -------
        observe_DuDv() : function
            Compute the displacement sensitivities with respect to variables
            `vars` for a given time. If `dv`is specified, `observe_DuDv` will
            compute the directional (Getaux) derivative in the `dv` direction.

        '''

        not_ignore_dFdv = not ignore_dFdv

        vars = list_variables_from_iterable(vars, (Constant,Function))

        if dv is None:
            DuDv = tuple(Function(self._V) for _ in vars)
        else:
            DuDv = (Function(self._V),)

        if not isinstance(DmDv, np.ndarray) \
           or (DmDv.ndim == 1 and dv is None) \
           or (DmDv.ndim == 2 and DmDv.shape[1] != len(vars)) \
           or len(DmDv) != self._n:
            raise TypeError('Expected `DmDv` to be a 2D `numpy.ndarray` whose '
                            'size in the first dimension equals the number of '
                            'model parameters ({self._n}) and whose size in the '
                            'second dimension equals the number of derivatives '
                            '({len(DuDv)}). Alternatively, `DmDv`, can be a 1D '
                            '`numpy.ndarray`, but `dv` must be provided.')

        if not_ignore_dFdv:

            if all(isinstance(v_i, Constant) for v_i in vars):
                dFdv = tuple(diff(self._F, v_i) for v_i in vars)

                if dv is not None:

                    if isinstance(dv, np.ndarray):
                        dv = dv.tolist()

                    dv = list_variables_from_iterable(dv, (float,int,Constant))

                    if len(dv) != len(vars):
                        raise TypeError('Expected parameter `dv` to contain {len(vars)} '
                                        '`dolfin.Constant`(s) or `float`(s) but instead the '
                                        'number of such types found in `dv` was {len(dv)}.')

                    dFdv = (sum(dv_i * dFdv_i for dFdv_i, dv_i in zip(dFdv, dv)),)

            elif all(isinstance(v_i, Function) for v_i in vars):

                if dv is None:
                    raise RuntimeError('Unable to compute the sensitivities with '
                                       'respect to a `dolfin.Function`. Only the '
                                       'directional sensitivity can be computed '
                                       'by providing parameter `dv`.')

                dv = list_variables_from_iterable(dv, Function)

                if len(dv) != len(vars):
                    raise TypeError('Expected parameter `dv` to contain {len(vars)} '
                                    '`dolfin.Function`(s) but instead the number of '
                                    'such type objects found in `dv` was {len(dv)}.')

                dFdv = (derivative(self._F, v_i) for v_i in vars)

                dFdv_dv = self._sum_actions(dFdv, dv)
                if dFdv_dv != 0: dFdv = (dFdv_dv,)
                else: not_ignore_dFdv = False

            else:
                raise TypeError('Parameter `vars` must exclusively contain either '
                                '`dolfin.Constant`(s) or `dolfin.Function`(s).')

            if not_ignore_dFdv: # could have become `False`
                dudv = tuple(Function(self._V) for _ in dFdv)
                def compute_dudv(): self._compute_dudv(dFdv, dudv)

        def observe_DuDv(t=None, copy=True):
            '''Compute the displacement sensitivities with respect to variables
            at observation time.'''

            if t is None:
                if self._t is None:
                    raise RuntimeError('Can not specify time parameter '
                                       '`t = None`. Require at least once '
                                       'to solve for an explicit time `t`.')
                t = self._t

            if not self._is_converged:
                logger.warning('Inverse solver is not converged.')

            if t != self._t:
                self.solve_nonlinear_problem(t)

            if self._is_missing_dudm:
                self._compute_dudm()

            if DmDv.ndim == 1: # and dv is not None
                dudm_DmDv = (sum(dudm_j * DmDv_ij for dudm_j, DmDv_ij
                    in zip(self._dudm, DmDv)),)
            elif dv is None: # and DmDv.ndim == 2
                dudm_DmDv = (sum(dudm_j * DmDv_ij for dudm_j, DmDv_ij
                    in zip(self._dudm, DmDv_i)) for DmDv_i in DmDv.T)
            else: # dv is not None and DmDv.ndim == 2
                dudm_DmDv = (sum(dudm_j * DmDv_ij for dudm_j, DmDv_ij
                    in zip(self._dudm, DmDv.dot(dv))),)

            if not_ignore_dFdv:
                compute_dudv()
                for DuDv_i, dudm_DmDv_i, dudv_i in zip(DuDv, dudm_DmDv, dudv):
                    DuDv_i.assign(dudm_DmDv_i + dudv_i)
            else:
                for DuDv_i, dudm_DmDv_i in zip(DuDv, dudm_DmDv):
                    DuDv_i.assign(dudm_DmDv_i)

            if copy:
                if dv is None:
                    return [DuDv_i.copy(True) for DuDv_i in DuDv]
                else:
                    return DuDv[0].copy(True)
            else:
                if dv is None:
                    return DuDv
                else:
                    return DuDv[0]

        return observe_DuDv


    def factory_observe_DfDv(self, forms, vars, DmDv, dv=None, ignore_dFdv=False):
        '''Factory for computing the total derivative of a form with respect to
        variables taking into account the stationarity of the model cost.

        Note, `DmDv` are the cumulative sensitivities that are obtained by
        summing the local sensitivities over all observation times.

        Parameters
        ----------
        vars : sequence of dolfin.Constant('s) or `dolfin.Function(`s)
            Obtain displacement sensitivities with respect to each of the
            variables in `vars`.

        DmDv : numpy.ndarray (1D or 2D) of float's
            The cumulative sensitivities of the model parameters with respect to
            variables. The length of `DmDv` in the first dimension must always
            equal the number of model parameters. The length of `DmDv` in its
            second dimension must equal the length of `vars` if `dv` is `None`;
            otherwise, if `dv` is not `None`, then either the length of `DmDv`
            in its second dimension must be `1` or `DmDv` must be a 1D array.

        dv : sequence of float's, int's, dolfin.Constant's, or dolfin.Function's
            Compute the directional (Getaux) sensitivity in the direction of `dv`.

            Note, in case `vars` are functions only the directional sensitivity
            can be computed. The one exception when non-directional sensitvities
            can be computed is when `vars` consists of a single function and the
            parameter `ignore_dFdv` can be assumed to be `True`.

        ignore_dFdv : bool
            Whether to ignore the partial interaction of `vars` with the weak form.

        Returns
        -------
        observe_DfDv() : function
            Compute the sensitivities of forms with respect to variables `vars`
            for at a given time. If `dv` is specified, `observe_DfDv` will
            compute the directional (Getaux) derivative in the `dv` direction.

        '''

        if isinstance(forms, (list,tuple)):
            flag_return_iterable_DfDv = True
            forms_count = len(forms)
        else:
            flag_return_iterable_DfDv = False
            forms = (forms,)
            forms_count = 1

        vars = list_variables_from_iterable(vars, (Constant,Function))

        if not all(isinstance(f_i, ufl.Form) for f_i in forms):
            raise TypeError('Parameter `forms` must be either a `dolfin.Form` '
                            'or a `list` or `tuple` of `dolfin.Form`s.')

        dfdu = [derivative(f_i, self._u) for f_i in forms]
        dfdm = [tuple(diff(f_i, m_j) for m_j in self._m) for f_i in forms]

        if all(isinstance(v_i, Constant) for v_i in vars):
            dfdv = [tuple(diff(f_i, v_j) for v_j in vars) for f_i in forms]

            if dv is not None:

                if isinstance(dv, np.ndarray):
                    dv = dv.tolist()

                dv = list_variables_from_iterable(dv, (float,int,Constant))

                if len(dv) != len(vars):
                    raise TypeError('Expected parameter `dv` to contain {len(vars)} '
                                    '`dolfin.Constant`(s) or `float`(s) but instead the '
                                    'number of such types found in `dv` was {len(dv)}.')

                for k in range(forms_count):
                    dfdv[k] = (sum(dv_i*dfdv_ki for dfdv_ki, dv_i in zip(dfdv[k], dv)),)

        elif all(isinstance(v_i, Function) for v_i in vars):
            dfdv = [tuple(derivative(f_i, v_j) for v_j in vars) for f_i in forms]

            if dv is not None:

                dv = list_variables_from_iterable(dv, Function)

                if len(dv) != len(vars):
                    raise TypeError('Expected parameter `dv` to contain {len(vars)} '
                                    '`dolfin.Function`(s) but instead the number of '
                                    'such type objects found in `dv` was {len(dv)}.')

                for k in range(forms_count):
                    dfdv[k] = (self._sum_actions(dfdv[k], dv),)

        else:
            raise TypeError('Parameter `vars` must exclusively contain '
                'either `dolfin.Constant`(s) or `dolfin.Function`(s).')

        deriv_count = len(vars) if dv is None else 1

        observe_DuDv = self.factory_observe_DuDv(
            vars, DmDv, dv, ignore_dFdv)

        assert isinstance(DmDv, np.ndarray)
        assert (DmDv.ndim == 1 and dv is not None) or \
               (DmDv.ndim == 2 and DmDv.shape[1] == len(vars))
        assert len(DmDv) == self._n

        def observe_DfDv(t=None):
            '''Compute the sensitivities of an integral expression with respect
            to variables at a given observation time.

            Returns
            -------
            dfdv : tuple of tuples of float's
                The items are index as dfdv[i_f][j_v].

            '''

            if t is None:
                if self._t is None:
                    raise RuntimeError('Can not specify time parameter '
                                       '`t = None`. Require at least once '
                                       'to solve for an explicit time `t`.')
                t = self._t

            DuDv = observe_DuDv(t, copy=False)
            assert t == self._t # solved for `t`

            if isinstance(DuDv, (list,tuple)):
                assert deriv_count == len(DuDv)
                dfdu_DuDv = tuple(tuple(action(dfdu_i, DuDv_j)
                    for DuDv_j in DuDv) for dfdu_i in dfdu)
            else: # isinstance(DuDv, Function)
                assert deriv_count == 1
                dfdu_DuDv = tuple((action(dfdu_i, DuDv),) for dfdu_i in dfdu)

            if DmDv.ndim == 1: # and dv is not None
                dfdm_DmDv = tuple((sum(DmDv[k] * dfdm_i[k]
                    for k in range(self._n)),) for dfdm_i in dfdm)
            elif dv is None: # and DmDv.ndim == 2
                dfdm_DmDv = tuple(tuple(
                    sum(DmDv_j[k] * dfdm_i[k] for k in range(self._n))
                    for DmDv_j in DmDv.T) for dfdm_i in dfdm)
            else: # dv is not None and DmDv.ndim == 2
                dfdm_DmDv = tuple((sum(DmDv[k].dot(dv) * dfdm_i[k]
                    for k in range(self._n)),) for dfdm_i in dfdm)

            DfDv = [tuple(
                assemble(dfdu_DuDv[i][j] + dfdm_DmDv[i][j] + dfdv[i][j])
                for j in range(deriv_count)) for i in range(forms_count)]

            if flag_return_iterable_DfDv:
                return DfDv
            else:
                return DfDv[0]

        return observe_DfDv


    def observe_dudm(self, t=None, copy=True):
        '''Compute displacement derivative with respect to model parameters
        at observation time. Note, this is a partial derivative.

        Note
        ----
        The derivative is defined as: du/dm_i := inv(dF/du)*(-dF/dm_i)

        Returns
        -------
        dudm : numpy.ndarray (2D) of float's
            Displacement sensitivities with respect to model parameters at
            observation time `t`.

        '''

        if t is None:
            if self._t is None:
                raise RuntimeError('Can not specify time parameter `t = None`. '
                    'Require at least once to solve for an explicit time `t`.')
            t = self._t

        if t != self._t:
            self.solve_nonlinear_problem(t)


        if self._is_missing_dudm:
            self._compute_dudm()

        if copy:
            return [dudm_i.copy(True) for dudm_i in self._dudm]
        else:
            return self._dudm


    def observe_dudm_eig(self, t=None):
        '''Compute displacement sensitivities with respect to the principal
        directions of the model parameter sensitivities.

        Returns
        -------
        dudm_eig : list of dolfin.Function's
            Displacement sensitivities with respect to model parameters at
            observation time `t` in the direction of eigenvalues of D2J/Dm2.

        dm_eigval : numpy.ndarray (1D)
            Array of eigenvalues of the Heassian of the model cost, D2J/Dm2

        dm_eigvec : numpy.ndarray (2D)
            Array of eigenvalues of the Heassian of the model cost, D2J/Dm2

        '''

        if t is None:
            if self._t is None:
                raise RuntimeError('Can not specify time parameter `t = None`. '
                    'Require at least once to solve for an explicit time `t`.')
            t = self._t

        if self._cumsum_D2JDm2 is None:
            self.solve_forward_problem()

        D2JDm2 = self._cumsum_D2JDm2.copy()

        dm_eigval, dm_eigvec = linalg.eigh(D2JDm2)
        dudm_eig = self.observe_dudm(t, copy=True)

        for i, dm in enumerate(dm_eigvec.T):
            tmp = sum(dudm_i * dm_i for dudm_i, dm_i in zip(self._dudm, dm))
            dudm_eig[i].assign(tmp)

        return dudm_eig, dm_eigval, dm_eigvec


    def observe_model_cost(self, observation_times=None,
        compute_gradient=False, sensitivity_method='default'):
        '''Compute the model cost `J` and the model cost gradient `DJDm`
        at each of the observation times for the current model parameters.

        Returns
        -------
        J_obs : list of float's
            Values of model costs for the observation times.
        DJDm_obs : list of numpy.ndarray's (2D) of float's
            Values of the model cost gradient for the observation times.

        '''

        J_obs = []
        DJDm_obs = []

        observation_times = \
            self._observation_times_getdefault(observation_times)

        if compute_gradient:
            compute_DJDm = self.factory_compute_DJDm(sensitivity_method)

        for t in observation_times:

            self.solve_nonlinear_problem(t)
            J_obs.append(assemble(self._J))
            print('cost of t=', t, 'is:', J_obs[-1])

            if compute_gradient:
                DJDm_obs.append(compute_DJDm())

        if compute_gradient:
            DJDm_obs = [np.ascontiguousarray(DJDm_i)
                for DJDm_i in np.stack(DJDm_obs, 1)]

        return J_obs, DJDm_obs

    def observe_model_cost_seperate(self, observation_times=None,
        compute_gradient=False, sensitivity_method='default'):
        '''Compute the model cost `J` and the model cost gradient `DJDm`
        at each of the observation times for the current model parameters.

        Returns
        -------
        J_obs : list of float's
            Values of model costs for the observation times.
        DJDm_obs : list of numpy.ndarray's (2D) of float's
            Values of the model cost gradient for the observation times.

        '''

        Ju_obs = []
        Jf_obs = []
        J_obs = []

        observation_times = \
            self._observation_times_getdefault(observation_times)

        for t in observation_times:

            self.solve_nonlinear_problem(t)
            J_obs.append(assemble(self._J))
            Ju_obs.append(assemble(self._Ju))
            Jf_obs.append(assemble(self._Jf))
            print('global cost of t=', t, 'is:', J_obs[-1])
            print('displacemnt cost of t=', t, 'is:', Ju_obs[-1])
            print('force cost of t=', t, 'is:', Jf_obs[-1])

        return J_obs, Ju_obs, Jf_obs


    def view_model_parameters(self):
        '''Return a view of the `model_parameters` where `dolfin.Constant`s are
        replaced with their `float` values. `model_parameters` can be nested.'''

        def extract(model_parameters):

            if hasattr(model_parameters, 'keys'):
                return {k : extract(model_parameters[k])
                    for k in model_parameters.keys()}

            elif isinstance(model_parameters, (list,tuple)):
                return [extract(v) for v in model_parameters]

            elif isinstance(model_parameters, Constant):
                return float(model_parameters.values())

            else:
                raise TypeError('Expected a `dolfin.Constant` type inside '
                    '`model_parameters` but instead got {model_parameters}.')

        return extract(self.model_parameters)


    def view_model_parameters_as_list(self):
        return [float(m_i.values()) for m_i in self._m]

    def view_model_parameters_as_array(self):
        return np.array(self.view_model_parameters_as_list())


    def view_cumsum_DJDm(self):
        if self._cumsum_DJDm is not None:
            return self._cumsum_DJDm.copy()
        else:
            logger.info('Solving forward problem for `DJDm`.')
            return self.solve_forward_problem()[0]

    def view_cumsum_D2JDm2(self):
        if self._cumsum_D2JDm2 is not None:
            return self._cumsum_D2JDm2.copy()
        else:
            logger.info('Solving forward problem for `D2JDm2`.')
            return self.solve_forward_problem()[1]

    def compute_condition_number(self):
        '''Condition number of `D2JDm2`.'''
        return np.linalg.cond(self.view_cumsum_D2JDm2())

    @property
    def m(self):
        '''Model parameters.'''
        return self._m

    @property
    def n(self):
        '''Number of model parameters.'''
        return self._n

    @property
    def t(self):
        '''Current observation time.'''
        return self._t

    @property
    def t_obs(self):
        '''All observation times.'''
        return self._observation_times

    @property
    def n_obs(self):
        '''All observation times.'''
        return len(self._observation_times)


    @property
    def model_parameters_listed(self):
        return list(self._m)

    @property
    def num_model_parameters(self):
        a = self._n
        return a

    @property
    def observation_time(self):
        return self._t

    @property
    def observation_times(self):
        return list(self._observation_times)

    @property
    def num_observation_times(self):
        return len(self._observation_times)


    def dir(self, attributes='public'):
        attributes = str(attributes)

        if attributes == 'public':
            return [a for a in dir(self) if not a.startswith('_')]

        elif attributes == 'private':
            return [a for a in dir(self) if a.startswith('_')]

        elif attributes == 'all':
            return dir(self)


def list_variables_from_iterable(iterable, valid_types, vars_list=None):
    '''Extract variables of needed type(s) from (a nested) iterable to a list.'''

    if vars_list is None:
        vars_list = []

    if isinstance(iterable, valid_types):
        vars_list.append(iterable)

    elif hasattr(iterable, 'keys'):
        for k in iterable.keys():
            list_variables_from_iterable(iterable[k], valid_types, vars_list)

    elif hasattr(iterable, 'index'):
        for iterable_i in iterable:
            list_variables_from_iterable(iterable_i, valid_types, vars_list)

    return vars_list
