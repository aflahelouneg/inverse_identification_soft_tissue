'''
Default solver parameters.

'''

import logging


# Assume the default logger
logger = logging.getLogger()


parameters_inverse_solver = {
    'solver_method': 'newton', # 'newton' or 'gradient'
    'sensitivity_method': 'adjoint', # 'adjoint' or 'direct'
    'maximum_iterations': 50,
    'maximum_divergences': 3,
    'absolute_tolerance': 1e-9,
    'relative_tolerance': 1e-6,
    'maximum_relative_change': None,
    'error_on_nonconvergence': False,
    'is_symmetric_form_dFdu': False,
    }

parameters_nonlinear_solver = {
    'nonlinear_solver': 'snes', # 'newton', 'snes'
    'symmetric': False,
    'print_matrix': False,
    'print_rhs': False,
    'newton_solver': {
        'absolute_tolerance': 1e-9,
        'convergence_criterion': 'residual',
        'error_on_nonconvergence': True,
        'linear_solver': 'lu',
        'maximum_iterations': 50,
        'preconditioner': 'default',
        'relative_tolerance': 1e-12,
        'relaxation_parameter': 1.0,
        'report': False,
        },
    'snes_solver' : {
        'absolute_tolerance': 1e-9,
        'error_on_nonconvergence': True,
        'line_search': 'bt', # 'basic' | 'bt'
        'linear_solver': 'lu',
        'maximum_iterations': 100,
        'maximum_residual_evaluations': 2000,
        'method': 'default',
        'preconditioner': 'default',
        'relative_tolerance': 1e-12,
        'report': False,
        'sign': 'default',
        'solution_tolerance': 1e-9,
        },
    }

parameters_linear_solver = {
    'report': False,
    'symmetric': False,
    'verbose': False,
    }
