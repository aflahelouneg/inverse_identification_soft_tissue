'''
Problem-specific configuration of solver parameters.

Notes
-----
* The gradient method of the inverse solver is not suitable when the inverse
  problem is a saddle-point problem, e.g. when the cost functional includes
  an equality constraint via the method of Lagrange multipliers.

'''

parameters_inverse_solver = {
    'solver_method': 'newton', # 'newton' or 'gradient'
    'sensitivity_method': 'adjoint', # 'adjoint' or 'direct'
    'maximum_iterations': 50,
    'maximum_divergences': 0,
    'absolute_tolerance': 1e-8,
    'relative_tolerance': 1e-6,
    'maximum_relative_change': None,
    'error_on_nonconvergence': False,
    }

parameters_nonlinear_solver = {
    'nonlinear_solver': 'snes', # 'newton', 'snes'
    'symmetric': True,
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
    'symmetric': True,
    'verbose': False,
    }
