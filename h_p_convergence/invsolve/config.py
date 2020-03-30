'''
invsolve/config.py

'''

parameters_inverse_solver = {
    'solver_method': 'newton', # 'newton' or 'gradient' (not good for saddle)
    'maximum_iterations': 50, # max iterations in the minimization of cost functional
    'maximum_diverged_iterations': 20, # max diverged iterations
    'model_parameter_delta_max': 0.5, # max relative change in model parameters
    'model_parameter_delta_nrm': 'L2', # 'L2' or `max` measure of the change in model parameters
    'absolute_tolerance': 1e-9, # for convergence
    'relative_tolerance': 1e-6, # for convergence defaul value : 1e-6
    'error_on_nonconvergence': True,
    'sensitivity_method': 'adjoint', # 'adjoint' or 'direct'
    'approximate_D2JDm2': True, # the approximation works unreasonably well
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
        'linear_solver': 'mumps',
        'maximum_iterations': 50,
        'preconditioner': 'default',
        'relative_tolerance': 1e-7,
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
        'relative_tolerance': 1e-9,
        'report': False,
        'sign': 'default',
        'solution_tolerance': 1e-9,
        },
    }

parameters_linear_solver = {
    'linear_solver': 'default',
    'preconditioner': 'default',
    'symmetric': True,
    'print_matrix': False,
    'print_rhs': False,
    'verbose': False,
    'report': False
    }
