'''
./utility_methods.py

Convenience functions for solving inverse problems.

'''

import ufl
import numpy
import dolfin


def find_key_indices(find_keys, in_keys):
    '''Use to find the indices of model parameter keys'''
    return [i for i, k in enumerate(in_keys) if k in find_keys]


def factory_update_noise_basis(lhs, initializer=lambda:None):
    '''Wrapper function for `update_noise_basis`. The purpose of the wrapped
    function is that it will call every time the `initializer` function, which
    may be convenient for re-initializing variables that depend on `lhs`.
    '''

    if initializer is None:
        initializer = lambda : None

    if not callable(initializer):
        raise TypeError('Parameter `initializer` must be callable.')

    def update_noise_basis_wrapped(rhs='random_normal', normalize=True):
        update_noise_basis(lhs, rhs, normalize); initializer();

    return update_noise_basis_wrapped


def update_noise_basis(lhs, rhs='random_normal', normalize=True):
    '''Update noise basis from `dolfin.Function` or `dolfin.Constant`.'''

    ERR_LHS = 'Invalid left hand side basis.'
    ERR_RHS = 'Invalid right hand side basis.'

    if isinstance(rhs, str):
        if rhs == 'random_normal':
            generate_random = numpy.random.normal
        elif rhs == 'random_uniform':
            generate_random = numpy.random.uniform
        else:
            raise TypeError(ERR_RHS)
    else:
        generate_random = None

    if isinstance(lhs, dolfin.Function):

        if generate_random is not None:
            vec = generate_random(size=lhs.function_space().dim())

        elif isinstance(rhs, (dolfin.Function, dolfin.Constant, dolfin.Expression)):
            vec = dolfin.interpolate(rhs, lhs.function_space()).vector()

        else:
            raise TypeError(ERR_RHS)

        lhs.vector()[:] = vec

        if normalize:
            lhs.vector()[:] /= dolfin.norm(lhs)

    elif isinstance(lhs, dolfin.Constant):

        if generate_random is not None:
            vec = generate_random(size=lhs.values().shape)

        elif isinstance(rhs, dolfin.Constant):

            if len(lhs) != len(rhs):
                raise TypeError(ERR_RHS)

            vec = rhs.values()

        elif hasattr(rhs, '__iter__'):

            if len(lhs) != len(rhs):
                raise TypeError(ERR_RHS)

            if isinstance(rhs, (list,tuple)):
                vec = numpy.array(rhs)
            elif isinstance(rhs, np.ndarray):
                vec = rhs
            else:
                raise TypeError(ERR_RHS)

        else:
            raise TypeError(ERR_RHS)

        if normalize:
            vec /= numpy.linalg.norm(vec)

        lhs.assign(dolfin.Constant(vec))

    else:
        raise TypeError(ERR_LHS)


def factory_map_parameters(mapping_function):
    '''Make a function that maps reference parameters to new parameters, which
    may satisfy a certain property that is enforced by the mapping function.'''

    def map_parameters(params, keys_or_indices=None):
        '''Map reference parameters whose keys or indices are provided to new
        parameters as defined by the mapping function. Note, if `keys_or_indices`
        is `None`, `map_parameters` will maps all parameters.'''

        if not isinstance(params, (dict, list, numpy.ndarray)):
            raise TypeError('Expected parameter `params` to be `dict`, `list`, '
                f'or `numpy.ndarray` but instead received "{type(params)}".')

        params = params.copy()

        if not keys_or_indices:

            if hasattr(params, 'keys'):
                keys_or_indices = params.keys()

            else: # hasattr(params, '__getitem__'):
                keys_or_indices = range(len(params))

        for k in keys_or_indices:
            params[k] = mapping_function(params[k])

        return params

    return map_parameters
