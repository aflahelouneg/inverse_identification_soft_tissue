'''Some useful methods.'''

import ufl
import dolfin


def factory_update_variables(vars, forms):
    '''For assigning variables with values of integrals (forms).

    Parameters
    ----------
    vars : dolfin.Constant, or a list or tuple of dolfin.Constant's
        Sequence of cost functional weights for the measurements.
    forms : ufl.Form, or a list or tuple of ufl.Form's
        Sequence of cost functional weights for the measurements.

    Important
    ---------
    All previously assembled forms are memoized. This is computationally
    efficient but can problematic if any coefficient in the form is updated.
    In this case, the cached values of the assembled forms must be cleared.
    Call the returned function's attribute method `clear_cache` to do this.

    '''

    if not isinstance(vars, (list,tuple)):  vars  = (vars,)
    if not isinstance(forms, (list,tuple)): forms = (forms,)

    if not all(isinstance(v, dolfin.Constant) for v in vars):
        raise TypeError('Expected `vars` to contain `dolfin.Constant`s.')

    if not all(isinstance(f, ufl.Form) for f in forms):
        raise TypeError('Expected `forms` to contain `ufl.Form`s.')

    # Memoize assembled
    assembled_forms = {}

    def update_variables(t_ref=None):
        '''Update variables `vars` (of type `dolfin.Constant`) by evaluating
        `forms` (of type `ufl.Form`). Parameter `t_ref` is used to make a
        key identifier for the value. The value can then be reused later. Note,
        if `t_ref` is `None`, the form is reevaluated. Function's member method
        `clear_cache` can be called to clear the dict of memoized forms.
        '''

        if t_ref is not None:

            for v, f in zip(vars, forms):
                k = (t_ref, id(v))

                if k in assembled_forms:
                    value = assembled_forms[k]
                else:
                    value = dolfin.assemble(f)
                    assembled_forms[k] = value

                v.assign(value)

        else:

            for v, f in zip(vars, forms):
                v.assign(dolfin.assemble(f))

    # Call this method to clear previously assembled forms
    update_variables.clear_cache = assembled_forms.clear

    return update_variables
