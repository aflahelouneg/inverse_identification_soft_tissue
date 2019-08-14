'''
Some utility methods.

'''

def list_values_from_iterable(iterable, value_types, value_list=None):
    '''Extract values of specified types from a nested iterable to a list.

    The traversal of `iterable` is left-to-right and depth-first.

    Parameters
    ----------
    iterable (iterable): Nested iterables with attributes "index" or "keys".
    value_types (tuple of type's): Extract values of types in `value_types`.
    value_list (list or None): Extract values to `value_list` (optional).

    Returns
    -------
    list: List of extracted values of types specified in `value_types`.

    '''

    if value_list is None:
        value_list = []

    if isinstance(iterable, value_types):
        value_list.append(iterable)

    elif hasattr(iterable, 'keys'):
        for k in iterable.keys():
            list_values_from_iterable(iterable[k], value_types, value_list)

    elif hasattr(iterable, 'index'):
        for iterable_i in iterable:
            list_values_from_iterable(iterable_i, value_types, value_list)

    return value_list


def replicate_tree_structure(iterable, value_types):
    '''Replicate tree-like structure of `iterable` keeping the original leaf
    values provided the types are contained by `value_types`. If a leaf value
    is not of any of the specified types, assume the value to be `None`.'''

    def copy(iterable):

        if isinstance(iterable, value_types):
            return iterable

        elif hasattr(iterable, 'keys'):
            return {k : copy(iterable[k]) for k in iterable.keys()}

        elif hasattr(iterable, 'index'):
            return [copy(iterable_i) for iterable_i in iterable]

        else:
            return None

    return copy(iterable)


def update_existing_keyvalues(lhs, rhs):
    '''Recursively update values of dict-like `lhs` with those in `rhs`.'''

    for k in rhs.keys():

        if k not in lhs.keys():
            raise KeyError(k)

        if hasattr(lhs[k], 'keys') and hasattr(rhs[k], 'keys'):
            update_existing_keyvalues(lhs[k], rhs[k])
        else:
            lhs[k] = rhs[k]
