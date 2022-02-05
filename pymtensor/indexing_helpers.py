from functools import reduce
from itertools import chain, product
from operator import mul


def apply_mapping(indices, mapping):
    """Apply a set of indices to a mapping.
    
    Parameters
    ----------
    indices : iterable
        Iterable containing the keys used to index into `mapping`
    mapping : dictionary
        Dictionary with objects to index.
    
    Returns
    -------
    objects_tuple: tuple of objects
        A tuple containing the objects corresponding to each index in `indices`.
    """
    return tuple(mapping[index] for index in indices)


def expand_all(indices):
    full = tuple(tuple(chain.from_iterable(val)) for val in  product(*indices))
    return full


def expand2full(mapping, iter_indices):
    full = tuple(tuple(chain.from_iterable(val))
                 for indices in iter_indices
                 for val in product(*apply_mapping(indices, mapping)))
    # def generate_full():
    #     for indices in iter_indices:
    #         # print('indices={}'.format(indices))
    #         expanded = tuple(mapping[index] for index in indices)
    #         # print(tuple(product(*expanded)))
    #         # print('expanded={}'.format(expanded))
    #         for val in product(*expanded):
    #             yield tuple(chain.from_iterable(val))
    # full = tuple(val for val in generate_full())
    return full


def sort_lists_convert2tuples(iter_of_lists):
    newiter = list(iter_of_lists)
    newiter.sort()
    return tuple(tuple(val) for val in newiter)


def form_matrix_entry(i, j, full_indices, symop):
    """Form an element of the 
    """
    row = full_indices[i][0]
    cols = full_indices[j]
    val = 0
    for col in cols:
        val += reduce(mul, (symop[irow][icol] for irow, icol in zip(row, col)))
    if i == j:
        val -= 1
    return val
