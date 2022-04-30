from functools import reduce
from itertools import chain, combinations_with_replacement, permutations, product
from operator import mul

from pymtensor.indexing_helpers import expand2full, sort_lists_convert2tuples
from pymtensor.sym_tensor import SymbolicTensor

# TODO: Create an easy-to-use function to call for finding tensor symmetries
def symfinder(indices, symops, reverse=True):
    # TODO: support different sorting options.  This has to be done in the 
    # symmetry finder code because the order of the variables in the system of
    # equations determines the sorting of variables.
    pass


class SparseSymbolicTensor(SymbolicTensor):
    def __init__(self, indices, symbol='c', start=1, voigt=True, reverse=True,
                 create_tensor=True):
        """
        PyMTensor applies crystal symmetries to dielectric, elastic, 
        piezoelectric and other material tensors of interest.  It determines 
        what components are zero and the relationships between the nonzero 
        components for all crystallographic point groups.  It is capable of 
        computing components of higher-order material tensors needed when 
        treating nonlinear effects.
        
        start: integer
            Allow for either 0- or 1-based indexing (`start=0` or `start=1`).
        """
        self.start_index = start
        dims, repeats = SymbolicTensor._parse_name(indices)
        # Total uncompressed tensor dimension
        tdim = sum(dims)
        print(dims, repeats, tdim)
        # Initialize all necessary Voigt and inverse-Voigt mappings based on the
        # unique dimensions
        ivm = {}
        vm = {}
        # First we find the minor symmetries (Voigt mappings)
        for dim in set(dims):
            ivm[dim], vm[dim] = SymbolicTensor.voigt_map(dim, start=0)
        print('vm=', vm)
        print('ivm=', ivm)
        self.vm = vm
        self.ivm = ivm
        # Now we find all of the major symmetries
        red_indices = self._reduced_indices(dims, repeats)
#         for (dim, repeats) in red_indices:
#             print(self._major_syms(len(ivm[dim]), repeats))
        major_and_full_syms = [self.major_and_full_syms(dim, repeats)
                               for (dim, repeats) in red_indices]
        major_syms = [val[0] for val in major_and_full_syms]
        print('major_syms = ', major_syms)
        full_syms =  [val[1] for val in major_and_full_syms]
        print('full_syms = ', full_syms)
        expanded_full_indices = tuple(product(*full_syms))
        print('expanded_full_indices=', expanded_full_indices)
        # self.assemble_matrix(expanded_full_indices, symops)
#         major_indices_voigt = [vals[0] for vals in major_syms]
#         degeneracies = [vals[1] for vals in major_syms]
#         print('degeneracies = ', degeneracies)
#         print('major_indices_voigt = ', major_indices_voigt)
        # unique_full_indices = self._unique_full_indices(ivm, major_syms)
        # print('unique_full_indices=', tuple(tuple(val) for val in unique_full_indices))

    @staticmethod
    def assemble_matrix(indices, symops):
        """
        Create the matrix for a given set of indices and symmetry operations.
        
        
        """
        pass
    
    @staticmethod
    def prod(iterable):
        return reduce(mul, iterable)
    
    @staticmethod
    def form_matrix_entry(i, j, full_indices, symop):
        """Form an element of the reduced linear system.
        """
        prod = SparseSymbolicTensor.prod
        # val = 1
        # print('full_indices[{}]={}'.format(i, full_indices[i]))
        # print('full_indices[{}]={}'.format(j, full_indices[j]))
        # for irow, icol in zip(full_indices[i], full_indices[j]):
        #     iirow = irow[0]
        #     print('iirow={}'.format(iirow))
            # sum1 = 0
            # for iicol in icol:
            #     sum1 += reduce(mul, (symop[iiirow][iiicol] for iiirow, iiicol in zip(iirow, iicol)))
            #     print('iicol={}'.format(iicol))
            #     print('sum1={}'.format(sum1))
            # val *= sum1
                # print('iirow[{}]={}'.format(i, iirow[i]))
                # print('iicol[{}]={}', icol[0])
        # The following three lines are equivalent to the nested for loops above.
        val = prod(sum(prod(symop[iiirow][iiicol] for iiirow, iiicol 
                            in zip(irow[0], iicol)) for iicol in icol)
                   for irow, icol in zip(full_indices[i], full_indices[j]))
        # for col in cols:
        #     val += reduce(mul, (symop[irow][icol] for irow, icol in zip(row, col)))
        if i == j:
            # TODO: we should figure out what unity is in the algebra in `symop`
            val -= 1
        return val

    @staticmethod
    def _unique_full_indices(major_indices):
        """
        Convert a tuple of Voigt indices to an expanded tuple of full indices.
        """
#         return [ivm[voigt_index] for voigt_index in voigt_indices]
        # full = tuple(chain(*foo) for foo in product(voigt_indices))
        full = tuple(product(major_indices))
        return full

    @staticmethod
    def _flatten_indices(indices):
        return list(tuple(chain.from_iterable(val)) 
                    for product_indices in indices
                    for val in product(*product_indices))

    @staticmethod
    def _reduced_indices(dims, repeats):
        """
        Find all unique index dimensions and their degeneracies.
        
        Parameters
        ==========
        dims : iterable of integers
            An iterable containing the tensor's dimensions
        repeats : iterable of iterable of integers
            An iterable containing the repeated indices for each unique dimension
        
        Returns
        =======
        major_syms : list of lists
            Each nested list contains the representative indices, degeneracy,
            and remaining indices for each unique dimension.
        """
#         unique_dims = set(dims)
#         print(dims, unique_dims, repeats)
        # (dim, number of repeats, indices to remove)
        red_indices = [(dim, 1) for dim in dims]
        rm_indices = [index for repeat in repeats for index in repeat[1:]]
        rm_indices.sort()
        for repeat in repeats:
            num_repeats = len(repeat)
            first_repeat = repeat[0]
            dim = dims[first_repeat]
            red_indices[first_repeat] = (dim, num_repeats)
        # Go through the list backwards to not mess up the list ordering
        for rm_ind in rm_indices[::-1]:
            red_indices.pop(rm_ind)
        return red_indices
    
#     @staticmethod
    def major_and_full_syms(self, dim_voigt, num_repeats):
        """
        Find all equivalent indices, their degeneracies, and store a 
        representative fully expanded (non-Voigt notation) set of indices.
        Currently, we store the canonically largest representative index but
        this may change in the future to allow optional ordering.
        
        Parameters
        ==========
        dim_voigt : int
            Dimension of minor (Voigt) symmetry
        num_repeats : int
            Number of repetitions of the current dimension
        
        Returns
        =======
        major_syms : tuple of int tuples
        """
        # Representative and remainder indices lists
        ivm = self.ivm
        num_voigt = len(ivm[dim_voigt])
        # We create sets of the permutations in order to pick out the unique
        # values.  We then convert the sets to lists so that they can be sorted.
        voigtmap = {key: sort_lists_convert2tuples(set(permutations(val)))
                    for key, val in ivm[dim_voigt].items()}
        print('voigtmap=', voigtmap)
        voigt_indices = range(num_voigt)
        print('voigt_indices=', voigt_indices)
        unique_indices = tuple(combinations_with_replacement(voigt_indices, num_repeats))
        print('unique_indices=', unique_indices)
        major_indices = tuple(list(set(tuple(permutations(indices)))) for indices in unique_indices)
        print('major_indices=', major_indices)
        # print(len(unique_indices), len(major_indices))
        full_indices = tuple(expand2full(voigtmap, indices)
                             for indices in major_indices)
        print('full_indices=', full_indices)
        return major_indices, full_indices
        # print('len(unique_indices)=', len(unique_indices))
        # print('len(major_indices)=', len(unique_indices))
        # # for major_index in major_indices:
        # #     print(tuple(tuple(chain.from_iterable(map(voigtmap.get, tuple(val)))) for val in major_index))
        #
        # # full_indices = [tuple(chain.from_iterable(val)) for val in product(*major_indices)]
        # print('full_indices=', full_indices[0])
        # indices_map = []
        # full_indices_map = []
        # for indices in combinations_with_replacement(voigt_indices, num_repeats):
        #     equiv_indices = set(permutations(indices))
        #     indices_map.append((indices, equiv_indices))
        #     print([val for equiv_index in equiv_indices
        #            for val in equiv_index])
        # print('indices_map=', indices_map)
        # return indices_map
