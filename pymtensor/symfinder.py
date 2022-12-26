from functools import reduce
from itertools import chain, combinations_with_replacement, permutations, product
from operator import mul

from pymtensor.indexing_helpers import expand2full, sort_lists_convert2tuples
from pymtensor.sym_tensor import SymbolicTensor
from sympy.polys.matrices.sdm import SDM
from sympy.polys.solvers import DomainMatrix

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
        self.tdim = tdim
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
        # print('major_syms = ', major_syms)
        full_syms =  [val[1] for val in major_and_full_syms]
        # print('full_syms = ', full_syms)
        expanded_full_indices = tuple(product(*full_syms))
        # print('expanded_full_indices=', expanded_full_indices)
        self.expanded_full_indices = expanded_full_indices
        # self.assemble_matrix(expanded_full_indices, symops)
#         major_indices_voigt = [vals[0] for vals in major_syms]
#         degeneracies = [vals[1] for vals in major_syms]
#         print('degeneracies = ', degeneracies)
#         print('major_indices_voigt = ', major_indices_voigt)
        # unique_full_indices = self._unique_full_indices(ivm, major_syms)
        # print('unique_full_indices=', tuple(tuple(val) for val in unique_full_indices))

    def apply_symmetry(self, symops, domain, timings=True):
        # if domain is None:
        #     symop = symops[0]
        # Convert all of the symmetry operators to the desired domain
        newsymops = [self.convert_symop(symop, domain) for symop in symops]
        print("Start assemble matrix")
        system = self.assemble_matrix(self.expanded_full_indices, newsymops, 
                                      self.form_matrix_entry, domain)
        print("Finish assemble matrix")
        # print('system=', system)
        print("Start RREF")
        sol, pivots = system.rref()
        print("Finish RREF")
        return sol, pivots
    
    def interpret_solution(self, sol, pivots):
        if not(SDM == type(sol.rep)):
            sol = sol.to_sparse()
        expanded_full_indices = self.expanded_full_indices
        total_vars = 3**self.tdim
        unique_vars = len(expanded_full_indices)
        nonzero_vars = 0
        solrep = sol.rep
        prod = self.prod
        zero_vars = 0
        print('len(pivots)=', len(pivots))
        for i, pivot in enumerate(pivots):
            num_equiv_vars = prod(len(val) for val in expanded_full_indices[pivot])
            unique_vars -= 1
            if len(solrep[i]) > 1:
                # The variables corresponding to this pivot are nonzero
                nonzero_vars += 1
            else:
                zero_vars += 1
                print("solrep[i]={}, expanded_full_indices[pivot][0]={}".format(solrep[i], expanded_full_indices[pivot][0]))
                # The variables corresponding to this pivot must be zero
                # zero_vars += 1#num_equiv_vars
        # print('total_vars={}, zero_vars={}'.format(total_vars, zero_vars))
        nonzero_vars += unique_vars
        print('zero_vars=', zero_vars)
        return nonzero_vars, unique_vars

    @staticmethod
    def assemble_matrix(indices, symops, func, domain=None):
        """
        Create the matrix for a given set of indices and symmetry operations.
        
        
        """
        # Figure out the size of the matrix
        ncols = len(indices)
        nrows = len(symops) * ncols
        # Generator for rows needs to return the absolute and local positions
        # zero = domain.zero
        # print('zero=', zero, '=', zero.__repr__())
        one = domain.one
        def rows_gen(symops):
            iglobal = 0
            for symop in symops:
                for ilocal in range(ncols):
                    yield (iglobal, ilocal, symop)
                    iglobal += 1
        entries = {}
        for iglobal, ilocal, symop in rows_gen(symops):
            row = {}
            for j in range(ncols):
                val = func(ilocal, j, indices, symop, one)
                # print('val=', val, ', bool(val)', bool(val))
                if val:# != zero:
                    row[j] = val
            if len(row) > 0:
                entries[iglobal] = row
        print("Create drep")
        drep = SDM(entries, (nrows, ncols), domain=domain)
        dmatrix = DomainMatrix.from_rep(drep)
        return dmatrix
        # if domain is not None:
    
    @staticmethod
    def convert_symop(symop, domain):
        return tuple(tuple(domain.convert(val) for val in row) 
                     for row in symop)
    
    @staticmethod
    def prod(iterable, zero=0):
        # res = zero + 1
        # for val in iterable:
        #     # print('type(val)={}'.format(type(val)))
        #     if val:
        #         res *= val
        #     else:
        #         return zero
        # return res
        itercopy = list(iterable)
        if not all(itercopy):
            return zero
        return reduce(mul, itercopy)
    
    @staticmethod
    def form_matrix_entry(i, j, full_indices, symop, one=1):
        """Form an element of the reduced linear system.
        """
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
        prod = SparseSymbolicTensor.prod
        zero = one * 0
        val = prod((sum(prod((symop[iiirow][iiicol] for iiirow, iiicol 
                            in zip(irow[0], iicol)), zero) for iicol in icol)
                   for irow, icol in zip(full_indices[i], full_indices[j])), zero)
        # for col in cols:
        #     val += reduce(mul, (symop[irow][icol] for irow, icol in zip(row, col)))
        if i == j:
            val -= one
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
        # print('voigtmap=', voigtmap)
        voigt_indices = range(num_voigt)
        # print('voigt_indices=', voigt_indices)
        unique_indices = tuple(combinations_with_replacement(voigt_indices, num_repeats))
        # print('unique_indices=', unique_indices)
        major_indices = tuple(list(set(tuple(permutations(indices)))) for indices in unique_indices)
        # print('major_indices=', major_indices)
        # print(len(unique_indices), len(major_indices))
        full_indices = tuple(expand2full(voigtmap, indices)
                             for indices in major_indices)
        # print('full_indices=', full_indices)
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
