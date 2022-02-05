from collections import defaultdict
from itertools import chain, combinations_with_replacement, permutations, product
import logging
import time
import psutil

from numpy import (array, empty, empty_like,  nditer, set_printoptions)
# from scipy.linalg import lu, svd
from sympy import Symbol
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys

from pymtensor.rot_tensor import rot_tensor#, to_voigt
from pymtensor.indexing_helpers import expand2full, sort_lists_convert2tuples
from abc import abstractstaticmethod


class SymbolicTensor(object):
    
    def __init__(self, indices, symbol, start=1, voigt=True, reverse=True,
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
        dims, repeats = SymbolicTensor._parse_name(indices)
        # Total uncompressed tensor dimension
        tdim = sum(dims)
        # Initialize all necessary Voigt and inverse-Voigt mappings based on the
        # unique dimensions
        ivm = {}
        vm = {}
        for dim in set(dims):
            ivm[dim], vm[dim] = SymbolicTensor.voigt_map(dim, start)
#         print('vm=', vm)
#         print('ivm=', ivm)
        self.dims = dims
        self.vm = vm
        self.ivm = ivm
        self.slices = SymbolicTensor._create_slices(dims)
        self.repeats = repeats
        # Don't create the tensor if just testing initialization methods
        if create_tensor:
            tensor = empty((3,)*tdim, dtype=object)
            # Create slices for applying the Voigt mappings
            # Find all unique tensor component names
            symbols_dict = {}
            print('start newindex')
            for index in product(range(3), repeat=tdim):
                newindex = self._newindex(index, start, voigt)
                name = ''.join([symbol] + [str(val) for val in newindex])
                print(index, newindex, name)
                symbols_dict[name] = Symbol(name)
            names = list(symbols_dict.keys())
            print('names = {}'.format(names))
            print('Number of unknowns = {}'.format(len(names)))
            sortkey = lambda i: int(i[1:])
            names.sort(reverse=reverse, key=sortkey)
            print('names=', names)
            poly_vals = ring(names, 'QQ<sqrt(3)>')
            poly_dict = dict((name, i) for i, name in enumerate(names))
            R = poly_vals[0]
            ring_vals = poly_vals[1:]
            # Place the ring values in the tensor
            for index in product(range(3), repeat=tdim):
                newindex = self._newindex(index, start, voigt)
                name = ''.join([symbol] + [str(val) for val in newindex])
                tensor[index] = ring_vals[poly_dict[name]]
            self.names = names
            self.R = R
            self.ring_vals = ring_vals
            self.tensor = tensor
        self.tdim = tdim
    
    @staticmethod
    def _create_slices(dims):
        start_slice = 0
        slices = []
        for dim in dims:
            end_slice = start_slice + dim
            slices.append(slice(start_slice, end_slice))
            start_slice = end_slice
        return slices
    
    @staticmethod
    def _parse_name(name):
        if ',' in name:
            indices = name.split(',')
            # Check if the last token is empty
            if indices[-1] == '':
                indices.pop()
        else:
            indices = []
            for val in name:
                if val.isupper():
                    indices.append(val + '2')
                else:
                    indices.append(val + '1')
        # TODO: generalize this to allow repeat characters before the 
        # integer and do some error checking
        dims = [int(val[1:]) for val in indices]
        repeats = SymbolicTensor._repeated_indices(indices)
        return dims, repeats
    
    @staticmethod
    def voigt_map(dim, start=0):
        v = {}
        v[1] = [(1,), (2,), (3,)]
        v[2] = [(1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2)]
        v[3] = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 2, 3), (1, 1, 3),
                (1, 1, 2), (2, 3, 3), (1, 3, 3), (1, 2, 2), (1, 2, 3)]
        v[4] = [(1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3), (2, 2, 2, 3), 
                (1, 1, 1, 3), (1, 1, 1, 2), (2, 2, 3, 3), (1, 1, 3, 3), 
                (1, 1, 2, 2), (2, 3, 3, 3), (1, 3, 3, 3), (1, 2, 2, 2),
                (1, 1, 2, 3), (2, 2, 1, 3), (3, 3, 1, 2)]
        vm = {}
        ivm = {}
        ordered_indices = None
        try:
            ordered_indices = v[dim]
        except NotImplementedError:
            print("Voigt map not implemented for dimension {}".format(dim))
        for i, unique_index in enumerate(ordered_indices):
            ishifted = i + start
            # Zero-based indexing: subtract 1
            shifted_index = tuple(val - 1 + start for val in unique_index)
            ivm[ishifted] = shifted_index
            # Every permutation of the index should map to the same Voigt
            # value
            for index in permutations(shifted_index):
                vm[index] = ishifted
        return ivm, vm
    
    def voigt_map_ascending(self, dim, start=0):
        v = {}
        
    
    def _newindex(self, index, start=1, voigt=True):
        """
        Create a new index possibly in Voigt notation with all symmetries and
        sortings possible.
        
        Parameters
        ----------
        index: tuple of integers
            The full tensor index to be processed
        repeats: list of lists of integers
            The indices that need to be sorted assuming all symmetric indices 
            are already reduced to Voigt notation
        start: int
            The starting integer used to choose 0- or 1- based indexing.
        
        Notes
        -----
        This function is separated from the tensor creation to simplify testing.
        """
        ivm = self.ivm
        vm = self.vm
        repeats = self.repeats
        slices = self.slices
        dims = self.dims
        # We compress to Voigt notation and then expand after sorting if the
        # user doesn't want Voigt notation
        # Example:
        # index = [0, 1, 0, 2, 1, 1, 1, 0]
        # repeats = [[0, 3]]
        # symmetric=[0, 1, 3]
        index = [val + start for val in index]
        newindex = []
        for dim, slice_ in zip(dims, slices):
            newindex.append(vm[dim][tuple(index[slice_])])
        sortkey = None
        # We can use a custom sort key if using full-tensor notation 
        if not voigt:
            sortdict = {1:1, 2:4, 3:6, 4:5, 5:3, 6:2}
            sortkey = lambda i: sortdict[i]
        # Repeated indices need to be sorted
        for irepeat in repeats:
            vals = [newindex[i] for i in irepeat]
            vals.sort(key=sortkey)
            for i, var in zip(irepeat, vals):
                newindex[i] = var
        if not voigt:
            index = newindex
            newindex = []
            for dim, val in zip(dims, index):
                newindex.extend(ivm[dim][val])
        return newindex
    
    @staticmethod
    def _repeated_indices(name):
        # Look for all repeated uppercase letters
        repeats = defaultdict(list)
        for i, val in enumerate(name):
            repeats[val].append(i)
        repeats = [val for val in repeats.values() if len(val) > 1]
        return repeats
    
    def apply_symmetry(self, symops, timings=True):
        tensor = self.tensor; R = self.R; ring_vals = self.ring_vals
        eqs = set()
        if timings: tic = time.perf_counter()
        for symop in symops:
            if timings: symtic = time.perf_counter()
            # Convert the components of the symmetry operation arrays into
            # members of the polynomial ring.
            poly_symop = array([[R(val) for val in row] for row in symop])
            row = (rot_tensor(poly_symop, tensor) - tensor).flatten()
            for eq in row:
                if eq != 0:
                    eqs.add(eq)
            if timings: symtoc = time.perf_counter()
            if timings: print(f"Apply symmetry = {symtoc - symtic:0.4f} seconds")
        if len(eqs) == 0:
            return {}
        print('len(eqs) = {}'.format(len(eqs)))
        print('eqs = ')
        for eq in eqs:
            print(eq.as_expr())
        if timings: toc = time.perf_counter()
        if timings: print(f"Rotate tensor = {toc - tic:0.4f} seconds")
        
        if timings: tic = time.perf_counter()
        process = psutil.Process()
        print('bytes used = {}'.format(process.memory_info().rss))  # in bytes 
        sol = solve_lin_sys(eqs, R, _raw=False)
        process = psutil.Process()
        print('bytes used = {}'.format(process.memory_info().rss))  # in bytes 
        if timings: toc = time.perf_counter()
        if timings: print(f"Solve linear system = {toc - tic:0.4f} seconds")
        numfreevars = len(ring_vals) - len(sol)
        print(f"Number of free variables = {numfreevars}")
        logging.info('Number of free variables = {}'.format(numfreevars))
        return sol
    
    def sol_details(self, symops):
        R = self.R; names = self.names
        sol = self.apply_symmetry(symops)
        tic = time.perf_counter()
        fullsol = {}
        for name in names:
            symbol = Symbol(name)
            fullsol[name] = sol.get(symbol, symbol)
        toc = time.perf_counter()
        print(f"To full solution = {toc - tic:0.4f} seconds")
        return fullsol, R
    
    def unique_tensor(self, symops):
        tensor = self.tensor
        sol = self.apply_symmetry(symops)
        tic = time.perf_counter()
        newtensor = empty_like(tensor)
        it = nditer(tensor, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            oldval = it[0].item()
            # Try to retrieve the new value from the solution dictionary.  If
            # the variable doesn't exist then it is unique and we just use the
            # original symbol via the dictionary `get` method
            newval = sol.get(oldval, oldval)
            newtensor[it.multi_index] = newval
            it.iternext()
        toc = time.perf_counter()
        print(f"Format solution = {toc - tic:0.4f} seconds")
        return newtensor
                


def symbolic_piezo(dim, sym, canonical=True, sort=True):
    voigt_map = {(0, 0): 0, (1, 1): 1, (2, 2): 2,
                 (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, 
                 (0, 1): 5, (1, 0): 5}
    tensor = empty((3,)*dim, dtype=object)
    indices = product(range(3), repeat=dim)
    if canonical:
        ls, nc, rs = [''] * 3
        shift = 1
    else:
        ls, nc, rs = ['[', ',', ']']
        shift = 0
    if dim == 2:
        for index in indices:
            i, j = index
            voigt_order = (i, j)
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 3:
        for index in indices:
            # The following statement handles the third-order dielectric constant
            if sort:
                newindex = tuple(sorted(index))
                i = newindex[0]
                J = voigt_map[newindex[1:3]]
            else:
                i = index[0]
                J = voigt_map[index[1:3]]
            voigt_order = (i, J)
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 4:
        for index in indices:
            I = voigt_map[index[0:2]]
            J = voigt_map[index[2:4]]
            if sort:
                voigt_order = sorted([I, J])
            else:
                voigt_order = [I, J]
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 5:
        for index in indices:
            i = index[0]
            J, K = [voigt_map[val] for val in [index[1:3], index[3:5]]]
            voigt_order = (i, J, K)
            if sort:
                voigt_order = [i] + sorted([J, K])
            else:
                voigt_order = [i, J, K]
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 6:
        for index in indices:
            I, J, K = [voigt_map[val] for val
                       in [index[2*i:2*i+2] for i in range(3)]]
            if sort:
                voigt_order = sorted([I, J, K])
            else:
                voigt_order = [I, J, K]
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    raise NotImplementedError("Dimension {} hasn't been programmed yet.".format(dim))
    return tensor


if __name__ == '__main__':
    set_printoptions(linewidth=200)
    sg = SgSymOps()
#     sg.groups['hexagonal']['3']
#     sg.groups['hexagonal']['3d']
    symops = sg('3m')
#     symops = sg('1')
#     symops = sg('32')
#     symops = sg('6')
#     symops = sg('m3d')
#     st = SymbolicTensor("aa", 'd')
    st = SymbolicTensor("abcdef", 'c')
#     st = SymbolicTensor('aB', 'e')
#     st = SymbolicTensor('AAA', 'c')
#     st = SymbolicTensor('ab', 'e')
#     C3p = sg.hexops["SIGd1"]
#     print('st=', to_voigt(st.tensor))
#     ans = to_voigt(rot_tensor(C3p, st.tensor))
#     for i in range(3):
#         for j in range(3):
#             print(i+1, j+1, ans[i, j].expand())
#     print(ans[0, 0].expand())
#     reduced_tensor = st.unique_tensor(symops)
    st.sol_details(symops)
