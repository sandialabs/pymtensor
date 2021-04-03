# from numpy import array, pi, einsum
from pymtensor.symmetry import (deg2rad, rotx, rotz, roty, rotu,
   SgSymOps)
from pymtensor.sym_tensor import SymbolicTensor, SparseSymbolicTensor
from pymtensor.rot_tensor import to_voigt
# We use the NumPy testing suite to make it easier to compare arrays
from numpy.testing import (assert_equal, assert_allclose, assert_array_equal, 
    assert_raises, run_module_suite, TestCase)
from numpy import array
from sympy import pi, Rational, Symbol, sqrt as sp_sqrt
# from sympy import symbols, MatrixSymbol


class TestRotations(TestCase):
    
    def test_90deg_rotations(self):
        """
        Notes
        -----
        These rotations appear to be clockwise when viewed from above the axis
        of rotation.
        """
        HALF = Rational(1, 2)
        theta = pi * HALF
        xhat = array([1, 0, 0])
        yhat = array([0, 1, 0])
        zhat = array([0, 0, 1])
        approx1 = rotx(theta)
        approx2 = rotu(theta, xhat)
        exact = array([[ 1,  0,  0],
                       [ 0,  0, -1],
                       [ 0,  1,  0]])
        for approx in [approx1, approx2]:
            assert_equal(approx, exact)
        approx1 = roty(theta)
        approx2 = rotu(theta, yhat)
        exact = array([[ 0,  0,  1],
                       [ 0,  1,  0],
                       [-1,  0,  0]])
        for approx in [approx1, approx2]:
            assert_equal(approx, exact)
        approx1 = rotz(theta)
        approx2 = rotu(theta, zhat)
        exact = array([[ 0, -1,  0],
                       [ 1,  0,  0],
                       [ 0,  0,  1]])
        for approx in [approx1, approx2]:
            assert_equal(approx, exact)
    

class TestDegToRad(TestCase):

    def test_known(self):
        deg_arr = [30,   45,   60,   90,   180]
        rad_arr = [pi/6, pi/4, pi/3, pi/2, pi]
        for deg, rad in zip(deg_arr, rad_arr):
            newrad = deg2rad(deg)
            assert_equal(newrad, rad)


class TestSymbolicTensor(TestCase):
    
    # Initialize the symmetry operations once for use in multiple tests
    SSO = SgSymOps()

    def test__create_slices(self):
        dims = [2, 1, 1, 3]
        exact = [slice(0, 2), slice(2, 3), slice(3, 4), slice(4, 7)]
        approx = SymbolicTensor._create_slices(dims)
        assert_equal(approx, exact)
        
    def test__newindex(self):
        # Following 3 lines are just to get the mappings
        st = SymbolicTensor('ABcDe', symbol='a', create_tensor=False)
        # one-based input index, voigt notation => (6, 5, 2, 2, 1)
        index = [[1, 2], [1, 3], [2], [2, 2], [1]]
        # Shift to make the index zero based
        index = [item - 1 for sublist in index for item in sublist]
        approx = st._newindex(index, start=1, voigt=True)
        exact = [6, 5, 2, 2, 1]
        assert_equal(approx, exact)
        # sort index 0 and 3
        st = SymbolicTensor('ABcAe', symbol='a', create_tensor=False)
        approx = st._newindex(index, start=1, voigt=True)
        # Unsorted => exact = [6, 5, 2, 2, 1]
        # Sorted   => swap indices 0 and 3
        exact = [2, 5, 2, 6, 1]
        assert_equal(approx, exact)
        # Try setting `voigt=False`, (2, 6, 5) -> (2, 5, 6) -> (2, 1, 3, 1, 2)
        # Note that sorting with `voigt=False`gives the following ordering:
        # (1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3)
        index = [[2], [1, 2], [1, 3]]
        index = [item - 1 for sublist in index for item in sublist]
        st = SymbolicTensor('aBC', symbol='a', create_tensor=False)
        approx = st._newindex(index, start=1, voigt=False)
        exact = [2, 1, 2, 1, 3]
        assert_equal(approx, exact)
    
    def test__parse_name(self):
        name = 'a2,b2,c1,b2,a2,b1'
        approx_dims, approx_repeats = SymbolicTensor._parse_name(name)
        exact_dims = [2, 2, 1, 2, 2, 1]
        exact_repeats = [[0, 4], [1, 3]]
        assert_equal(approx_dims, exact_dims)
        approx_repeats.sort()
        assert_equal(approx_repeats, exact_repeats)
        name = 'ABcAe'
        exact_dims = [2, 2, 1, 2, 1]
        exact_repeats = [[0, 3]]
        approx_dims, approx_repeats = SymbolicTensor._parse_name(name)
        assert_equal(approx_dims, exact_dims)
        approx_repeats.sort()
        assert_equal(approx_repeats, exact_repeats)

    def test__repeated_indices(self):
        name = 'ABcDefgABcDeABc'
        exact = [[0, 7, 12], [1, 8, 13], [2, 9, 14], [3, 10], [4, 11]]
        # Don't try creating a tensor with more than six indices, you'll 
        # probably run out of memory.
        approx = SymbolicTensor._repeated_indices(name)
        approx.sort()
        assert_equal(approx, exact)
        name = ['i1', 'j2', 'i1', 'i1', 'j2', 'k1']
        approx = SymbolicTensor._repeated_indices(name)
        approx.sort()
        exact = [[0, 2, 3], [1, 4]]
        assert_equal(approx, exact)
        
    def test_voigt_map(self):
        exact_vm = {(1, 1): 1, (2, 2): 2, (3, 3): 3, (2, 3): 4, (3, 2): 4,
                    (1, 3): 5, (3, 1): 5, (1, 2): 6, (2, 1): 6}
        exact_ivm = {1: (1, 1), 2: (2, 2), 3: (3, 3), 4: (2, 3), 5: (1, 3), 
                     6: (1, 2)}
        approx_ivm, approx_vm = SymbolicTensor.voigt_map(dim=2, start=1)
        assert_equal(approx_vm, exact_vm)
        assert_equal(approx_ivm, exact_ivm)
    
    
#     def test_known_tensors(self):
#         exact = array([[Symbol("a11"), Symbol("a12"), Symbol("a13")],
#                        [Symbol("a21"), Symbol("a22"), Symbol("a23")],
#                        [Symbol("a31"), Symbol("a32"), Symbol("a33")]])
#         st = SymbolicTensor("ab", 'a')
#         approx = st.tensor
#         assert_array_equal(approx, exact)
#         exact = array([[Symbol("e11"), Symbol("e12"), Symbol("e13")],
#                        [Symbol("e12"), Symbol("e22"), Symbol("e23")],
#                        [Symbol("e13"), Symbol("e23"), Symbol("e33")]])
#         st = SymbolicTensor("aa", 'e')
#         approx = st.tensor
#         assert_array_equal(approx, exact)
# 
#     def test_known_dielectric(self):
#         syms = [Symbol(val) for val in ("e11", "e12", "e13", "e21", "e22", 
#                                         "e23", "e31", "e32", "e33")]
#         e11, e12, e13, e21, e22, e23, e31, e32, e33 = syms
#         st = SymbolicTensor("aa", 'e')
#         # Triclinic
#         symops = self.SSO('1')
#         approx = st.apply_symmetry(symops)
#         exact = array([[e11, e12, e13],
#                        [e12, e22, e23],
#                        [e13, e23, e33]])
#         assert_array_equal(approx, exact)
#         # Monoclinic
#         symops = self.SSO('2')
#         approx = st.apply_symmetry(symops)
#         exact = array([[e11,   0, e13],
#                        [  0, e22,   0],
#                        [e13,   0, e33]])
#         assert_array_equal(approx, exact)
#         # Hexagonal
#         symops = self.SSO('3dm')
#         approx = st.apply_symmetry(symops)
#         exact = array([[e11,   0,   0],
#                        [  0, e11,   0],
#                        [  0,   0, e33]])
#         assert_array_equal(approx, exact)
#         
#     def test_known_piezo(self):
#         syms = [Symbol(val) for val in ("e11", "e12", "e13", "e14", "e15", "e16",
#                                         "e21", "e22", "e23", "e24", "e25", "e26",
#                                         "e31", "e32", "e33", "e34", "e36", "e36")]
#         e11, e12, e13, e14, e15, e16 = syms[:6]
#         e21, e22, e23, e24, e25, e26 = syms[6:12]
#         e31, e32, e33, e34, e35, e36 = syms[12:]
#         st = SymbolicTensor("aB", 'e')
#         # Orthorhombic
#         symops = self.SSO('mm2')
#         approx = to_voigt(st.apply_symmetry(symops))
#         exact = array([[   0,   0,   0,   0, e15,   0],
#                        [   0,   0,   0, e24,   0,   0],
#                        [ e31, e32, e33,   0,   0,   0]])
#         assert_array_equal(approx, exact)
#         # Tetragonal
#         symops = self.SSO('4d2m')
#         approx = to_voigt(st.apply_symmetry(symops))
#         exact = array([[   0,   0,   0, e14,   0,   0],
#                        [   0,   0,   0,   0, e14,   0],
#                        [   0,   0,   0,   0,   0, e36]])
#         assert_array_equal(approx, exact)
#         # Hexagonal
#         symops = self.SSO('3m')
#         approx = to_voigt(st.apply_symmetry(symops))
#         exact = array([[   0,   0,   0,   0, e15, e16],
#                        [ e16,-e16,   0, e15,   0,   0],
#                        [ e31, e31, e33,   0,   0,   0]])
#         assert_array_equal(approx, exact)
#         symops = self.SSO('6dm2')
#         approx = to_voigt(st.apply_symmetry(symops))
#         exact = array([[   0,   0,   0,   0,   0, e16],
#                        [ e16,-e16,   0,   0,   0,   0],
#                        [   0,   0,   0,   0,   0,   0]])
#         assert_array_equal(approx, exact)
#         # Cubic
#         symops = self.SSO('4d3m')
#         approx = to_voigt(st.apply_symmetry(symops))
#         exact = array([[   0,   0,   0, e14,   0,   0],
#                        [   0,   0,   0,   0, e14,   0],
#                        [   0,   0,   0,   0,   0, e14]])
#         assert_array_equal(approx, exact)
#     
#     def test_known_stress(self):
#         syms = [Symbol(val) for val in ("e11", "e12", "e13", "e14", "e15", "e16",
#                                         "e21", "e22", "e23", "e24", "e25", "e26",
#                                         "e31", "e32", "e33", "e34", "e35", "e36",
#                                         "e41", "e42", "e43", "e44", "e45", "e46",
#                                         "e51", "e52", "e53", "e54", "e55", "e56",
#                                         "e61", "e62", "e63", "e64", "e65", "e66")]
#         e11, e12, e13, e14, e15, e16 = syms[:6]
#         e21, e22, e23, e24, e25, e26 = syms[6:12]
#         e31, e32, e33, e34, e35, e36 = syms[12:18]
#         e41, e42, e43, e44, e45, e46 = syms[18:24]
#         e51, e52, e53, e54, e55, e56 = syms[24:30]
#         e61, e62, e63, e64, e65, e66 = syms[30:]
#         st = SymbolicTensor("AA", 'e')
#         # Orthorhombic
#         symops = self.SSO('mmm')
#         approx = to_voigt(st.apply_symmetry(symops))
#         exact = array([[ e11, e12, e13,   0,   0,   0],
#                        [ e12, e22, e23,   0,   0,   0],
#                        [ e13, e23, e33,   0,   0,   0],
#                        [   0,   0,   0, e44,   0,   0],
#                        [   0,   0,   0,   0, e55,   0],
#                        [   0,   0,   0,   0,   0, e66]])
#         assert_array_equal(approx, exact)
#         # Tetragonal
#         symops = self.SSO('4d')
#         approx = to_voigt(st.apply_symmetry(symops))
#         exact = array([[ e11, e12, e13,   0,   0, e16],
#                        [ e12, e11, e13,   0,   0,-e16],
#                        [ e13, e13, e33,   0,   0,   0],
#                        [   0,   0,   0, e44,   0,   0],
#                        [   0,   0,   0,   0, e44,   0],
#                        [ e16,-e16,   0,   0,   0, e66]])
#         assert_array_equal(approx, exact)
#         # Hexagonal
#         symops = self.SSO('3m')
# #         var1 = (e11-e12) / 2
#         var1 = (e11-e12) * 0.5
#         approx = to_voigt(st.apply_symmetry(symops, atol=1e-13))
# #         approx[-1, -1] = var1
# #         approx = to_voigt(st.apply_symmetry(symops,numerical=False))
#         exact = array([[ e11, e12, e13, e14,   0,   0],
#                        [ e12, e11, e13,-e14,   0,   0],
#                        [ e13, e13, e33,   0,   0,   0],
#                        [ e14,-e14,   0, e44,   0,   0],
#                        [   0,   0,   0,   0, e44, e14],
#                        [   0,   0,   0,   0, e14,var1]])
#         prob = approx[-1, -1]-exact[-1,-1]
# #         print(approx[-1,-1].coeff(e11))
# #         print(prob.coeff(e11), prob.coeff(e12))
# #         print('Test stress')
# #         print('approx=')
# #         print(approx.__repr__())
# #         print('exact=')
# #         print(exact.__repr__())
#         assert_array_equal(approx, exact)
#         # Cubic

class TestSparseSymbolicTensor(TestCase):
    
    # Initialize the symmetry operations once for use in multiple tests
#     SST = SparseSymbolicTensor()

    def test_mapping(self):
        sst = SparseSymbolicTensor('a1,a1', 'c')
        sst = SparseSymbolicTensor('a2,b1,a2,a2,b1', 'c')
        sst = SparseSymbolicTensor('AbAAb', 'c')
#         exact = [slice(0, 2), slice(2, 3), slice(3, 4), slice(4, 7)]
#         approx = SymbolicTensor._create_slices(dims)
#         assert_equal(approx, exact)
        

if __name__ == "__main__":
#     run_module_suite()
    run_module_suite(argv=['', '--nocapture'])