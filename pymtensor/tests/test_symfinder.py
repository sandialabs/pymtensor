import unittest

from sympy import Matrix, symbols
from sympy.polys.domains import QQ, ZZ
from sympy.polys.matrices.sdm import SDM
from sympy.polys.solvers import DomainMatrix

from pymtensor.symfinder import SparseSymbolicTensor


class TestSparseSymbolicTensor(unittest.TestCase):
    
    # Initialize the symmetry operations once for use in multiple tests
#     SST = SparseSymbolicTensor()

    def test__reduced_indices(self):
        # indices = 'a1,a1' = 'aa'
        dims = [1, 1]
        repeats = [[0, 1]]
        actual = SparseSymbolicTensor._reduced_indices(dims, repeats)
        expected = [(1, 2)]
        self.assertEqual(actual, expected)
        # indices = 'a2,b1,a2,a2,b1,c3' = 'AbAAb'
        dims = [2, 1, 2, 2, 1, 3]
        repeats = [[0, 2, 3], [1, 4]]
        actual = SparseSymbolicTensor._reduced_indices(dims, repeats)
        expected = [(2, 3), (1, 2), (3, 1)]
        self.assertEqual(actual, expected)
        # indices = 'a2,b1,c2,a2,c2,c2' = 'AbBABB'
        dims = [2, 1, 2, 2, 2, 2]
        repeats = [[0, 3], [1], [2, 4, 5]]
        actual = SparseSymbolicTensor._reduced_indices(dims, repeats)
        expected = [(2, 2), (1, 1), (2, 3)]
        self.assertEqual(actual, expected)

    def test_major_and_full_syms(self):
        print("inside test__major_syms")
        # indices = 'a1,a1' = 'aa'
        # num_voigt = 3
        # num_repeats = 2
        sst = SparseSymbolicTensor('aabb', 'c')
        sst = SparseSymbolicTensor('AAA', 'c')
        # sst = SparseSymbolicTensor('AAa', 'c')
#         actual = SparseSymbolicTensor._major_syms(num_voigt, num_repeats)
        # (0, 0), -> (0, 0, 0, 0)
        # (0, 2), (2, 0) -> (0, 0, 3, 3), (3, 3, 0, 0)
        # (0, 3), (3, 0) -> (0, 0, 2, 3), (0, 0, 3, 2), (2, 3, 0, 0), (3, 2, 0, 0)
        expected = [((0, 0), 1, set()), ((0, 1), 2, {(1, 0)}), ((0, 2), 2, {(2, 0)}), ((1, 1), 1, set()), ((1, 2), 2, {(2, 1)}), ((2, 2), 1, set())]
        # self.assertEqual(actual, expected)
#         print(actual)


    def test__flatten_indices(self):
        print('inside test__flatten_indices')
        indices = [([(1, 2), (2, 1)], [(2, 2)]), ([(2, 2)], [(1, 2), (2, 1)])]
        expected = [(1, 2, 2, 2), (2, 1, 2, 2), (2, 2, 1, 2), (2, 2, 2, 1)]
        actual = SparseSymbolicTensor._flatten_indices(indices)
        self.assertEqual(actual, expected)
        indices = [([(1, 1)], [(1, 1)])]
        expected = [(1, 1)]
    
    def test__full_indices(self):
        # indices = 'a1,a1' = 'aa'
        indices = 'a2,a2'
        symbol = 'c'
        print("Inside `test__full_indices`")
        sst = SparseSymbolicTensor(indices, symbol)
        
    def test_form_matrix_entry(self):
        (a, b, c, d, e, f, g, h, i) = symbols('a b c d e f g h i')
        R = [[a, b, c], 
             [d, e, f], 
             [g, h, i]]
        full_indices = ((((2, 0), (0, 2)), ((2, 2),)),
                        (((0, 1), (1, 0)), ((2, 0), (0, 2))),)
        # irow = (((2, 0), (0, 2)), ((2, 2),))
        # icol = (((0, 1), (1, 0)), ((2, 0), (0, 2)))
        # (R[2][0] * R[0][1] + R[2][1] * R[0][0]) * (R[2][2] * R[2][0] + R[2][0] * R[2][2]) - krondel(0, 1)
        expected = (g * b + h * a) * (i * g + g * i) - 0
        actual = SparseSymbolicTensor.form_matrix_entry(0, 1, full_indices, R)
        self.assertEqual(actual, expected)
        # irow = (((0, 1), (1, 0)), ((2, 0), (0, 2)))
        # icol = (((0, 1), (1, 0)), ((2, 0), (0, 2)))
        # (R[0][0] * R[1][1] + R[0][1] * R[1][0]) * (R[2][2] * R[0][0] + R[2][0] * R[0][2]) - krondel(1, 1)
        expected = (a * e + b * d) * (i * a + g * c) - 1
        actual = SparseSymbolicTensor.form_matrix_entry(1, 1, full_indices, R)
        self.assertEqual(actual, expected)
        # TODO: test for valid `full_indices`
        full_indices = ((((0, 0, 0, 0),), ((0,),)), (((0, 0, 0, 0),), ((1,),)), (((0, 0, 0, 0),), ((2,),)), (((0, 0, 1, 1), (1, 1, 0, 0)), ((0,),)), (((0, 0, 1, 1), (1, 1, 0, 0)), ((1,),)), (((0, 0, 1, 1), (1, 1, 0, 0)), ((2,),)), (((2, 2, 0, 0), (0, 0, 2, 2)), ((0,),)), (((2, 2, 0, 0), (0, 0, 2, 2)), ((1,),)), (((2, 2, 0, 0), (0, 0, 2, 2)), ((2,),)), (((1, 2, 0, 0), (2, 1, 0, 0), (0, 0, 1, 2), (0, 0, 2, 1)), ((0,),)), (((1, 2, 0, 0), (2, 1, 0, 0), (0, 0, 1, 2), (0, 0, 2, 1)), ((1,),)), (((1, 2, 0, 0), (2, 1, 0, 0), (0, 0, 1, 2), (0, 0, 2, 1)), ((2,),)), (((0, 2, 0, 0), (2, 0, 0, 0), (0, 0, 0, 2), (0, 0, 2, 0)), ((0,),)), (((0, 2, 0, 0), (2, 0, 0, 0), (0, 0, 0, 2), (0, 0, 2, 0)), ((1,),)), (((0, 2, 0, 0), (2, 0, 0, 0), (0, 0, 0, 2), (0, 0, 2, 0)), ((2,),)), (((0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)), ((0,),)), (((0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)), ((1,),)), (((0, 0, 0, 1), (0, 0, 1, 0), (0, 1, 0, 0), (1, 0, 0, 0)), ((2,),)), (((1, 1, 1, 1),), ((0,),)), (((1, 1, 1, 1),), ((1,),)), (((1, 1, 1, 1),), ((2,),)), (((1, 1, 2, 2), (2, 2, 1, 1)), ((0,),)), (((1, 1, 2, 2), (2, 2, 1, 1)), ((1,),)), (((1, 1, 2, 2), (2, 2, 1, 1)), ((2,),)), (((1, 1, 1, 2), (1, 1, 2, 1), (1, 2, 1, 1), (2, 1, 1, 1)), ((0,),)), (((1, 1, 1, 2), (1, 1, 2, 1), (1, 2, 1, 1), (2, 1, 1, 1)), ((1,),)), (((1, 1, 1, 2), (1, 1, 2, 1), (1, 2, 1, 1), (2, 1, 1, 1)), ((2,),)), (((0, 2, 1, 1), (2, 0, 1, 1), (1, 1, 0, 2), (1, 1, 2, 0)), ((0,),)), (((0, 2, 1, 1), (2, 0, 1, 1), (1, 1, 0, 2), (1, 1, 2, 0)), ((1,),)), (((0, 2, 1, 1), (2, 0, 1, 1), (1, 1, 0, 2), (1, 1, 2, 0)), ((2,),)), (((1, 1, 0, 1), (1, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 1)), ((0,),)), (((1, 1, 0, 1), (1, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 1)), ((1,),)), (((1, 1, 0, 1), (1, 1, 1, 0), (0, 1, 1, 1), (1, 0, 1, 1)), ((2,),)), (((2, 2, 2, 2),), ((0,),)), (((2, 2, 2, 2),), ((1,),)), (((2, 2, 2, 2),), ((2,),)), (((1, 2, 2, 2), (2, 1, 2, 2), (2, 2, 1, 2), (2, 2, 2, 1)), ((0,),)), (((1, 2, 2, 2), (2, 1, 2, 2), (2, 2, 1, 2), (2, 2, 2, 1)), ((1,),)), (((1, 2, 2, 2), (2, 1, 2, 2), (2, 2, 1, 2), (2, 2, 2, 1)), ((2,),)), (((0, 2, 2, 2), (2, 0, 2, 2), (2, 2, 0, 2), (2, 2, 2, 0)), ((0,),)), (((0, 2, 2, 2), (2, 0, 2, 2), (2, 2, 0, 2), (2, 2, 2, 0)), ((1,),)), (((0, 2, 2, 2), (2, 0, 2, 2), (2, 2, 0, 2), (2, 2, 2, 0)), ((2,),)), (((2, 2, 0, 1), (2, 2, 1, 0), (0, 1, 2, 2), (1, 0, 2, 2)), ((0,),)), (((2, 2, 0, 1), (2, 2, 1, 0), (0, 1, 2, 2), (1, 0, 2, 2)), ((1,),)), (((2, 2, 0, 1), (2, 2, 1, 0), (0, 1, 2, 2), (1, 0, 2, 2)), ((2,),)), (((1, 2, 1, 2), (1, 2, 2, 1), (2, 1, 1, 2), (2, 1, 2, 1)), ((0,),)), (((1, 2, 1, 2), (1, 2, 2, 1), (2, 1, 1, 2), (2, 1, 2, 1)), ((1,),)), (((1, 2, 1, 2), (1, 2, 2, 1), (2, 1, 1, 2), (2, 1, 2, 1)), ((2,),)), (((1, 2, 0, 2), (1, 2, 2, 0), (2, 1, 0, 2), (2, 1, 2, 0), (0, 2, 1, 2), (0, 2, 2, 1), (2, 0, 1, 2), (2, 0, 2, 1)), ((0,),)), (((1, 2, 0, 2), (1, 2, 2, 0), (2, 1, 0, 2), (2, 1, 2, 0), (0, 2, 1, 2), (0, 2, 2, 1), (2, 0, 1, 2), (2, 0, 2, 1)), ((1,),)), (((1, 2, 0, 2), (1, 2, 2, 0), (2, 1, 0, 2), (2, 1, 2, 0), (0, 2, 1, 2), (0, 2, 2, 1), (2, 0, 1, 2), (2, 0, 2, 1)), ((2,),)), (((0, 1, 1, 2), (0, 1, 2, 1), (1, 0, 1, 2), (1, 0, 2, 1), (1, 2, 0, 1), (1, 2, 1, 0), (2, 1, 0, 1), (2, 1, 1, 0)), ((0,),)), (((0, 1, 1, 2), (0, 1, 2, 1), (1, 0, 1, 2), (1, 0, 2, 1), (1, 2, 0, 1), (1, 2, 1, 0), (2, 1, 0, 1), (2, 1, 1, 0)), ((1,),)), (((0, 1, 1, 2), (0, 1, 2, 1), (1, 0, 1, 2), (1, 0, 2, 1), (1, 2, 0, 1), (1, 2, 1, 0), (2, 1, 0, 1), (2, 1, 1, 0)), ((2,),)), (((0, 2, 0, 2), (0, 2, 2, 0), (2, 0, 0, 2), (2, 0, 2, 0)), ((0,),)), (((0, 2, 0, 2), (0, 2, 2, 0), (2, 0, 0, 2), (2, 0, 2, 0)), ((1,),)), (((0, 2, 0, 2), (0, 2, 2, 0), (2, 0, 0, 2), (2, 0, 2, 0)), ((2,),)), (((0, 2, 0, 1), (0, 2, 1, 0), (2, 0, 0, 1), (2, 0, 1, 0), (0, 1, 0, 2), (0, 1, 2, 0), (1, 0, 0, 2), (1, 0, 2, 0)), ((0,),)), (((0, 2, 0, 1), (0, 2, 1, 0), (2, 0, 0, 1), (2, 0, 1, 0), (0, 1, 0, 2), (0, 1, 2, 0), (1, 0, 0, 2), (1, 0, 2, 0)), ((1,),)), (((0, 2, 0, 1), (0, 2, 1, 0), (2, 0, 0, 1), (2, 0, 1, 0), (0, 1, 0, 2), (0, 1, 2, 0), (1, 0, 0, 2), (1, 0, 2, 0)), ((2,),)), (((0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)), ((0,),)), (((0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)), ((1,),)), (((0, 1, 0, 1), (0, 1, 1, 0), (1, 0, 0, 1), (1, 0, 1, 0)), ((2,),)))
        actual = SparseSymbolicTensor.form_matrix_entry(1, 1, full_indices, R)
        print(actual)

        
    def test_assemble_matrix(self):
        domain = QQ
        data = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        # R1 = Matrix([[0, 1], [2, 3]])
        # R2 = Matrix([[4, 5], [6, 7]])
        symops = [DomainMatrix.from_list_sympy(2, 2, block).convert_to(domain) for block in data]
        indices = [0, 1]
        def func(i, j, indices, symop, one=1):
            # Ignore indices for now
            # print('type(symop[(i, j)])=', type(symop[(i, j)]), 'symop[(i, j)].__repr__=', symop[(i, j)].__repr__())
            return symop.rep.getitem(i, j)
        actual = SparseSymbolicTensor.assemble_matrix(indices, symops, func, domain)
        data_flattened = [line for block in data for line in block]
        expected = DomainMatrix.from_list_sympy(4, 2, data_flattened).convert_to(domain)
        print('actual assemble_matrix = ', actual)
        print(actual.rep)
        print(expected.rep)
        print('rref=', actual.rref())
        print('rref=', expected.rref())
        self.assertEqual(actual, expected.to_sparse())

if __name__ == "__main__":
    import sys
    # sys.argv = ['', 'TestSparseSymbolicTensor.test_major_and_full_syms']
    # sys.argv = ['', 'TestSparseSymbolicTensor.test_form_matrix_entry']
    sys.argv = ['', 'TestSparseSymbolicTensor.test_assemble_matrix']
    unittest.main()
