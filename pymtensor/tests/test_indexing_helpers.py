import unittest

from pymtensor.indexing_helpers import (expand_all, expand2full, 
    sort_lists_convert2tuples, apply_mapping, form_matrix_entry)


class TestIndexingHelpers(unittest.TestCase):

    mapping = {0: ((0, 0),), 1: ((1, 1),), 2: ((2, 2),), 
               3: ((1, 2), (2, 1)),
               4: ((0, 2), (2, 0)),
               5: ((0, 1), (1, 0))}
    
    def test_apply_mapping(self):
        print("inside test_apply_mapping")
        mapping = self.mapping
        indices = (0, 3)
        actual = apply_mapping(indices, mapping)
        expected = (((0, 0),), ((1, 2), (2, 1)))
        self.assertEqual(actual, expected)

    def test_expand_all(self):
        print("inside test_expand_all")
        indices1 = ([(0, 0)], [(0, 1), (1, 0)],)
        indices2 = ([(1, 2)], [(2, 3), (3, 2)],)
        indices = (indices1, indices2)
        actual = expand_all(indices)
        print('actual=', actual)

    def test_expand2full(self):
        print("inside test_expand_to_full")
        mapping = self.mapping
        iter_indices = [(4, 5)]
        actual = expand2full(mapping, iter_indices)
        expected1 = ((0, 2, 0, 1), (0, 2, 1, 0), (2, 0, 0, 1), (2, 0, 1, 0))
        self.assertEqual(actual, expected1)
        iter_indices = [(5, 4)]
        actual = expand2full(mapping, iter_indices)
        expected2 = ((0, 1, 0, 2), (0, 1, 2, 0), (1, 0, 0, 2), (1, 0, 2, 0))
        self.assertEqual(actual, expected2)
        iter_indices = [(4, 5), (5, 4)]
        actual = expand2full(mapping, iter_indices)
        self.assertEqual(actual, expected1 + expected2)

    def test_sort_lists_convert2tuples(self):
        print('inside test_sort_lists_convert2tuples')
        iter_of_lists = ([4, 1], [1, 2], [1, 1], [3, 1], [2, 2])
        expected = ((1, 1), (1, 2), (2, 2), (3, 1), (4, 1))
        actual = sort_lists_convert2tuples(iter_of_lists)
        print('actual={}'.format(actual))
        self.assertEqual(actual, expected)
    
    def test_form_matrix_entry(self):
        # TODO: replace symop with SymPy variables to be fully general
        symop = [[1, 2, 3], 
                 [4, 5, 6], 
                 [7, 8, 9]]
        full_indices = (((0, 1),), ((0, 2), (1, 1)))
        # row = (0, 1) and cols = ((0, 2), (1, 1))
        # symop[0][0] * symop[1][2] + symop[0][1] + symop[1][3] - krondel(0, 1)
        expected = 1 * 6 + 2 * 5 - 0
        actual = form_matrix_entry(0, 1, full_indices, symop)
        self.assertEqual(actual, expected)
        # row = (0, 2) and cols = ((0, 2), (1, 1))
        # symop[0][0] * symop[2][2] + symop[0][1] * symop[2][1] - krondel(1, 1)
        expected = 1 * 9 + 2 * 8 - 1
        actual = form_matrix_entry(1, 1, full_indices, symop)
        self.assertEqual(actual, expected)
        # TODO: test for valid `full_indices`


if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()