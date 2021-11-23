import unittest

from pymtensor.indexing_helpers import (expand2full, sort_lists_convert2tuples,
    apply_mapping)


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



if __name__ == "__main__":
    #import sys;sys.argv = ['', 'Test.testName']
    unittest.main()