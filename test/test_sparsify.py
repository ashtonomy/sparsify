import sys
import os
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(root_dir, "sparsify"))

import unittest
from sparsify import sparsify
import numpy as np
from typing import Optional
from tqdm import tqdm

class TestSparsify(unittest.TestCase):

    def get_test_matrix(self, m: int, n: Optional[int]=None) -> np.ndarray:
        """
        Generate matrix for testing.
        """
        assert m > 0 and n > 0
        if n is not None:
            return np.random.rand(m, n)
        else:
            return np.random.rand(n, n)


    def test_sparsify(self):
        """
        Test handling of bad input arguments
        """
        self.assertRaises(ValueError, sparsify, 
                          self.get_test_matrix(3,3), method="l2")
        self.assertRaises(ValueError, sparsify, 
                          self.get_test_matrix(3,3), method="")
        

    def test_hybrid(self):
        """
        Test hybrid-(l1,l2) sampling sparsification
        """
        first_dims = [5, 10, 100]
        second_dims = [5, 10, 100]

        pbar = tqdm(total=len(first_dims)*len(second_dims), 
                    desc="Testing hybrid sparsification") 
        
        for i in first_dims:
            for j in second_dims:
                test_mat = self.get_test_matrix(i, j)
                
                for eps in [0.01, 0.1]:
                    for delta in [0.01, 0.1]:
                        sparse_mat = sparsify(test_mat, eps=eps, delta=delta)
                        
                        self.assertIsInstance(sparse_mat, np.ndarray)
                        self.assertEqual(test_mat.shape, sparse_mat.shape)
                        self.assertNotEqual(np.abs(sparse_mat).sum(), 0)

                        pbar.update(1)
        pbar.close()
                
    def test_l1(self):
        """
        Test l1 sampling sparsification
        """
        first_dims = [10, 100, 500]
        second_dims = [10, 100, 500]

        pbar = tqdm(total=len(first_dims)*len(second_dims), 
                    desc="Testing l1 sparsification") 

        for i in second_dims:
            for j in first_dims:
                test_mat = self.get_test_matrix(i, j)

                for eps in [0.01, 0.1]:
                    for delta in [0.01, 0.1]:
                        sparse_mat = sparsify(test_mat, eps=eps, 
                                              delta=delta, method="l1")
                        
                        self.assertIsInstance(sparse_mat, np.ndarray)
                        self.assertEqual(test_mat.shape, sparse_mat.shape)
                        self.assertNotEqual(np.abs(sparse_mat).sum(), 0)

                        pbar.update(1)
        pbar.close()

unittest.main()