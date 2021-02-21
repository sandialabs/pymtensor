from numpy import array, pi, einsum
from pymtensor.rot_tensor import (euler_rot, to_voigt, to_tensor, rot_tensor, 
                                rotu, rotx, roty, rotz)
from numpy import identity
from numpy.testing import assert_allclose, assert_array_equal, run_module_suite, TestCase
from sympy import symbols, MatrixSymbol


class TestRank1(TestCase):
    """
    Test numerical rotations routines `rotx`, `roty`, `rotz`, `rotu`.
    """
    
    def test_unitvecs(self):
        theta = pi / 2e0
        # 90-degree rotation around <1, 0, 0>
        rx = rotx(theta)
        rx_exact = array([[ 1,  0,  0],
                          [ 0,  0, -1],
                          [ 0,  1,  0]])
        assert_allclose(rx, rx_exact, verbose=True, atol=1e-12)
        # 90-degree rotation around <0, 1, 0>
        ry = roty(theta)
        ry_exact = array([[ 0,  0,  1],
                          [ 0,  1,  0],
                          [-1,  0,  0]])
        assert_allclose(ry, ry_exact, verbose=True, atol=1e-12)
        # 90-degree rotation around <0, 0, 1>
        rz = rotz(theta)
        rz_exact = array([[ 0, -1,  0],
                          [ 1,  0,  0],
                          [ 0,  0,  1]])
        assert_allclose(rz, rz_exact, verbose=True, atol=1e-12)
        # Test `rotu`
        arbrx = rotu(theta, array([1e0, 0, 0]))
        assert_array_equal(rx, arbrx, verbose=True)
        arbry = rotu(theta, array([0, 1e0, 0]))
        assert_array_equal(ry, arbry, verbose=True)
        arbrz = rotu(theta, array([0, 0, 1e0]))
        assert_array_equal(rz, arbrz, verbose=True)
        ghalf = rotx(pi / 4e0)
        print('Test successive rotations')
        unitvecs = identity(3, dtype=float)
        unitvecs[0,1]=-2 
        print(unitvecs)
        print(rot_tensor(rx, unitvecs))
        print(rot_tensor(ghalf, rot_tensor(ghalf, unitvecs)))
        ghalf = euler_rot(*[pi/4e0]*3)
        print('test single rotation')
        for vec in unitvecs:
            rot90CCWxy = rot_tensor(rx, vec)
            print(rot90CCWxy)
        g = euler_rot(*[pi/2e0]*3)
        print('test combined rotation')
        for vec in unitvecs:
            print(rot_tensor(g, vec))


class TestTensorTransforms(TestCase):
    
    def test_to_voigt(self):
        N = 3
        voigt = array([[ 1,  4,  6,  2,  5,  3],
                       [ 7, 10, 12,  8, 11,  9],
                       [13, 16, 18, 14, 17, 15]])
        tensor = array([[[1, 2, 3],
                         [2, 4, 5],
                         [3, 5, 6]],
                        [[7, 8, 9],
                         [8, 10, 11],
                         [9, 11, 12]],
                        [[13, 14, 15],
                         [14, 16, 17],
                         [15, 17, 18]]])
        assert_array_equal(voigt, to_voigt(tensor, order='Vasp'), 
                           err_msg="to_voigt failed", verbose=False)
        assert_array_equal(tensor, to_tensor(voigt, order='Vasp'), 
                           err_msg="to_tensor failed", verbose=False)


class TestSymbolic(TestCase):
    
    def test_symbolic_rotation(self):
        phi, theta = symbols(['phi', 'theta'])
        rotation = array(MatrixSymbol('R', 3, 3))
        tensor = array(MatrixSymbol('x', 3, 1)).flatten()
        foo = rot_tensor(rotation, tensor)
        print(foo)
        print(type(rotation), type(tensor))
        bar = einsum('ij,j', rotation, tensor)
        print(bar)
        assert_array_equal(foo, bar, 
                           err_msg="Tensor rotation failed", verbose=False)
        
        

if __name__ == "__main__":
#     run_module_suite()
    run_module_suite(argv=['', '--nocapture'])