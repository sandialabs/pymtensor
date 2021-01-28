from numpy import array, cos, dot, empty, nditer, sin, tensordot
from numpy.linalg import norm
from itertools import chain, permutations, product



# TODO: add more error checking for input arrays
def to_tensor(voigt, order='Voigt'):
    """
    Convert from compressed to standard tensor notation.
    
    1 -> 11, 2 -> 22, 3 -> 33
    Voigt: 4 -> 23 & 32, 5 -> 13 & 31, 6 -> 12 & 21
    VASP:  4 -> 12 & 21, 5 -> 23 & 32, 6 -> 13 & 31
    """
    index_map = {0: (0, 0), 1: (1, 1), 2: (2, 2)}
    if order.lower() == 'voigt':
        index_map.update({3: (1, 2), 4: (0, 2), 5: (0, 1)})
    else:
        index_map.update({3: (0, 1), 4: (1, 2), 5: (2, 0)})
    it = nditer(voigt, flags=['multi_index'])
    # TODO: Replace `if` statements using SymPy's `factorint` function.
    if voigt.size == 3*3:
        res = voigt.copy()
    if voigt.size == 3*6:
        res = empty((3,)*3, dtype=voigt.dtype)
        while not it.finished:
            i, jk = it.multi_index
            j, k = index_map[jk]
    #         print(it[0], it.multi_index)
            res[i, j, k] = it[0]
            res[i, k, j] = it[0]
            it.iternext()
    elif voigt.size == 6*6:
        res = empty((3,)*4, dtype=voigt.dtype)
        while not it.finished:
            ij, kl = it.multi_index
            (i, j), (k, l) = [index_map[(ind)] for ind in [ij, kl]]
    #         print(it[0], it.multi_index)
            res[i, j, k, l] = it[0]
            res[i, j, l, k] = it[0]
            res[j, i, k, l] = it[0]
            res[j, i, l, k] = it[0]
            it.iternext()
    elif voigt.size == 3*6*6:
        res = empty((3,)*5, dtype=voigt.dtype)
        while not it.finished:
            i, jk, lm = it.multi_index
            (j, k), (l, m) = [index_map[(ind)] for ind in [jk, lm]]
    #         print(it[0], it.multi_index)
            res[i, j, k, l, m] = it[0]
            res[i, j, k, m, l] = it[0]
            res[i, k, j, l, m] = it[0]
            res[i, k, j, m, l] = it[0]
            it.iternext()
    elif voigt.size == 6*6*6:
        res = empty((3,)*6, dtype=voigt.dtype)
        while not it.finished:
            tensorindices = [index_map[ind] for ind in it.multi_index]
            # Permute each pair of indices based on symmetry arguments and
            # find all possible combinations of indices
            # 3! * 2^3 = 48 possibilities per Voigt index
            for index in product(*(permutations(index) for index in tensorindices)):
                flatindex = tuple(chain.from_iterable(index))
#                 print(flatindex)
                res[flatindex] = it[0]
            it.iternext()
    return res
        

def to_voigt(tensor, order='Voigt'):
    """
    Convert from standard tensor to compressed notation.
    
    1 -> 11, 2 -> 22, 3 -> 33
    Voigt: 4 -> 23 & 32, 5 -> 13 & 31, 6 -> 12 & 21
    VASP:  4 -> 12 & 21, 5 -> 23 & 32, 6 -> 13 & 31
    """
    index_map = {(0, 0): 0, (1, 1): 1, (2, 2): 2}
#     (0, 1): 3, (1, 0): 3, (1, 2): 4, (2, 1): 4, (2, 0): 5, (0, 2): 5}
    if order.lower() == 'voigt':
        index_map.update({(1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, 
                          (0, 1): 5, (1, 0): 5})
    else:
        index_map.update({(0, 1): 3, (1, 0): 3, (1, 2): 4, (2, 1): 4, 
                          (2, 0): 5, (0, 2): 5})
    dims = len(tensor.shape)
    it = nditer(tensor, flags=['multi_index', 'refs_ok'])
    if dims == 2:
        res = tensor.copy()
    if dims == 3:
        res = empty((3, 6), dtype=tensor.dtype)
        while not it.finished:
            i, j, k = it.multi_index
            jk = index_map[(j, k)]
    #         print(it[0], it.multi_index)
            res[i, jk] = it[0].item()
            it.iternext()
    elif dims == 4:
        res = empty((6, 6), dtype=tensor.dtype)
        while not it.finished:
            i, j, k, l = it.multi_index
            ij, kl = [index_map[(ind1, ind2)] for ind1, ind2
                      in [(i, j), (k, l)]]
            res[ij, kl] = it[0].item()
            it.iternext()
    elif dims == 5:
        res = empty((3, 6, 6), dtype=tensor.dtype)
        while not it.finished:
            i, j, k, l, m = it.multi_index
            jk, lm = [index_map[(ind1, ind2)] for ind1, ind2
                      in [(j, k), (l, m)]]
            res[i, jk, lm] = it[0].item()
            it.iternext()
    elif dims == 6:
        res = empty((6, 6, 6), dtype=tensor.dtype)
        while not it.finished:
            I, J, K = [index_map[indices] for indices
                       in [it.multi_index[2*i:2*i+2] for i in range(3)]]
            res[I, J, K] = it[0].item()
            it.iternext()
    return res


def rotu(theta, u):
    # Normalize u
    ux, uy, uz = u / norm(u)
    c = cos(theta)
    omc = 1 - c
    s = sin(theta)
    return array([[c + ux**2 * omc,        ux * uy * omc - uz * s, ux * uz * omc + uy * s],
                  [uy * ux * omc + uz * s, c + uy**2 * omc,        uy * uz * omc - ux * s],
                  [uz * ux * omc - uy * s, uz * uy * omc + ux * s, c + uz**2 * omc       ]])


def rotx(theta):
    c = cos(theta)
    s = sin(theta)
    return array([[1, 0,  0],
                  [0, c, -s],
                  [0, s,  c]])


def roty(theta):
    c = cos(theta)
    s = sin(theta)
    return array([[ c, 0,  s],
                  [ 0, 1,  0],
                  [-s, 0,  c]])


def rotz(theta):
    c = cos(theta)
    s = sin(theta)
    return array([[c, -s, 0],
                  [s,  c, 0],
                  [0,  0, 1]])


def euler_rot(alpha, beta, gamma):
    """
    Rotation by proper Euler angles
    """
    rot = rotz(alpha)
    rot = dot(rotx(beta), rot)
    rot = dot(rotz(gamma), rot)
    return rot


# Based on this StackOverflow question:
# https://stackoverflow.com/questions/4962606/fast-tensor-rotation-with-numpy/42347571#42347571
def rot_tensor(rotation, tensor):
    """
    Apply a rotation to a tensor of arbitrary dimensions.
    
    Parameters
    ----------
    rotation : 3x3 array
    tensor : 3^n array
    
    Example
    -------
    (rotated d_ijk) = A_il A_jm A_kn d_lmn
    """
    dims = len(tensor.shape)
    res = tensor #  We don't need a deep copy because tensordot returns a new array
    r = rotation
#     r = rotation.T
    for i in range(dims):
        res = tensordot(r, res, axes=((-1),(-1)))
#         res = tensordot(res, r, axes=((0),(0)))
    return res