from sympy import Matrix, sqrt, symbols
from sympy.polys.domains import QQ, ZZ
from sympy.polys.matrices.sdm import SDM
from sympy.polys.solvers import DomainMatrix

from pymtensor.symfinder import SparseSymbolicTensor
from pymtensor.symmetry import RedSgSymOps


def test_interpret_solution():
    sst = SparseSymbolicTensor('AAAA', 'c')
    sg = RedSgSymOps()
    symops = sg("3dm")
    symops = sg("2")
    print('symops=', symops)
    domain = QQ.algebraic_field(sqrt(3))
    sol, pivots = sst.apply_symmetry(symops, domain)
    nonzero_vars, unique_vars = sst.interpret_solution(sol, pivots)
    print('nonzero_vars = {}, unique_vars={}'.format(nonzero_vars, unique_vars))


if __name__ == '__main__':
    test_interpret_solution()