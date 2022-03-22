from numpy import array
from sympy import cos, exp, Matrix, sin, sqrt, Rational, Symbol
from sympy.polys.matrices import DomainMatrix

from pymtensor.symmetry import RedSgSymOps
from pymtensor.sym_tensor import SymbolicTensor


def sym_reduce(indices, symbol_name, sym_group):
    # The inf||Z3 symmetry operator
    # a = Rational(1, 5)
    # b = Rational(2, 5) * sqrt(6)
    # symops = [array([[ a, -b,  0], [ b,  a,  0], [ 0,  0,  1]])]
    rsso = RedSgSymOps()
    symops = [rsso.symops[val] for val in ('2parZ3',
                                           '3parZ3', '3dparZ3', 
                                           '4parZ3', '4dparZ3', 
                                           '6parZ3', '6dparZ3')]
    print(symops)
    print(indices)
    print(symbol_name)
    st = SymbolicTensor(indices, symbol_name[0], start=1)
    
    # Solve for the unique tensor elements 
    fullsol, polyring = st.sol_details(symops)
    
    print(fullsol)


def complex_exponential_example():
    x = Symbol('x')
    A = Matrix([val.rewrite(exp).expand() for val in 
                (sin(x) * cos(x), cos(x)**3,-cos(x), sin(x)**2)]).reshape(2, 2)
    Adm = DomainMatrix.from_Matrix(A)
    print(Adm)
    print(Adm.rref())


if __name__ == '__main__':
    tensors = {
        'c_{_2ijkl}'      : ('AA'   , '2nd order elastic'),
        'c_{_3ijklmn}'    : ('AAA'  , '3rd-order elastic'),
        'c_{_4ijklmnpq}'  : ('AAAA' , '4th order elastic'),
        'c_{_5ijklmnpqrs}': ('AAAAA', '5th order elastic'),
    }
    
    sym_group = 'oo'
    tensor_str = 'c_{_2ijkl}'
    tensor_str = 'c_{_3ijklmn}'
    # tensor_str = 'c_{_4ijklmnpq}'
#     tensor_str = 'foo'
    # Uncomment a line above with the tensor of interest
    indices, description = tensors[tensor_str]
    symbol = tensor_str.split('_')[0]
    print('Symmetry group = {}'.format(sym_group))
    print('Tensor = {}, {}'.format(tensor_str, description))
    sym_reduce(indices, symbol, sym_group)
#     sym_reduce('ab', 'c', sym_group)
    complex_exponential_example()