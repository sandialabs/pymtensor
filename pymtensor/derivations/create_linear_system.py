from itertools import product

from sympy import Matrix, Symbol
from sympy.printing import latex

from pymtensor.symmetry import RedSgSymOps
from pymtensor.sym_tensor import SymbolicTensor


def create_linear_system(symbol, superscript, sym_group='622', tdim=2):
    sg = RedSgSymOps()
    print(sg.symops['6parZ3'])
    symops = sg(sym_group)
    symop = symops[0]
    R = Matrix(symop)
    Rsym = Matrix([[Symbol('R_{{{},{}}}'.format(i, j)) for j in range(1, 4)]
                   for i in range(1, 4)])
    print('Rsym=\n', latex(Rsym))
    print('R=\n', latex(R))
    print(symop)
    ivm, vm = SymbolicTensor.voigt_map(2)
    print(ivm)
    print(vm)
    indices0 = list(product(range(3), repeat=tdim))
    indices1 = list(product(range(1, 4), repeat=tdim))
    print(indices0)
    print(indices1)
    
    lhs = Matrix([[Rsym[I, i] * Rsym[J, j] for (i, j) in indices0]
                  for (I, J) in indices0])
    print(latex(lhs))
    vec = Matrix([[Symbol('c_{{{},{}}}'.format(i, j))] for (i, j) in indices1])
    print(latex(vec))
    vvec = Matrix([[Symbol('c_{{{}}}'.format(vm[k]+1))] for k in indices0])
    print(latex(vvec))
    lines = []
    frac_lines = []
    redfrac_lines = []
    symbol += '^{{{}}}'.format(superscript)
    for (I, J) in indices:
        line = '&'.join(["{}_{{{}{}}} {}_{{{}{}}}".format(symbol, I, i, symbol, J, j)
                         for (i, j) in indices])
        lines.append(line)
        Iint = int(I) - 1
        Jint = int(J) - 1
        frac_line = '&'.join(["{} \cdot {}".format(symop[Iint, int(i)-1], symop[Jint, int(j)-1])
                              for (i, j) in indices])
        frac_line = frac_line.replace('sqrt(3)', '\\sqrt{3}')
        frac_lines.append(frac_line)
        redfrac_line = '&'.join(["{}".format(4 * symop[Iint, int(i)-1] * symop[Jint, int(j)-1])
                                 for (i, j) in indices])
        redfrac_line = redfrac_line.replace('sqrt(3)', '\\sqrt{3}')
        redfrac_lines.append(redfrac_line)
    print('\\\\'.join(lines))
    print('\\\\'.join(frac_lines))
    print('\\\\'.join(redfrac_lines))


if __name__ == '__main__':
    superscript = '6\parallel Z_3'
    superscript = '2\parallel Z_1'
    create_linear_system('a', superscript)