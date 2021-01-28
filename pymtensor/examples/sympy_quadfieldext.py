import time
from random import seed, randint
from sympy import Matrix, sqrt
from sympy.polys.domains import QQ
from sympy.polys.solvers import DomainMatrix


nmax = 7
print('SymPy RREF timings\n')
methods = ['Qfield', 'General']
for method in methods:
    if method == 'Qfield':
        print('Quadratic field matrix')
    else:
        print('General symbolic matrix') 
    for n in range(2, nmax+1):
        dims = [2**(n+1), 2**n]
        rt3 = sqrt(3)
        if method == 'Qfield':
            A = DomainMatrix([[randint(-50, 50) * rt3 / randint(1, 50)
                               for i in range(dims[1])]
                              for j in range(dims[0])], shape=dims, 
                              domain=QQ.algebraic_field(sqrt(3)))
        else:
            A = Matrix([[randint(-50, 50) * rt3 / randint(1, 50)
                         for i in range(dims[1])]
                        for j in range(dims[0])])
        tic = time.time()
        sol = A.rref()
        toc = time.time()                         
        print("dims = {}, RREF time = {:.4f} seconds".format(dims, toc - tic))
    print('')