import time
from random import seed, randint
from sympy import Matrix, sqrt
from sympy.polys.domains import QQ, ZZ, QQ_gmpy
from sympy.polys.solvers import DomainMatrix


nmax = 7
print('SymPy RREF timings\n')
# methods = ['Qfield', 'General']
methods = ['Qfield']
for method in methods:
    if method == 'Qfield':
        print('Quadratic field matrix')
    else:
        print('General symbolic matrix') 
    for n in range(2, nmax+1):
        dims = [2**(n+1), 2**n]
        rt3 = sqrt(3)
        tic = time.time()
        if method == 'Qfield':
            K = QQ.algebraic_field(sqrt(3))
#             K = ZZ.algebraic_field(sqrt(3))
            rt3 = K.convert(rt3)
#             A = DomainMatrix([[K.convert(randint(-50, 50)) * rt3 / K.convert(randint(1, 50))
#                                for i in range(dims[1])]
#                               for j in range(dims[0])], shape=dims, 
#                               domain=K)
#             A = DomainMatrix([[randint(-50, 50)
#                                for i in range(dims[1])]
#                               for j in range(dims[0])], shape=dims, 
#                               domain=QQ)
            A = DomainMatrix([[K.convert(randint(-250, 250) * rt3)
                               for i in range(dims[1])]
                              for j in range(dims[0])], shape=dims, 
                              domain=K)
        else:
            A = Matrix([[randint(-50, 50) * rt3 / randint(1, 50)
                         for i in range(dims[1])]
                        for j in range(dims[0])])
        sol = A.rref()#(normalize_last=True)
        toc = time.time()                         
        print("dims = {}, RREF time = {:.4f} seconds".format(dims, toc - tic))
    print('')
