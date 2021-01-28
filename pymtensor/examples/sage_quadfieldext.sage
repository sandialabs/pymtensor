import time
from random import seed, randint
from sage.all import *


nmax = 7
print('Sage RREF timings\n')
methods = ['Qfield', 'General']
for method in methods:
    if method == 'Qfield':
        K.<rt3> = QuadraticField(3)
        print('Quadratic field matrix')
    else:
        print('General symbolic matrix') 
        rt3 = sqrt(3)
    for n in range(2, nmax+1):
        dims = [2**(n+1), 2**n]
        if method == 'Qfield':
            A = matrix(K, [[K(randint(-50, 50) * rt3 / randint(1, 50)) for i in range(dims[1])]
                           for j in range(dims[0])])
        else:
            A = Matrix([[randint(-50, 50) * rt3 / randint(1, 50) for i in range(dims[1])]
                        for j in range(dims[0])])
        tic = time.time()
        sol = A.rref()
        toc = time.time()                         
        print("dims = {}, RREF time = {:.4f} seconds".format(dims, toc - tic))
    print('')