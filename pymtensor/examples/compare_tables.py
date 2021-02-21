from os.path import join
import logging

from numpy import array
from sympy.parsing.sympy_parser import parse_expr
from sympy import Symbol

from pymtensor.symmetry import SgSymOps
from pymtensor.sym_tensor import SymbolicTensor
from pymtensor.rot_tensor import to_voigt


def parse_table(fname):
    data = []
    with open(fname, 'r') as f:
        line = f.readline()
        colnames = line.rstrip('\n').split(',')
        print('number of columns = {}'.format(len(colnames)))
        for line in f:
            data.append([val.strip() for val in line.rstrip('\n').split(',')])
    return colnames, array(data)
    

def compare_tables(indices_symbol, symbolnames, colnames, data):
    """
    Parameters
    ----------
    indices_symbol: string
        Tensor symbol, e.g. 'aB' for 2nd-order piezoelectric tensor
    indices: list or tuple of integers
        A list containing tuples with the same length as the number of indices 
        of the tensor of interest.
    colnames : list of strings
        The column names of Chris Kube's tables
    data : 2d array of strings
        The columns of data corresponding to Chris Kube's tensor symmetries
    """
    # Discrepancies
    # 6/mm should be 6/mmm
    # m\bar{3} should be \bar{3}m
    # m3m should be m\bar{3}m
    # 6m2 should be \bar{6}m2
    # m3 should be m\bar{3}
    tex_map = {# triclinic
               '1': '1', '1d': r'$\bar{1}$',
               # monoclinic
               '2': '2', 'm': r'$m$', '2/m': r'$2/m$',
               # orthorhombic
               '222': '222', 'mm2': r'$mm2$', 'mmm': r'$mmm$',
               # tetragonal
               '4': '4', '4d': r'$\bar{4}$', '4/m': '$4/m$', '422': '422',
               '4mm': r'$4mm$', '4d2m': r'$\bar{4}2m$', '4/mmm': r'$4/mmm$',
               # cubic
               '23': '23', 'm3d': r'$m\bar{3}$', '432': '432',
               '4d3m': r'$\bar{4}3m$', 
               'm3dm': r'$m\bar{3}m$',
               # hexagonal 
               '3': '3', '3d': r'$\bar{3}$', '32': '32', '3m': r'$3m$',
               '3dm': r'$\bar{3}m$', '6': '6', '6d': r'$\bar{6}$', 
               '6/m': r'$6/m$', '622': '622', '6mm': r'$6mm$', 
               '6dm2': r'$\bar{6}m2$', '6/mmm': r'$6/mmm$'}
    sg = SgSymOps()
    crystalclasses = sg.flat_ieee.keys()
    # Debug misspellings in Kube tables
#     print('number of crystalclasses = {}'.format(len(crystalclasses)))
#     for crystalclass in crystalclasses:
#         kube_key = tex_map[crystalclass]
#         if kube_key in colnames:
#             colnames.remove(kube_key)
#         else:
#             print(kube_key)
#     print(colnames)
    local_dict = dict((name, Symbol(name)) for name in symbolnames)
    first_letter = symbolnames[0][0]
    differences = False
    # Used for checking if the input file has the correct names
#     for crystalclass in crystalclasses:
#         print(colnames.index(tex_map[crystalclass]))
    for crystalclass in crystalclasses:
#         if crystalclass != '6':
#             continue
        print('Crystal class {}'.format(crystalclass))
        logging.info('Crystal class {}'.format(crystalclass))
        symops = sg(crystalclass)
        st = SymbolicTensor(indices_symbol, first_letter)
        fullsol, R = st.sol_details(symops)
#         print(fullsol)
        inputcol = data[:, colnames.index(tex_map[crystalclass])]
        for inputval, symbolname in zip(inputcol, symbolnames):
            val = fullsol[symbolname]
            parsedval = parse_expr(inputval, local_dict)
            if R(val) != R(parsedval):
                differences = True
                print('{}: val = {}, inputval = {}'.format(symbolname, val, inputval))
                logging.info('{}: val = {}, inputval = {}'.format(symbolname, val, inputval))
    print('Differences = {}'.format(differences))
    logging.info('Differences = {}'.format(differences))


def capture_comparisons(tname, indices_symbols, fext='.csv', tdir=None, 
                        logdir=None):
    print('{} comparison\n'.format(tname))
    logname = tname + '_comparison.log'
    if logdir is not None:
        logname = join(logdir, logname)
    # `filemode` argument opens a new file each run
    logging.basicConfig(filename=logname, level=logging.INFO, 
                        format='%(message)s', filemode='w')
    fname = tname + fext
    if tdir is not None:
        fname = join(tdir, fname)
    colnames, data = parse_table(fname)
    symbolnames = data[:, 0]
    compare_tables(indices_symbols, symbolnames, colnames, data)


if __name__ == '__main__':
    tdir = 'tensor_tables'
    jobs = {
        'a_{_1ijklmn}': 'ABB', # 1st even electroelastic
        'b_{ijkl}': 'AB', # Electrostrictive
        'c_{_2ijkl}': 'AA', # 2nd order elastic
        'c_{_3ijklmn}': 'AAA', # 3rd-order elastic
        'c_{_4ijklmnpq}': 'AAAA', # 4th order elastic
        'c_{_5ijklmnpqrs}': 'AAAAA', # 5th order elastic
        'chi_{_2ij}': 'A', # 2nd-order electric permeability
        'chi_{_3ijk}': 'A3,', # 3rd-order electric permeability
        'chi_{_4ijkl}': 'A4,', # 4th-order electric permeability
        'd_{_1ijklm}': 'aBB', # 1st odd electroelastic
        'd_{_2ijklmnp}': 'aBBB', # 2nd odd electroelastic
        'd_{_3ijklm}': 'a3,b2', # 3rd odd electroelastic
        'd_{_5ijklmnpqr}': 'aBBBB', # 5th odd electroelastic
        'e_{ijk}': 'aB', # 1st-order piezoelectric
        'e_{ijklm}': 'aBB', # 2nd-order piezoelectric
        }
#     tname = 'd_{_5ijklmnpqr}'
#     tname = 'c_{_5ijklmnpqrs}'
#     capture_comparisons(tname, jobs[tname], tdir=tdir)
    for tname, indices_symbols in jobs.items():
        capture_comparisons(tname, indices_symbols, tdir=tdir, logdir=tdir)