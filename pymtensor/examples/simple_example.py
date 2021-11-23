from pymtensor.symmetry import RedSgSymOps
from pymtensor.sym_tensor import SymbolicTensor


def sym_reduce(indices, symbol_name, sym_group):
    # Choose a symmetry group (e.g. '3m')
    sg = RedSgSymOps()
    symops = sg(sym_group)
    print('symops=', symops)
    
    # Create a 5th-rank symbolic tensor with indices 1 and 3 interchangeable
    print(indices)
    print(symbol_name)
    st = SymbolicTensor(indices, symbol_name[0], start=1)
    
    # Solve for the unique tensor elements 
    fullsol, polyring = st.sol_details(symops)
    
    print(fullsol)



if __name__ == '__main__':
    tensors = {
        'a_{_1ijklmn}'    : ('ABB'  , '1st even electroelastic'),
        'b_{ijkl}'        : ('AB'   , 'Electrostrictive'),
        'c_{_2ijkl}'      : ('AA'   , '2nd order elastic'),
        'c_{_3ijklmn}'    : ('AAA'  , '3rd-order elastic'),
        'c_{_4ijklmnpq}'  : ('AAAA' , '4th order elastic'),
        'c_{_5ijklmnpqrs}': ('AAAAA', '5th order elastic'),
        'chi_{_2ij}'      : ('A'    , '2nd-order electric permeability'),
        'chi_{_3ijk}'     : ('A3'   , '3rd-order electric permeability'),
        'chi_{_4ijkl}'    : ('A4'   , '4th-order electric permeability'),
        'd_{_1ijklm}'     : ('aBB'  , '1st odd electroelastic'),
        'd_{_2ijklmnp}'   : ('aBBB' , '2nd odd electroelastic'),
        'd_{_3ijklm}'     : ('a3,b2', '3rd odd electroelastic'),
        'd_{_5ijklmnpqr}' : ('aBBBB', '5th odd electroelastic'),
        'e_{ijk}'         : ('aB'   , '1st-order piezoelectric'),
        'e_{ijklm}'       : ('aBB'  , '2nd-order piezoelectric'),
        'foo': ('a4,', 'foo'),
    }
    
    sym_group = '622'
    tensor_str = 'c_{_5ijklmnpqrs}'
    tensor_str = 'chi_{_2ij}'
    tensor_str = 'e_{ijk}'
    tensor_str = 'c_{_2ijkl}'
#     tensor_str = 'foo'
    # Uncomment a line above with the tensor of interest
    indices, description = tensors[tensor_str]
    symbol = tensor_str.split('_')[0]
    print('Symmetry group = {}'.format(sym_group))
    print('Tensor = {}, {}'.format(tensor_str, description))
    sym_reduce(indices, symbol, sym_group)
#     sym_reduce('ab', 'c', sym_group)