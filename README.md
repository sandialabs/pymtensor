
# PyMTensor

Material tensor Python package for computing the unique tensor elements for material tensors of arbitrary order and arbitrary crystallographic point group.

Copyright 2021 National Technology & Engineering Solutions of Sandia, LLC (NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S. Government retains certain rights in this software.


# Installation
To get started right away the pymtensor code can be added to your python path:

```bash
export PYTHONPATH=$PYTHONPATH:/path/to/pymtensor
```

The package can also be installed via the setuptools setup.py script:

```bash
python setup.py
```

PyMTensor requires Python 3.5 or higher, NumPy version 1.6 or greater, and SymPy version 1.7 or greater.
The optional dependency gmpy2 for SymPy is highly recommended for speed improvements.
TODO: add Windows installation instructions

# Quick Start
The following code snippet will compute the unique tensor elements of a 5th-rank tensor with indices 1 and 3 interchangeable.

```python
from pymtensor.symmetry import SgSymOps, SymbolicTensor
from pymtensor.rot_tensor import to_voigt

# Choose a symmetry group (e.g. '3m')
sg = SgSymOps()
symops = sg('3m')

# Create a 5th-rank symbolic tensor with indices 1 and 3 interchangeable
st = SymbolicTensor("abacd", 'c')

# Solve for the unique tensor elements 
fullsol, polyring = st.sol_details(symops)

# Convert to Voigt notation and print the tensor
print(fullsol)
```

For more examples please refer to the `pymtensor/examples/compare_tables.py` file.
