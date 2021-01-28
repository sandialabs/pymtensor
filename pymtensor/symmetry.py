from collections import defaultdict
from itertools import permutations, product
import logging
import time
import os
import psutil

from numpy import (abs, argsort, array, atleast_2d, empty, empty_like, eye, max, min, 
                   nditer, nonzero, rint, set_printoptions)
from scipy.linalg import lu, svd
from sympy import Add, acos, cos, Matrix, Mul, pi, sin, sqrt, Symbol
from sympy.polys.rings import ring
from sympy.polys.solvers import solve_lin_sys

from pymtensor.rot_tensor import rot_tensor, to_voigt
from abc import abstractstaticmethod


def round_if_safe(val, atol):
    ival = rint(val)
    if abs(ival - val) < atol:
        return int(ival)
    else:
        return val 


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


def deg2rad(deg):
    """Convert from degrees to radians."""
    return deg * pi / 180


def norm(v):
    return sqrt(v.dot(v))


def angle(v1, v2):
    """Compute the angle between two vectors using the dot product."""
    lhs = v1.dot(v2)
    v1mag = norm(v1)
    v2mag = norm(v2)
    theta = acos(lhs / (v1mag * v2mag))
    return theta


def reflectu(u):
    # See https://en.wikipedia.org/wiki/Transformation_matrix#Reflection_2
    u = u / norm(u)
    return eye(3, dtype=int) - 2 * u.reshape((3, 1)).dot(u.reshape((1, 3)))


def rotu(theta, u):
    # Normalize u
    ux, uy, uz = u / norm(u)
    c = cos(theta)
    omc = 1 - c
    s = sin(theta)
    return array([[c + ux**2 * omc,        ux * uy * omc - uz * s, ux * uz * omc + uy * s],
                  [uy * ux * omc + uz * s, c + uy**2 * omc,        uy * uz * omc - ux * s],
                  [uz * ux * omc - uy * s, uz * uy * omc + ux * s, c + uz**2 * omc       ]])


class SgSymOps(object):
    def __init__(self):
        self.groups = {}
        self._E = eye(3, dtype=int)  # Identity
        self._I = -self._E  # inversion
        self._SIGz = array([[ 1,  0,  0],
                            [ 0,  1,  0],
                            [ 0,  0, -1]])
        # Store the symmetry operators for easy testing
        self.hexops, hexgroup = self.init_hexagonal()
        self.ops, groups = self.init_nonhexagonal()
        groups['hexagonal'] = hexgroup
        self.groups = groups
        self.flat = self.flatten_dict(groups)
        self._init_ieee()
    
    def __call__(self, cname, ieee=True):
        """
        Parameters
        ----------
        cname: string
            Class name
        ieee: boolean
            Use IEEE standard
        """
        if ieee:
            return self.flat_ieee[cname]
        else:
            return self.flat[cname]
    
    @staticmethod
    def flatten_dict(d):
        flat = {}
        for group in d.values():
            for key, val in group.items():
                flat[key] = val
        return flat
    
    def _init_ieee(self):
        ops = self.ops
        E = ops["E"]; C2y = ops["C2y"]; SIGy = ops["SIGy"]; I = ops["I"]
        groups = self.groups.copy()
        # The y-axis is the IEEE axis of symmetry for monoclinic symmetry
        mono = {"2": (E, C2y),
                "m": (E, SIGy),
                "2/m": (E, C2y, I, SIGy),
               }
        groups["monoclinic"] = mono
        tetragonal = groups["tetragonal"]
        tetragonal["4d2m"] = tetragonal.pop("4d2m(1)")
        tetragonal.pop("4d2m(2)")
        hexagonal = groups["hexagonal"]
        hexagonal.pop("32(1)")
        hexagonal["32"] = hexagonal.pop("32(2)")
#         from sympy import Rational
#         half = Rational(1, 2)
#         hexagonal["32"] = (array([[-half, sqrt(3) * half, 0],
#                                   [-sqrt(3) * half, -half, 0],
#                                   [0, 0, 1]]),
#                            array([[1, 0, 0],
#                                   [0, -1, 0],
#                                   [0, 0, -1]]),
#         )
        hexagonal.pop("3m(1)")
        hexagonal["3m"] = hexagonal.pop("3m(2)")
        hexagonal.pop("3dm(1)")
        hexagonal["3dm"] = hexagonal.pop("3dm(2)")
        hexagonal["6dm2"] = hexagonal.pop("6dm2(1)")
        hexagonal.pop("6dm2(2)")
        self.groups_ieee = groups
        self.flat_ieee = self.flatten_dict(groups)

    def init_nonhexagonal(self):
        # a1 != a2 != a3 and alpha != beta != gamma != 90 (triclinic)
        E = self._E  # Identity
        I = self._I  # inversion
        tri = {"1": (E,),
               "1d": (E, I)
               }
        # a1 != a2 != a3 and alpha = beta = 90 != gamma (monoclinic)
        C2z = rotz(deg2rad(180))  # 180 rotation about a3
        SIGz = self._SIGz
        mono = {"2": (E, C2z),
                "m": (E, SIGz),
                "2/m": (E, C2z, I, SIGz),
                }
        # a1 != a2 != a3 and alpha = beta = gamma = 90 (orthorhombic)
        C2x = rotx(deg2rad(180))  # 180 rotation about a1
        C2y = roty(deg2rad(180))  # 180 rotation about a2
        SIGx = array([[-1,  0,  0],
                      [ 0,  1,  0],
                      [ 0,  0,  1]])
        SIGy = array([[ 1,  0,  0],
                      [ 0, -1,  0],
                      [ 0,  0,  1]])
        ortho = {"222": (E, C2x, C2y, C2z),
                 "mm2": (E, C2z, SIGx, SIGy),
                 "mmm": (E, C2x, C2y, C2z, I, SIGx, SIGy, SIGz),
                }
        # a1 = a2 != a3 and alpha = beta = gamma = 90 (tetragonal)
        a1 = array([1, 0, 0])
        a2 = array([0, 1, 0])
        C2a = rotu(deg2rad(180), a1 + a2)
        C2b = rotu(deg2rad(180), a1 - a2)
        C4zp = rotz(deg2rad(90))
        C4zm = rotz(deg2rad(270))
        SIGda = reflectu(a1 + a2)
        SIGdb = reflectu(a1 - a2)
        S4zp = SIGz.dot(rotz(deg2rad(90)))
        S4zm = SIGz.dot(rotz(deg2rad(270)))
        tet = {"4": (E, C4zp, C4zm, C2z),
               "4d": (E, S4zp, S4zm, C2z),
               "4/m": (E, C4zp, C4zm, C2z, I, SIGz),
               "422": (E, C4zp, C4zm, C2x, C2y, C2z, C2a, C2b),
               "4mm": (E, C4zp, C4zm, C2z, SIGx, SIGy, SIGda, SIGdb),
               "4d2m(1)": (E, S4zp, S4zm, C2x, C2y, C2z, SIGda, SIGdb),
               "4d2m(2)": (E, S4zp, S4zm, C2z, C2a, C2b, SIGx, SIGy),
               "4/mmm": (E, C4zp, C4zm, C2x, C2y, C2z, C2a, C2b, I, 
                         S4zp, S4zm, SIGx, SIGy, SIGz, SIGda, SIGdb),
              }
        # a1 = a2 = a3 and alpha = beta = gamma = 90 (cubic)
        a3 = array([0, 0, 1])
        C2c = rotu(deg2rad(180), a1 + a3)
        C2d = rotu(deg2rad(180), a2 + a3)
        C2e = rotu(deg2rad(180), a1 - a3)
        C2f = rotu(deg2rad(180), a2 - a3)
        C31p = rotu(deg2rad(120), a1 + a2 + a3)
        C31m = rotu(deg2rad(240), a1 + a2 + a3)
        C32p = rotu(deg2rad(120), -a1 - a2 + a3)
        C32m = rotu(deg2rad(240), -a1 - a2 + a3)
        C33p = rotu(deg2rad(120), a1 - a2 - a3)
        C33m = rotu(deg2rad(240), a1 - a2 - a3)
        C34p = rotu(deg2rad(120), -a1 + a2 - a3)
        C34m = rotu(deg2rad(240), -a1 + a2 - a3)
        C4xp = rotx(deg2rad(90))
        C4xm = rotx(deg2rad(270))
        C4yp = roty(deg2rad(90))
        C4ym = roty(deg2rad(270))
        C4zp = rotz(deg2rad(90))
        C4zm = rotz(deg2rad(270))
        SIGdc = reflectu(a1 + a3)
        SIGdd = reflectu(a2 + a3)
        SIGde = reflectu(a1 - a3)
        SIGdf = reflectu(a2 - a3)
        S61p = reflectu(a1 + a2 + a3).dot(rotu(deg2rad(60), a1 + a2 + a3))
        S61m = reflectu(a1 + a2 + a3).dot(rotu(deg2rad(300), a1 + a2 + a3))
        S62p = reflectu(-a1 - a2 + a3).dot(rotu(deg2rad(60), -a1 - a2 + a3))
        S62m = reflectu(-a1 - a2 + a3).dot(rotu(deg2rad(300), -a1 - a2 + a3))
        S63p = reflectu(a1 - a2 - a3).dot(rotu(deg2rad(60), a1 - a2 - a3))
        S63m = reflectu(a1 - a2 - a3).dot(rotu(deg2rad(300), a1 - a2 - a3))
        S64p = reflectu(-a1 + a2 - a3).dot(rotu(deg2rad(60), -a1 + a2 - a3))
        S64m = reflectu(-a1 + a2 - a3).dot(rotu(deg2rad(300), -a1 + a2 - a3))
        S4xp = SIGx.dot(rotx(deg2rad(90)))
        S4xm = SIGx.dot(rotx(deg2rad(270)))
        S4yp = SIGy.dot(roty(deg2rad(90)))
        S4ym = SIGy.dot(roty(deg2rad(270)))
        cubic = {"23": (E, C31p, C31m, C32p, C32m, C33p, C33m, C34p, C34m, 
                        C2x, C2y, C2z),
                 "m3d": (E, C31p, C31m, C32p, C32m, C33p, C33m, C34p, C34m,
                         C2x, C2y, C2z, I, S61p, S61m, S62p, S62m, S63p, S63m, 
                         S64p, S64m, SIGx, SIGy, SIGz),
                 "432": (E, C31p, C31m, C32p, C32m, C33p, C33m, C34p, C34m,
                         C2x, C2y, C2z, C2a, C2b, C2c, C2d, C2e, C2f,
                         C4xp, C4xm, C4yp, C4ym, C4zp, C4zm),
                 "4d3m": (E, C31p, C31m, C32p, C32m, C33p, C33m, C34p, C34m,
                          C2x, C2y, C2z, SIGda, SIGdb, SIGdc, SIGdd, SIGde, 
                          SIGdf, S4xp, S4xm, S4yp, S4ym, S4zp, S4zm),
                 "m3dm": (E, C31p, C31m, C32p, C32m, C33p, C33m, C34p, C34m,
                          C2x, C2y, C2z, C4xp, C4xm, C4yp, C4ym, C4zp, C4zm,
                          C2a, C2b, C2c, C2d, C2e, C2f, I, 
                          S61p, S61m, S62p, S62m, S63p, S63m, S64p, S64m, 
                          SIGx, SIGy, SIGz, SIGda, SIGdb, SIGdc, SIGdd, SIGde, 
                          SIGdf, S4xp, S4xm, S4yp, S4ym, S4zp, S4zm),
                }
        ops = {"E": E, "I": I, "C2z": C2z, "SIGz": SIGz, "C2x": C2x,
               "C2y": C2y, "SIGx": SIGx, "SIGy": SIGy, "C2a": C2a, "C2b": C2b,
               "C4z+": C4zp, "C4z-": C4zm, "SIGda": SIGda, "SIGdb": SIGdb,
               "S4z+": S4zp, "S4z-": S4zm, "C2c": C2c, "C2d": C2d, "C2e": C2e,
               "C2f": C2f, "C31+": C31p, "C31-": C31m, "C32+": C32p, 
               "C32-": C32m, "C33+": C33p, "C33-": C33m, "C34+": C34p, 
               "C34-": C34m, "C4x+": C4xp, "C4x-": C4xm, "C4y+": C4yp, 
               "C4y-": C4ym, "C4z+": C4zp, "C4z-": C4zm, "SIGdc": SIGdc, 
               "SIGdd": SIGdd, "SIGde": SIGde, "SIGdf": SIGdf, "S61+": S61p,
               "S61-": S61m, "S62+": S62p, "S62-": S62m, "S63+": S63p, 
               "S63-": S63m, "S64+": S64p, "S64-": S64m, "S4x+": S4xp,
               "S4x-": S4xm, "S4y+": S4yp, "S4y-": S4ym
               }
        groups = {"triclinic": tri, "monoclinic": mono, "orthorhombic": ortho,
                  "tetragonal": tet, "cubic": cubic}
        return ops, groups
    
    def init_hexagonal(self):
        """
        All hexagonal classes.
        """
        """
        Dictionary of all symmetry operations for hexagonal lattices.
        
        Without loss of generality we set the following basis vectors 
        satisfying the conditions a1 = a2, alpha = beta = 90, and gamma = 120
        a1 = <1,0,0>
        a2 = <1/2,sqrt(3)/2,0>
        a3 = <0,0,1>
        """
        SIGz = self._SIGz
        a1 = array([1, 0, 0])
        a2 = array([cos(deg2rad(120)), sin(deg2rad(120)),  0])
#         a3 = array([0, 0, 1])
        E = self._E  # Identity
        I = self._I  # inversion
        C2 = rotz(deg2rad(180))  # 180 rotation about a3
        C3p = rotz(deg2rad(120))  # 120 rotation about a3
        C3m = rotz(deg2rad(240))  # 240 rotation about a3
        C6p = rotz(deg2rad(60))  # 60 rotation about a3
        C6m = rotz(deg2rad(300))  # 300 rotation about a3
        C21_p = rotu(deg2rad(180), a1 + 2 * a2)  # 180 rotation about a1+2*a2
        C22_p = rotu(deg2rad(180), 2 * a1 + a2)  # 180 rotation about 2*a1+a2
        C23_p = rotu(deg2rad(180), a1 - a2)  # 180 rotation about a1-a2
        C21_pp = rotu(deg2rad(180), a1)  # 180 rotation about a1
        C22_pp = rotu(deg2rad(180), a2)  # 180 rotation about a2
        C23_pp = rotu(deg2rad(180), a1 + a2)  # 180 rotation about a1+a2
        SIGh = SIGz  # reflection in plane perpendicular to a3
        # 120 rotation about a3 followed by reflection in plane perpendicular to a3
        S3p = SIGz.dot(rotz(deg2rad(120)))
        # 120 rotation about a3 followed by reflection in plane perpendicular to a3
        S3m = SIGz.dot(rotz(deg2rad(240)))
        # 60 rotation about a3 followed by reflection in plane perpendicular to a3
        S6p = SIGz.dot(rotz(deg2rad(60)))
        # 60 rotation about a3 followed by reflection in plane perpendicular to a3
        S6m = SIGz.dot(rotz(deg2rad(300)))
        SIGd1 = reflectu(a1 + 2 * a2)
        SIGd2 = reflectu(2 * a1 + a2)
        SIGd3 = reflectu(a1 - a2)
        SIGv1 = reflectu(a1)
        SIGv2 = reflectu(a2)
        SIGv3 = reflectu(a1 + a2)
        ops = {"E": E, "C2": C2, "C3+": C3p, "C3-": C3m, "C6+": C6p, "C6-": C6m,
               "C21'": C21_p, "C22'": C22_p, "C23'": C23_p, 
               "C21''": C21_pp, "C22''": C22_pp, "C23''": C23_pp, "I": I,
               "SIGh": SIGh, "S3+": S3p, "S3-": S3m, "S6+": S6p, "S6-": S6m,
               "SIGd1": SIGd1, "SIGd2": SIGd2, "SIGd3": SIGd3, "SIGv1": SIGv1,
               "SIGv2": SIGv2, "SIGv3": SIGv3}
        group = {"3": (E, C3p, C3m),
                 "3d": (E, C3p, C3m, I, S6p, S6m),
                 "32(1)": (E, C3p, C3m, C21_p, C22_p, C23_p),
                 "32(2)": (E, C3p, C3m, C21_pp, C22_pp, C23_pp),
                 "3m(1)": (E, C3p, C3m, SIGd1, SIGd2, SIGd3),
                 "3m(2)": (E, C3p, C3m, SIGv1, SIGv2, SIGv3),
                 "3dm(1)": (E, C3p, C3m, C21_p, C22_p, C23_p, I, S6p, S6m,
                            SIGd1, SIGd2, SIGd3),
                 "3dm(2)": (E, C3p, C3m, C21_pp, C22_pp, C23_pp, I, S6p, S6m, 
                            SIGv1, SIGv2, SIGv3),
                 "6": (E, C6p, C6m, C3p, C3m, C2),
                 "6d": (E, S3p, S3m, C3p, C3m, SIGh),
                 "6/m": (E, C6p, C6m, C3p, C3m, C2, I,
                         S3p, S3m, S6p, S6m, SIGh),
                 "622": (E, C6p, C6m, C3p, C3m, C2, C21_p, C22_p, C23_p,
                         C21_pp, C22_pp, C23_pp),
                 "6mm": (E, C6p, C6m, C3p, C3m, C2, SIGd1, SIGd2, SIGd3,
                         SIGv1, SIGv2, SIGv3),
                 "6dm2(1)": (E, C3p, C3m, C21_p, C22_p, C23_p, SIGh, S3p, S3m,
                             SIGv1, SIGv2, SIGv3),
                 "6dm2(2)": (E, C3p, C3m, C21_pp, C22_pp, C23_pp, SIGh, S3p, S3m,
                             SIGd1, SIGd2, SIGd3),
                 "6/mmm": (E, C6p, C6m, C3p, C3m, C2, C21_p, C22_p, C23_p,
                           C21_pp, C22_pp, C23_pp, I, S3p, S3m, S6p, S6m,
                           SIGh, SIGd1, SIGd2, SIGd3, SIGv1, SIGv2, SIGv3), 
                 }
        return ops, group
    
    
class SymbolicTensor(object):
    
    def __init__(self, indices, symbol, start=1, voigt=True, reverse=True,
                 create_tensor=True):
        """
        PyMTensor applies crystal symmetries to dielectric, elastic, 
        piezoelectric and other material tensors of interest.  It determines 
        what components are zero and the relationships between the nonzero 
        components for all crystallographic point groups.  It is capable of 
        computing components of higher-order material tensors needed when 
        treating nonlinear effects.
        
        start: integer
            Allow for either 0- or 1-based indexing (`start=0` or `start=1`).
        """
        dims, repeats = SymbolicTensor._parse_name(indices)
        # Total uncompressed tensor dimension
        tdim = sum(dims)
        # Initialize all necessary Voigt and inverse-Voigt mappings based on the
        # unique dimensions
        ivm = {}
        vm = {}
        for dim in set(dims):
            ivm[dim], vm[dim] = SymbolicTensor.voigt_map(dim, start)
        self.dims = dims
        self.vm = vm
        self.ivm = ivm
        self.slices = SymbolicTensor._create_slices(dims)
        self.repeats = repeats
        # Don't create the tensor if just testing initialization methods
        if create_tensor:
            tensor = empty((3,)*tdim, dtype=object)
            # Create slices for applying the Voigt mappings
            # Find all unique tensor component names
            symbols_dict = {}
            for index in product(range(3), repeat=tdim):
                newindex = self._newindex(index, start, voigt)
                name = ''.join([symbol] + [str(val) for val in newindex])
                symbols_dict[name] = Symbol(name)
            names = list(symbols_dict.keys())
            print('Number of unknowns = {}'.format(len(names)))
            sortkey = lambda i: int(i[1:])
            names.sort(reverse=reverse, key=sortkey)
            poly_vals = ring(names, 'QQ<sqrt(3)>')
            poly_dict = dict((name, i) for i, name in enumerate(names))
            R = poly_vals[0]
            ring_vals = poly_vals[1:]
            # Place the ring values in the tensor
            for index in product(range(3), repeat=tdim):
                newindex = self._newindex(index, start, voigt)
                name = ''.join([symbol] + [str(val) for val in newindex])
                tensor[index] = ring_vals[poly_dict[name]]
            self.names = names
            self.R = R
            self.ring_vals = ring_vals
            self.tensor = tensor
        self.tdim = tdim
    
    @staticmethod
    def _create_slices(dims):
        start_slice = 0
        slices = []
        for dim in dims:
            end_slice = start_slice + dim
            slices.append(slice(start_slice, end_slice))
            start_slice = end_slice
        return slices
    
    @staticmethod
    def _parse_name(name):
        if ',' in name:
            indices = name.split(',')
            # Check if the last token is empty
            if indices[-1] == '':
                indices.pop()
        else:
            indices = []
            for val in name:
                if val.isupper():
                    indices.append(val + '2')
                else:
                    indices.append(val + '1')
        # TODO: generalize this to allow repeat characters before the 
        # integer and do some error checking
        dims = [int(val[1:]) for val in indices]
        repeats = SymbolicTensor._repeated_indices(indices)
        return dims, repeats
    
    @staticmethod
    def voigt_map(dim, start=0):
        v = {}
        v[1] = [(1,), (2,), (3,)]
        v[2] = [(1, 1), (2, 2), (3, 3), (2, 3), (1, 3), (1, 2)]
        v[3] = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 2, 3), (1, 1, 3),
                (1, 1, 2), (2, 3, 3), (1, 3, 3), (1, 2, 2), (1, 2, 3)]
        v[4] = [(1, 1, 1, 1), (2, 2, 2, 2), (3, 3, 3, 3), (2, 2, 2, 3), 
                (1, 1, 1, 3), (1, 1, 1, 2), (2, 2, 3, 3), (1, 1, 3, 3), 
                (1, 1, 2, 2), (2, 3, 3, 3), (1, 3, 3, 3), (1, 2, 2, 2),
                (1, 1, 2, 3), (2, 2, 1, 3), (3, 3, 1, 2)]
        vm = {}
        ivm = {}
        ordered_indices = None
        try:
            ordered_indices = v[dim]
        except NotImplementedError:
            print("Voigt map not implemented for dimension {}".format(dim))
        for i, unique_index in enumerate(ordered_indices):
            ishifted = i + start
            # Zero-based indexing: subtract 1
            shifted_index = [val - 1 + start for val in unique_index]
            ivm[ishifted] = shifted_index
            # Every permutation of the index should map to the same Voigt
            # value
            for index in permutations(shifted_index):
                vm[index] = ishifted
        return ivm, vm
    
    def voigt_map_ascending(self, dim, start=0):
        v = {}
        
    
    def _newindex(self, index, start=1, voigt=True):
        """
        Create a new index possibly in Voigt notation with all symmetries and
        sortings possible.
        
        Parameters
        ----------
        index: tuple of integers
            The full tensor index to be processed
        repeats: list of lists of integers
            The indices that need to be sorted assuming all symmetric indices 
            are already reduced to Voigt notation
        start: int
            The starting integer used to choose 0- or 1- based indexing.
        
        Notes
        -----
        This function is separated from the tensor creation to simplify testing.
        """
        ivm = self.ivm
        vm = self.vm
        repeats = self.repeats
        slices = self.slices
        dims = self.dims
        # We compress to Voigt notation and then expand after sorting if the
        # user doesn't want Voigt notation
        # Example:
        # index = [0, 1, 0, 2, 1, 1, 1, 0]
        # repeats = [[0, 3]]
        # symmetric=[0, 1, 3]
        index = [val + start for val in index]
        newindex = []
        for dim, slice_ in zip(dims, slices):
            newindex.append(vm[dim][tuple(index[slice_])])
        sortkey = None
        # We can use a custom sort key if using full-tensor notation 
        if not voigt:
            sortdict = {1:1, 2:4, 3:6, 4:5, 5:3, 6:2}
            sortkey = lambda i: sortdict[i]
        # Repeated indices need to be sorted
        for irepeat in repeats:
            vals = [newindex[i] for i in irepeat]
            vals.sort(key=sortkey)
            for i, var in zip(irepeat, vals):
                newindex[i] = var
        if not voigt:
            index = newindex
            newindex = []
            for dim, val in zip(dims, index):
                newindex.extend(ivm[dim][val])
        return newindex
    
    @staticmethod
    def _repeated_indices(name):
        # Look for all repeated uppercase letters
        repeats = defaultdict(list)
        for i, val in enumerate(name):
            repeats[val].append(i)
        repeats = [val for val in repeats.values() if len(val) > 1]
        return repeats
    
    def apply_symmetry(self, symops, timings=True):
        tensor = self.tensor; R = self.R; ring_vals = self.ring_vals
        eqs = set()
        if timings: tic = time.perf_counter()
        for symop in symops:
            if timings: symtic = time.perf_counter()
            # Convert the components of the symmetry operation arrays into
            # members of the polynomial ring.
            poly_symop = array([[R(val) for val in row] for row in symop])
            row = (rot_tensor(poly_symop, tensor) - tensor).flatten()
            for eq in row:
                if eq != 0:
                    eqs.add(eq)
            if timings: symtoc = time.perf_counter()
            if timings: print(f"Apply symmetry = {symtoc - symtic:0.4f} seconds")
        if len(eqs) == 0:
            return {}
        print('len(eqs) = {}'.format(len(eqs)))
#         print('eqs = ')
#         for eq in eqs:
#             print(eq.as_expr())
        if timings: toc = time.perf_counter()
        if timings: print(f"Rotate tensor = {toc - tic:0.4f} seconds")
        
        if timings: tic = time.perf_counter()
        process = psutil.Process()
        print(process.memory_info().rss)  # in bytes 
        sol = solve_lin_sys(eqs, R, _raw=False)
        process = psutil.Process()
        print(process.memory_info().rss)  # in bytes 
        if timings: toc = time.perf_counter()
        if timings: print(f"Solve linear system = {toc - tic:0.4f} seconds")
        numfreevars = len(ring_vals) - len(sol)
        print(f"Number of free variables = {numfreevars}")
        logging.info('Number of free variables = {}'.format(numfreevars))
        return sol
    
    def sol_details(self, symops):
        R = self.R; names = self.names
        sol = self.apply_symmetry(symops)
        tic = time.perf_counter()
        fullsol = {}
        for name in names:
            symbol = Symbol(name)
            fullsol[name] = sol.get(symbol, symbol)
        toc = time.perf_counter()
        print(f"To full solution = {toc - tic:0.4f} seconds")
        return fullsol, R
    
    def unique_tensor(self, symops):
        tensor = self.tensor
        sol = self.apply_symmetry(symops)
        tic = time.perf_counter()
        newtensor = empty_like(tensor)
        it = nditer(tensor, flags=['multi_index', 'refs_ok'])
        while not it.finished:
            oldval = it[0].item()
            # Try to retrieve the new value from the solution dictionary.  If
            # the variable doesn't exist then it is unique and we just use the
            # original symbol via the dictionary `get` method
            newval = sol.get(oldval, oldval)
            newtensor[it.multi_index] = newval
            it.iternext()
        toc = time.perf_counter()
        print(f"Format solution = {toc - tic:0.4f} seconds")
        return newtensor
                

def symbolic_piezo(dim, sym, canonical=True, sort=True):
    voigt_map = {(0, 0): 0, (1, 1): 1, (2, 2): 2,
                 (1, 2): 3, (2, 1): 3, (0, 2): 4, (2, 0): 4, 
                 (0, 1): 5, (1, 0): 5}
    tensor = empty((3,)*dim, dtype=object)
    indices = product(range(3), repeat=dim)
    if canonical:
        ls, nc, rs = [''] * 3
        shift = 1
    else:
        ls, nc, rs = ['[', ',', ']']
        shift = 0
    if dim == 2:
        for index in indices:
            i, j = index
            voigt_order = (i, j)
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 3:
        for index in indices:
            # The following statement handles the third-order dielectric constant
            if sort:
                newindex = tuple(sorted(index))
                i = newindex[0]
                J = voigt_map[newindex[1:3]]
            else:
                i = index[0]
                J = voigt_map[index[1:3]]
            voigt_order = (i, J)
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 4:
        for index in indices:
            I = voigt_map[index[0:2]]
            J = voigt_map[index[2:4]]
            if sort:
                voigt_order = sorted([I, J])
            else:
                voigt_order = [I, J]
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 5:
        for index in indices:
            i = index[0]
            J, K = [voigt_map[val] for val in [index[1:3], index[3:5]]]
            voigt_order = (i, J, K)
            if sort:
                voigt_order = [i] + sorted([J, K])
            else:
                voigt_order = [i, J, K]
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    if dim == 6:
        for index in indices:
            I, J, K = [voigt_map[val] for val
                       in [index[2*i:2*i+2] for i in range(3)]]
            if sort:
                voigt_order = sorted([I, J, K])
            else:
                voigt_order = [I, J, K]
            sindex = nc.join([str(val+shift) for val in voigt_order])
            symbol = ''.join([sym, ls, sindex, rs])
            tensor[index] = Symbol(symbol)
    raise NotImplementedError("Dimension {} hasn't been programmed yet.".format(dim))
    return tensor


if __name__ == '__main__':
    set_printoptions(linewidth=200)
    sg = SgSymOps()
#     sg.groups['hexagonal']['3']
#     sg.groups['hexagonal']['3d']
    symops = sg('3m')
#     symops = sg('1')
#     symops = sg('32')
#     symops = sg('6')
#     symops = sg('m3d')
#     st = SymbolicTensor("aa", 'd')
    st = SymbolicTensor("abcdef", 'c')
#     st = SymbolicTensor('aB', 'e')
#     st = SymbolicTensor('AAA', 'c')
#     st = SymbolicTensor('ab', 'e')
#     C3p = sg.hexops["SIGd1"]
#     print('st=', to_voigt(st.tensor))
#     ans = to_voigt(rot_tensor(C3p, st.tensor))
#     for i in range(3):
#         for j in range(3):
#             print(i+1, j+1, ans[i, j].expand())
#     print(ans[0, 0].expand())
#     reduced_tensor = st.unique_tensor(symops)
    st.sol_details(symops)
