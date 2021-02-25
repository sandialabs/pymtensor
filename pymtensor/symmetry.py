from numpy import (abs, array, eye, rint)
# from scipy.linalg import lu, svd
from sympy import acos, cos, pi, sin, sqrt, Rational


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
    
    
class RedSgSymOps(object):
    
    def __init__(self):
        symops = {}
        # Generator matrices
        symops['1'] = eye(3, dtype=int)
        symops['1d'] = -symops['1']
        symops['2parZ1'] = array([[ 1,  0,  0],
                                  [ 0, -1,  0],
                                  [ 0,  0, -1]])
        symops['2parZ2'] = array([[-1,  0,  0],
                                  [ 0,  1,  0],
                                  [ 0,  0, -1]])
        symops['2parZ3'] = array([[-1,  0,  0],
                                  [ 0, -1,  0],
                                  [ 0,  0,  1]])
        symops['mperZ1'] = array([[-1,  0,  0],
                                  [ 0,  1,  0],
                                  [ 0,  0,  1]])
        symops['mperZ2'] = array([[ 1,  0,  0],
                                  [ 0, -1,  0],
                                  [ 0,  0,  1]])
        symops['mperZ3'] = array([[ 1,  0,  0],
                                  [ 0,  1,  0],
                                  [ 0,  0, -1]])
        symops['mper[11d0]'] = array([[ 0,  1,  0],
                                     [ 1,  0,  0],
                                     [ 0,  0, -1]])
        rt3 = sqrt(3)
        onehalf = Rational(1, 2)
        rt3 *=  onehalf
        symops['3parZ3'] = array([[  -onehalf, rt3,   0],
                                  [-rt3,  -onehalf,   0],
                                  [   0,   0,   1]])
        symops['3dparZ3'] = -symops['3parZ3']
        symops['3par[111]'] = array([[ 0,  1,  0],
                                     [ 0,  0,  1],
                                     [ 1,  0,  0]])
        symops['3dpar[111]'] = -symops['3par[111]']
        symops['4parZ3'] = array([[ 0,  1,  0],
                                  [-1,  0,  0],
                                  [ 0,  0,  1]])
        symops['4dparZ3'] = -symops['4parZ3']
        symops['6parZ3'] = array([[   onehalf, rt3,   0],
                                  [-rt3,   onehalf,   0],
                                  [   0,   0,   1]])
        symops['6dparZ3'] = -symops['6parZ3']
        group = {}
        def symtuple(names):
            return tuple(symops[name] for name in names)
        group["1"] = symtuple(['1'])
        group["1d"] = symtuple(['1d'])
        group["2"] = symtuple(['2parZ2'])
        group["m"] = symtuple(['mperZ2'])
        group["2/m"] = symtuple(['2parZ2', 'mperZ2'])
        group["222"] = symtuple(['2parZ1', '2parZ2'])
        group["mm2"] = symtuple(['mperZ1', 'mperZ2'])
        group["mmm"] = symtuple(['mperZ1', 'mperZ2', 'mperZ3'])
        group["4"] = symtuple(['4parZ3'])
        group["4d"] = symtuple(['4dparZ3'])
        group["4/m"] = symtuple(['4parZ3', 'mperZ3'])
        group["422"] = symtuple(['4parZ3', '2parZ1'])
        group["4mm"] = symtuple(['4parZ3', 'mperZ1'])
        group["4d2m"] = symtuple(['4dparZ3', '2parZ1'])
        group["4/mmm"] = symtuple(['4parZ3', 'mperZ3', 'mperZ1'])
        group["3"] = symtuple(['3parZ3'])
        group["3d"] = symtuple(['3dparZ3'])
        group["32"] = symtuple(['3parZ3', '2parZ1'])
        group["3m"] = symtuple(['3parZ3', 'mperZ1'])
        group["3dm"] = symtuple(['3dparZ3', 'mperZ1'])
        group["6"] = symtuple(['6parZ3'])
        group["6d"] = symtuple(['6dparZ3'])
        group["6/m"] = symtuple(['6parZ3', 'mperZ3'])
        group["622"] = symtuple(['6parZ3', '2parZ1'])
        group["6mm"] = symtuple(['6parZ3', 'mperZ1'])
        group["6dm2"] = symtuple(['6dparZ3', 'mperZ1'])
        group["6/mmm"] = symtuple(['6parZ3', 'mperZ3', 'mperZ1'])
        group["23"] = symtuple(['2parZ3', '3par[111]'])
        group["m3d"] = symtuple(['mperZ1', '3dpar[111]'])
        group["432"] = symtuple(['4parZ3', '3par[111]'])
        group["4d3m"] = symtuple(['4dparZ3', '3par[111]'])
        group["m3dm"] = symtuple(['4parZ3', '3dpar[111]', 'mper[11d0]'])
        self.group = group
    
    def __call__(self, cname):
        """
        Parameters
        ----------
        cname: string
            Class name
        """
        return self.group[cname]
