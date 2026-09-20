"""
Rational LITA schemes for even square matrix multiplication.

R(N) = N^3/3 + 3*N^2 + 37*N/6 + 5.

API:
    rank = lita_rank(N)
    scheme = lita(N)
    scheme.save("scheme.npz")

N must be even and at least 8.
"""

from array import array
from itertools import combinations, permutations
import json
from math import gcd
from pathlib import Path
import sys

import numpy as np


# ---------- Integer linear forms ----------


def _check_dimension(N):
    if type(N) is not int or N < 8 or N % 2:
        raise ValueError("N must be an even integer at least 8")


def lita_rank(N):
    _check_dimension(N)
    D = N // 2
    return 16*(D+1)*D*(D-1)//6 + 12*D*(D-1) + 27*D + 5


def _add_scaled(row, other, scale):
    if not scale:
        return row
    for index, value in other.items():
        value = row.get(index, 0) + scale * value
        if value:
            row[index] = value
        elif index in row:
            del row[index]
    return row


def _linear(forms, coefficients):
    result = {}
    for form, coefficient in zip(forms, coefficients):
        _add_scaled(result, form, coefficient)
    return result


def _sum(forms):
    result = {}
    for form in forms:
        _add_scaled(result, form, 1)
    return _Form(result)


def _scaled_form(form, scale):
    return form if scale == 1 else _linear((form,), (scale,))


class _Form(dict):
    """Integer linear forms on a fixed rational coefficient grid."""

    def __add__(self, other):
        return _add_scaled(_Form(self), other, 1)

    def __sub__(self, other):
        return _add_scaled(_Form(self), other, -1)

    def __neg__(self):
        return self * -1

    def __mul__(self, scalar):
        return _Form((i, c * scalar) for i, c in self.items() if c * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar):
        result = _Form()
        for i, value in self.items():
            quotient, remainder = divmod(value, scalar)
            if remainder:
                raise ArithmeticError("nonintegral coefficient on the LITA grid")
            if quotient:
                result[i] = quotient
        return result


# ---------- Signed blocks and row-column parameters ----------


class _Coordinates:
    """Forms in original row-major coordinates, stored as grid*F.

    The grid covers the border denominator 4*(D-2)^2, the common form,
    half-sums, and the boundary and polarization factors. Divisions are exact.
    """

    def __init__(self, N):
        self.N, self.D = N, N // 2
        self.w = 2-self.D
        self.grid = 96*(self.D-2)**2*(self.D-3)
        self._entries = {}

    def weight(self, i):
        return self.w if i == self.D else 1

    def entry(self, block, i, j):
        key = block, i, j
        if key not in self._entries:
            D, N = self.D, self.N
            br, bc = divmod(block, 2)
            if i < D and j < D:
                value = self.grid * (-1 if bc else 1)
                result = _Form({(br*D+i)*N+bc*D+j: value})
            elif i < D:
                result = _sum(self.entry(2*br+b, i, k)
                              for b in range(2) for k in range(D))/(2*(D-2))
            elif j < D:
                result = _sum(self.entry(2*a+bc, k, j)
                              for a in range(2) for k in range(D))/(2*(D-2))
            else:
                result = _sum(self.entry(b, r, s) for b in range(4)
                              for r in range(D) for s in range(D))/(4*(D-2)**2)
            self._entries[key] = result
        return self._entries[key]


class _View:
    """One input's anchor parameters and its closing condition."""

    def __init__(self, coordinates, axis):
        self.coordinates = coordinates
        self.D = D = coordinates.D
        X = coordinates.entry
        self.gamma = (X(0,2,2)+X(1,2,3)+X(2,3,2)+X(3,3,3))/3
        self.r0 = []
        for i in range(D):
            if axis == 0:
                r = (X(0,0,1)+X(2,0,1) if i == 0 else
                     X(0,i,i)-X(3,i,i)+X(2,1,i)+X(3,1,i))
            elif axis == 1 and i == 1:
                r = X(0,1,1)-X(3,1,1)+X(2,0,1)+X(3,0,1)
            else:
                r = X(0,i,0)+X(2,i,0)
            self.r0.append(r)
        self.r0.append(_sum(self.r0)/(D-2))
        self.c0 = [self.r0[i]-X(0,i,i)+self.gamma for i in range(D)]
        self.c1 = [X(3,i,i)-self.gamma for i in range(D)]
        closing_sum = _sum(self.c0+self.c1)/(D-2)
        if axis == 0:
            closing_c0 = self.r0[D]-X(0,D,D)+self.gamma
        elif axis == 1:
            closing_c0 = closing_sum+X(0,D,D)+self.r0[D]+self.gamma
        else:
            closing_c0 = closing_sum-X(0,D,D)+self.gamma
        self.c0.append(closing_c0)
        self.c1.append(closing_sum-closing_c0)
        self._entries = {}
        # Half-sums include the row-column parameters.
        self.row_sums = {b: tuple(_sum(coordinates.weight(k)*self.entry(b,j,k)
                                     for k in range(D+1)) for j in range(D+1))
                         for b in (0,3)}
        self.column_sums = {b: tuple(_sum(coordinates.weight(k)*self.entry(b,k,i)
                                        for k in range(D+1)) for i in range(D+1))
                            for b in (0,3)}

    def entry(self, block, i, j):
        key = block, i, j
        if key not in self._entries:
            X = self.coordinates.entry(block,i,j)
            if block == 0:
                result = X-self.r0[i]+self.c0[j]
            elif block == 1:
                result = X+self.r0[i]+self.c1[j]
            elif block == 2:
                result = X-self.c0[j]
            else:
                result = X-self.c1[j]
            self._entries[key] = result
        return self._entries[key]

    def half_sum(self, block, i, j):
        return (self.row_sums[block][j]+self.column_sums[block][i])/2


# ---------- One weighted aggregation identity ----------


def _cycle(view, i, j, k, barred):
    b = 3*barred
    return _sum(view.entry(b,u,v) for u,v in ((i,j),(j,k),(k,i)))


def _mixed(view, i, j, k, barred):
    X = view.entry
    return X(2-barred,j,k)+X(1+barred,k,i)-X(3*barred,i,j)


def _mixed_factors(views, i, j, k, barred):
    A,B,C = views
    return (_mixed(A,i,j,k,barred), _mixed(B,k,i,j,barred),
            (-1 if barred else 1)*_mixed(C,j,k,i,barred))


def _weighted(term, coordinates, ids):
    return tuple(f*coordinates.weight(i) for f,i in zip(term,ids))


def _cyclic(recipes):
    """Rotate recipes first, evaluating each in its destination input."""
    for shift in range(3):
        yield tuple(recipes[a][(a+shift)%3] for a in range(3))


def _triangle_terms(coordinates, views):
    for vertices in combinations(range(coordinates.D+1),3):
        for i,j,k in permutations(vertices):
            for barred in (0,1):
                if i < j < k or k < j < i:
                    term = [_cycle(v,i,j,k,barred) for v in views]
                    term[2] = (-1 if barred else 1)*term[2]
                    yield _weighted(term,coordinates,(i,j,k))
                if barred == 1 and i == 1 and k == 0 and 2 <= j < coordinates.D:
                    continue
                term = _mixed_factors(views,i,j,k,barred)
                if barred == 0 and i == 0 and j == 1 and 2 <= k < coordinates.D:
                    other = _mixed_factors(views,1,k,0,1)
                    if term[:2] != other[:2]:
                        raise ArithmeticError("anchor triangle alignment failed")
                    term = term[0],term[1],term[2]+other[2]
                yield _weighted(term,coordinates,(i,j,k))


def _seeds(view, i, j):
    x0,x1,x2,x3 = (view.entry(b,i,j) for b in range(4))
    return (x3-x0, -x2-x3, -x1-x3,
            x0+view.half_sum(3,i,j), -x3-view.half_sum(0,i,j))


def _heptad(seeds):
    yield tuple(h[0] for h in seeds)
    yield from _cyclic(tuple((h[1],h[2],h[3]) for h in seeds))
    yield from _cyclic(tuple((h[4],-h[0]-h[2],-h[0]-h[1]) for h in seeds))


# ---------- Ordinary-edge prototypes and pooled vertex terms ----------


def _edge_recipe(view, i, j):
    Y,g = view.entry,view.gamma
    a,b,c,d = (Y(k,i,j) for k in range(4))
    ap,bp,cp,dp = (Y(k,j,i) for k in range(4))
    ui,uj = Y(1,i,i)+g,Y(1,j,j)+g
    li,lj = Y(2,i,i)+g,Y(2,j,j)+g
    pi = b+cp-g
    return ((-c-d,b+d,2*(dp+g-view.half_sum(3,i,j))),
            (c+d+li,pi,bp+dp+a+c+lj+ui),
            (a+c,2*(ap+g-view.half_sum(0,i,j)),a+b),
            (-a-c-lj,ap+bp+c+d+li+uj,pi))


def _edge_terms(coordinates, views):
    for i,j in combinations(range(coordinates.D),2):
        for left,right in ((i,j),(j,i)):
            recipes = tuple(_edge_recipe(v,left,right) for v in views)
            for r in range(4):
                yield from _cyclic(tuple(recipe[r] for recipe in recipes))


def _vertex_terms(coordinates, views):
    D,w = coordinates.D,coordinates.w
    for i in range(D):
        recipes = []
        for v in views:
            Y,g = v.entry,v.gamma
            u,l = Y(1,i,i)+g,Y(2,i,i)+g
            k0 = v.half_sum(0,i,i)+w*(Y(1,i,D)+Y(2,D,i))+(D-4)*g
            k3 = v.half_sum(3,i,i)+w*(Y(1,D,i)+Y(2,i,D))+(D-4)*g
            recipes.append(((2*u,l,k0),(-2*l,u,k3)))
        for r in range(2):
            yield from _cyclic(tuple(recipe[r] for recipe in recipes))


# ---------- Boundary edges and the closing vertex ----------


def _boundary_terms(coordinates, views):
    D,w = coordinates.D,coordinates.w
    for i in range(D):
        for barred in (0,1):
            b = 3*barred
            term = tuple(v.entry(b,i,D)+v.entry(b,D,i)
                         +((D-2)*v.entry(b,D,D)-v.gamma)/(D-3) for v in views)
            yield (D-2)*(D-3)*term[0],term[1],(-1 if barred else 1)*term[2]
        for ids in ((i,i,D),(i,D,i),(i,D,D),(D,i,i),(D,i,D),(D,D,i)):
            for barred in (0,1):
                yield _weighted(_mixed_factors(views,*ids,barred),coordinates,ids)
        for left,right in ((i,D),(D,i)):
            for u,v,z in _heptad(tuple(_seeds(view,left,right) for view in views)):
                yield coordinates.weight(left)*u,coordinates.weight(right)*v,2*z


def _center_terms(coordinates, views):
    D = coordinates.D
    t = D-2
    recipes,polar = [],[]
    for view in views:
        a,b,c,d = (view.entry(k,D,D) for k in range(4))
        g,q = view.gamma,coordinates.entry(0,D,D)
        h = d-a
        eta = 3*g-4*(5*D-12)*q-2*t*(b-c)
        L = (2*g-a-d)/(2*(D-3))-4*q
        U = a+view.half_sum(3,D,D)+t*(3*g-10*q+2*c+2*d-eta)
        V = -d-view.half_sum(0,D,D)-t*(3*g-10*q+2*a+2*c-eta)
        recipes.append(((c+d+h/(D-3),b+d,U),
                        (U+2*t*(a+b),c+d,b+d),
                        (b+d,U,c+d),
                        (V,a+b,a+c-h/(D-3)),
                        (a+c,V,a+b),
                        (a+b,a+c,V-2*t*(b+d))))
        polar.append((eta,h,L))
    for r in range(6):
        u,v,z = (recipes[axis][r][axis] for axis in range(3))
        yield 2*t*t*u,v,z
    for epsilon in (-1,1):
        for delta in (-1,1):
            u,v,z = (eta+epsilon*h+delta*L for eta,h,L in polar)
            yield (-t**3*epsilon*delta*u)/4,v,z


# ---------- Sparse serialization ----------


class SparseAxis:
    def __init__(self):
        self.indptr = array("q", (0,))
        self.indices = array("i")
        self.numerators = array("q")
        self.denominators = array("q")

    @property
    def rows(self):
        return len(self.indptr) - 1

    def append(self, row, denominator, transpose=0):
        if denominator <= 0:
            raise ValueError("the common denominator must be positive")
        if transpose:
            N = transpose
            items = (((index % N) * N + index // N, value) for index, value in row.items())
        else:
            items = row.items()
        start = len(self.indices)
        try:
            for index, numerator in sorted(items):
                if not numerator:
                    continue
                common = gcd(abs(numerator), denominator)
                self.indices.append(index)
                self.numerators.append(numerator // common)
                self.denominators.append(denominator // common)
        except OverflowError:
            del self.indices[start:]
            del self.numerators[start:]
            del self.denominators[start:]
            raise OverflowError("LITA CSR indices or reduced coefficients exceed their integer storage") from None
        self.indptr.append(len(self.indices))

    def arrays(self, name):
        return {
            f"{name}_indptr": np.frombuffer(self.indptr, dtype=np.int64),
            f"{name}_indices": np.frombuffer(self.indices, dtype=np.int32),
            f"{name}_numerators": np.frombuffer(self.numerators, dtype=np.int64),
            f"{name}_denominators": np.frombuffer(self.denominators, dtype=np.int64),
        }


class Scheme:
    def __init__(self, N):
        _check_dimension(N)
        self.N = N
        self.U = SparseAxis()
        self.V = SparseAxis()
        self.W = SparseAxis()

    @property
    def rank(self):
        return self.U.rows

    def append(self, u, v, w):
        self.U.append(*u)
        self.V.append(*v)
        self.W.append(*w, transpose=self.N)

    def save(self, path):
        if not self.U.rows == self.V.rows == self.W.rows:
            raise ValueError("the three factor axes must have equal row counts")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {}
        arrays.update(self.U.arrays("u"))
        arrays.update(self.V.arrays("v"))
        arrays.update(self.W.arrays("w"))
        arrays["metadata_json"] = np.asarray(json.dumps({
            "coefficient_field": "Q",
            "format": "fmmp.qcsr.v1",
            "rank": self.rank,
            "tensor": [self.N, self.N, self.N],
            "is_complete_matrix_multiplication_scheme": True,
            "axis_convention": "U,V read row-major inputs; W writes row-major C=AB.",
            "construction": "LITA row-column aggregation",
        }, separators=(",", ":")))
        np.savez_compressed(path, **arrays)


def lita(N):
    """Construct the explicit rational weighted LITA scheme over Q."""
    _check_dimension(N)
    coordinates = _Coordinates(N)
    views = tuple(_View(coordinates,a) for a in range(3))
    scheme = Scheme(N)
    for family in (_triangle_terms,_edge_terms,_vertex_terms,_boundary_terms,_center_terms):
        for term in family(coordinates,views):
            if not all(term):
                continue
            scheme.append(*((f,coordinates.grid) for f in term))
    if scheme.rank != lita_rank(N):
        raise RuntimeError("internal LITA rank mismatch")
    return scheme


if __name__ == "__main__":
    if len(sys.argv) not in (2,3):
        raise SystemExit("usage: python lita.py N [output.npz]")
    dimension = int(sys.argv[1])
    result = lita(dimension)
    if len(sys.argv) == 3:
        result.save(sys.argv[2])
    print(f"N={dimension} rank={result.rank}")
    if len(sys.argv) == 3:
        print(Path(sys.argv[2]))
