"""
Rational LITA schemes for odd square matrix multiplication.

R(N) = N^3/3 + 7*N^2/2 + 14*N/3 - 9/2.

API:
    rank = lita_odd_rank(N)
    scheme = lita_odd(N)
    scheme.save("scheme.npz")

N must be odd with 9 <= N < 32.
"""

import argparse
from array import array
from itertools import combinations, permutations
import json
from math import comb, gcd
from pathlib import Path

import numpy as np


# ---------- Integer linear forms ----------


def _check_dimension(N):
    if type(N) is not int or N < 9 or N >= 32 or N % 2 != 1:
        raise ValueError("N must be an odd integer with 9 <= N < 32")


def lita_odd_rank(N):
    _check_dimension(N)
    m = (N-3)//2
    return 16*comb(m,3)+68*comb(m,2)+98*m+50


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


def _sum(forms):
    result = {}
    for form in forms:
        _add_scaled(result, form, 1)
    return _Form(result)


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


# ---------- Signed blocks and column parameters ----------


class _Coordinates:
    """Sparse row completion: M=[[X,X*s0],[0,0]], Z=M*H^{-1}.

    The extra lower row is zero. A single added column gives Q*s=0.
    All ordinary entries are signed physical entries; no dense projection
    is required. Defects are computed from the actual block column sums.
    """

    def __init__(self, N):
        _check_dimension(N)
        self.physical_N = N
        self.K = K = (N+1)//2
        self.D = K-1
        self.w = 2-self.D
        self.grid = 24*(self.D-2)*(self.D-3)
        self._entries = {}
        self.row_dot = tuple(_Form({i*N+j:self.grid*(1 if j<K else -1)
                                        for j in range(N)}) for i in range(N))
        self.defects = {
            b: tuple(_sum(self.weight(k)*(self.entry(left,k,i)
                                               +self.entry(right,k,i))
                               for k in range(K)) for i in range(K))
            for b,left,right in ((0,0,2),(3,1,3))
        }
        self.q = _sum(self.entry(b,self.D,self.D) for b in range(4))/4

    def weight(self, i):
        return self.w if i == self.D else 1

    def entry(self, block, i, j):
        key = block,i,j
        if key not in self._entries:
            br,bc = divmod(block,2)
            row,col = br*self.K+i,bc*self.K+j
            if row == self.physical_N:
                out = _Form()
            elif col == self.physical_N:
                out = self.row_dot[row]/(-self.w)
            else:
                value = self.grid*((-1) if bc else 1)//self.weight(j)
                out = _Form({row*self.physical_N+col:value})
            self._entries[key] = out
        return self._entries[key]


class _View:
    """One column source with the pooled closing conditions."""

    def __init__(self, coordinates, axis):
        self.coordinates = c = coordinates
        self.D = D = c.D
        X = c.entry
        self.gamma = g = (X(0,0,0)+X(1,0,1)+X(2,1,0)+X(3,1,1))/3
        self.a = [g-X(0, i, i) for i in range(D)]
        self.b = [X(3, i, i)-g for i in range(D)]
        total = -_sum(self.a+self.b)/c.w
        if axis == 0:
            last = g-X(0,D,D)
        elif axis == 1:
            last = total+X(1,D,D)+g
        else:
            last = total-X(3,D,D)+g
        self.a.append(last)
        self.b.append(total-last)
        self._entries = {}
        self.row_sums = {
            b: tuple(_sum(c.weight(k)*self.entry(b, i, k)
                              for k in range(D+1)) for i in range(D+1))
            for b in (0, 3)
        }
        self.column_sums = {
            b: tuple(_sum(c.weight(k)*self.entry(b, k, i)
                              for k in range(D+1)) for i in range(D+1))
            for b in (0, 3)
        }

    def entry(self, block, i, j):
        key = block, i, j
        if key not in self._entries:
            delta = self.a[j] if block in (0, 2) else self.b[j]
            self._entries[key] = self.coordinates.entry(block, i, j)
            self._entries[key] += (1 if block < 2 else -1)*delta
        return self._entries[key]

    def half_sum(self, block, i, j):
        return (self.row_sums[block][j]+self.column_sums[block][i])/2


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


def _triangle_terms(c, views):
    for vertices in combinations(range(c.D+1), 3):
        for i, j, k in permutations(vertices):
            for barred in (0, 1):
                if i < j < k or k < j < i:
                    row = tuple(_cycle(v, i, j, k, barred) for v in views)
                    row = row[0], row[1], (-1 if barred else 1)*row[2]
                    yield _weighted(row, c, (i, j, k))
                yield _weighted(_mixed_factors(views, i, j, k, barred),
                                     c, (i, j, k))


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


def _vertex_terms(c, views):
    D, w = c.D, c.w
    for i in range(D):
        recipes = []
        for v in views:
            Y, g = v.entry, v.gamma
            u, ell = Y(1, i, i)+g, Y(2, i, i)+g
            k0 = (v.half_sum(0, i, i)+w*(Y(1, i, D)+Y(2, D, i))
                  +(D-4)*g-c.defects[0][i])
            k3 = (v.half_sum(3, i, i)+w*(Y(1, D, i)+Y(2, i, D))
                  +(D-4)*g-c.defects[3][i])
            recipes.append(((2*u, ell, k0), (-2*ell, u, k3)))
        for r in range(2):
            yield from _cyclic(tuple(recipe[r] for recipe in recipes))


def _column_correction_terms(c, views):
    """Two cyclic prototypes per ordered off-diagonal pair.

    The actual column defects are unchanged by the source.
    Row closure allows subtracting the diagonal in the third factors;
    hence no diagonal correction products remain, including at the center.
    """
    for i in range(c.D+1):
        for j in range(c.D+1):
            if i == j:
                continue
            recipes = []
            for v in views:
                Y = v.entry
                recipes.append((
                    (c.defects[0][i], Y(0, i, j)+Y(1, i, j),
                     Y(2, i, j)-Y(2, i, i)),
                    (-c.defects[3][i], Y(2, i, j)+Y(3, i, j),
                     Y(1, i, j)-Y(1, i, i)),
                ))
            for r in range(2):
                for u, v, z in _cyclic(tuple(recipe[r] for recipe in recipes)):
                    yield c.weight(i)*u, c.weight(j)*v, z


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
        g,q = view.gamma,coordinates.q
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
        self.statistics = {}

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
            "construction": "LITA odd column-defect aggregation",
        }, separators=(",", ":")))
        np.savez_compressed(path, **arrays)


def iter_terms(N, statistics=None):
    c = _Coordinates(N)
    views = tuple(_View(c,axis) for axis in range(3))
    D = c.D
    families = (
        ("triangles", _triangle_terms, 16*comb(D+1, 3)),
        ("edges", _edge_terms, 24*comb(D, 2)-6),
        ("vertices", _vertex_terms, 6*D),
        ("boundary", _boundary_terms, 28*D),
        ("center", _center_terms, 10),
        ("column_correction", _column_correction_terms, 6*D*(D+1)),
    )
    counts = {}
    for name, family, expected in families:
        count = 0
        for term in family(c, views):
            if not all(term):
                continue
            count += 1
            yield tuple((f, c.grid) for f in term)
        if count != expected:
            raise ArithmeticError(f"{name}: emitted {count}, expected {expected}")
        counts[name] = count
    if sum(counts.values()) != lita_odd_rank(N):
        raise ArithmeticError("direct odd-LITA rank mismatch")
    if statistics is not None:
        statistics.update(N=N, D=D, m=D-1, rank=sum(counts.values()),
                          family_counts=counts, numerator_grid=c.grid,
                          construction="sparse direct odd LITA with column defects")


def lita_odd(N):
    scheme = Scheme(N)
    for term in iter_terms(N, scheme.statistics):
        scheme.append(*term)
    return scheme


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("N", type=int)
    parser.add_argument("output", nargs="?")
    parser.add_argument("--rank-only", action="store_true")
    args = parser.parse_args()
    if args.rank_only:
        print(f"N={args.N} rank={lita_odd_rank(args.N)}")
        return
    scheme = lita_odd(args.N)
    if args.output:
        scheme.save(args.output)
    print(scheme.statistics)


if __name__ == "__main__":
    main()
