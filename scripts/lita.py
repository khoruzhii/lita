"""
Rational LITA schemes for even square matrix multiplication.

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
    return 16*(D+1)*D*(D-1)//6 + 12*D*(D-1) + 28*D + 8


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


def _basis(size):
    return tuple(_Form({i: 1}) for i in range(size))


# ---------- Weighted closed blocks ----------


class _Coordinates:
    """Ordinary vertices have weight 1, the closing vertex has weight 2-D.

    The total weight is 2. Forms store K*F, K=6*(D-2)*(D-3). Every division is exact.
    Projection and fraction reduction happen once, at serialization.
    """

    def __init__(self, N):
        self.N, self.D = N, N // 2
        self.w = 2-self.D
        self.grid = 6*(self.D-2)*(self.D-3)
        self.traces = tuple(_sum(self.entry(b, i, i) for i in range(self.D))
                            for b in range(4))

    def shared(self, block):
        """Original block sum and trace, expressed in lifted coordinates."""
        closing = self.w*self.entry(block, self.D, self.D)
        return 2*closing, self.traces[block]+closing

    def entry(self, block, i, j):
        br, bc = divmod(block, 2)
        d = self.D+1
        return _Form({(br*d+i)*(self.N+2)+bc*d+j: self.grid})

    def weight(self, i):
        return self.w if i == self.D else 1

    def edge(self, i, j):
        diagonal = tuple(self.entry(b, v, v) for v in (i, j) for b in range(4))
        cross = tuple(self.entry(b, u, v) for b in range(4)
                      for u, v in ((i, j), (j, i)))
        return diagonal + cross


_BLOCK_SIGNS = ((1, 1, 1, 1), (-1, 1, 1, -1), (-1, 1, -1, 1))


class _BlockView:
    """Signed blocks with a weighted-zero-sum diagonal field."""

    def __init__(self, coordinates, axis):
        self.coordinates = coordinates
        self.signs = _BLOCK_SIGNS[axis]
        self.D, self.axis = coordinates.D, axis
        sums, traces = zip(*(coordinates.shared(b) for b in range(4)))
        x, y, z = self.signs[0]*traces[0], self.signs[3]*traces[3], self.signs[3]*sums[3]
        fixed, moving = (y, x) if axis == 2 else (x, y)
        self.common = (2*fixed+(self.D-4)*moving-z)/(2*(self.D-2))
        self.step = moving/(self.D-2)
        self.heptad = (2*fixed-z)/(2*(self.D-2))
        self.difference = x-y

    def face(self, i, j, k):
        closing = (i == self.D)+(j == self.D)+(k == self.D)
        return self.common+closing*self.step

    def entry(self, block, i, j):
        return self.signs[block]*self.coordinates.entry(block, i, j)

    def diagonal(self, block, i):
        if i == self.D:
            C = self.coordinates
            return -self.signs[block]*_Form(C.traces[block])/C.w
        return self.entry(block, i, i)


# ---------- One weighted aggregation identity ----------


def _cycle(view, i, j, k, barred):
    b = 3*barred
    return view.signs[b]*(_sum(view.entry(b, u, v) for u, v in ((i,j),(j,k),(k,i)))
                          - _sum(view.diagonal(b, v) for v in (i,j,k)) + view.face(i,j,k))


def _mixed(view, i, j, k, barred):
    Y, z = view.entry, view.diagonal
    u, v = (j, i) if barred else (i, j)
    b = 3*barred
    return (Y(2-barred, j, k) + Y(1+barred, k, i) - Y(b, i, j)
            + z(0, u) + z(2, u) + z(3, u) - z(2, v) + z(b, k) - view.face(i,j,k))


def _mixed_factors(views, i, j, k, barred):
    A, B, C = views
    return (_mixed(A, i, j, k, barred), _mixed(B, k, i, j, barred),
            (-1 if barred else 1)*_mixed(C, j, k, i, barred))


def _weighted(term, coordinates, ids):
    return tuple(_scaled_form(f, coordinates.weight(i)) for f, i in zip(term, ids))


def _triangle_terms(coordinates):
    views = tuple(_BlockView(coordinates, a) for a in range(3))
    for vertices in combinations(range(coordinates.D+1), 3):
        for i, j, k in permutations(vertices):
            for barred in (0, 1):
                if i < j < k or k < j < i:
                    term = tuple(_cycle(v, i, j, k, barred) for v in views)
                    yield _weighted(term, coordinates, (i,j,k))
                yield _weighted(_mixed_factors(views, i, j, k, barred), coordinates, (i,j,k))


# ---------- Six fixed edge pairs and their reflections ----------


def _edge_half(x):
    """Six fixed pairs; h and t carry the two shared additions."""
    h, t = x[16:]
    a, b, c, e = x[:4]
    A, B, C, E = x[4:8]
    p, P, q, Q, r, R, s, S = x[8:16]

    u = a+c-p-r
    v = c-e-A+B+2*C-2*E-P+Q-r+s
    yield u, v, -a+C+p-R+h
    yield u, v+2*P+t, -C+E-p+q

    u = c+e-R-S
    w = a+c-B-E+P-q+R-s
    yield u, -e+A-2*C-Q+r+2*S+h, w
    yield u, A-C-Q+S, -2*(w+s)-t

    v = -C+E+r-s
    u = 2*a+b+2*c+e+A+C-p+Q-r+S
    yield u, v, -a+E+q-R+h
    yield -2*S-t, v, -a-c+q+s

    v = a-c-p+r
    yield C+E+p+q, v, -2*P+t
    yield a+E+q+R-h, v, c-e+A-B+P-Q+r-s

    w = -c+e+R-S
    u = A+C+Q+S
    v = a-c-B+E+P-q-R+s
    yield u, -2*(v+s)-t, w
    yield 2*u+e+A+Q+r-h, v, w

    w = A+C-P-R
    yield -a-b-C-E-p-q-R-S, -e-A+Q+r+h, w
    yield -2*p-t, -c+e+P-Q, w


def _edge_recipe():
    x = _basis(18)
    reverse = x[4:8] + x[:4] + tuple(x[j] for i in range(8,16,2) for j in (i+1,i)) + x[16:]
    direct, reflected = tuple(_edge_half(x)), tuple(_edge_half(reverse))
    for pair in range(6):
        for half in (direct, reflected):
            yield from half[2*pair:2*pair+2]


def _edge_terms(coordinates):
    recipe = tuple(_edge_recipe())
    views = tuple(_BlockView(coordinates, a) for a in range(3))
    shared = tuple((v.signs[0]*v.common,
                    (1 if a == 2 else v.signs[3])*(coordinates.D-2)*v.step)
                   for a,v in enumerate(views))
    for i,j in combinations(range(coordinates.D),2):
        local = coordinates.edge(i,j)
        bases = tuple(local+f for f in shared)
        for term in recipe:
            yield tuple(_Form(_linear((bases[a][k] for k in f),f.values()))
                        for a,f in enumerate(term))


# ---------- Boundary edges and the closing vertex ----------


def _edge_seeds(view, i, j):
    x = tuple(view.diagonal(b,i) for b in range(4))
    y = tuple(view.diagonal(b,j) for b in range(4))
    p = tuple(view.entry(b,i,j) for b in range(4))
    return (x[0]+x[2]-y[2]-y[3]-p[0]+p[3],
            y[2]+y[3]-p[2]-p[3], -x[0]-x[2]-p[1]-p[3],
            y[2]-x[0]-x[2]-x[3]+p[0]+view.heptad,
            y[0]+y[2]+y[3]-x[2]-p[3]-view.heptad)



def _heptad(seeds):
    """Five forms generate seven products; the last two forms are dependent."""
    expanded = []
    for e, a, b, c, ap in seeds:
        expanded.append((e, a, b, c, ap, -e-b, -e-a))
    A, B, C = expanded
    yield A[0], B[0], C[0]
    for start in (1, 4):
        for offset in range(3):
            i, j, k = (start + (offset+t) % 3 for t in range(3))
            yield A[i], -B[j], -C[k]


def _boundary_terms(coordinates):
    views = tuple(_BlockView(coordinates,a) for a in range(3))
    D, w = coordinates.D, coordinates.w
    for i in range(D):
        for barred in (0,1):
            row = [_cycle(v,i,i,D,barred) for v in views]
            if not barred:
                # Balance the merged product before projection.
                for a in (0,1):
                    row[a] = (D-3)*row[a]-views[a].signs[0]*views[a].difference
                row[2] = -w*row[2]/(D-3)
            else:
                # Only the first two axes need coincide in order to merge.
                row[2] = w*((1+w)*row[2]-views[2].difference)
            yield tuple(row)
        for ids in ((i,i,D),(i,D,i),(i,D,D),(D,i,i),(D,i,D),(D,D,i)):
            for barred in (0,1):
                yield _weighted(_mixed_factors(views,*ids,barred),coordinates,ids)
        for left,right in ((i,D),(D,i)):
            seeds = tuple(_edge_seeds(v,left,right) for v in views)
            for u,v,z in _heptad(seeds):
                yield coordinates.weight(left)*u, coordinates.weight(right)*v, 2*z


def _shared_basis(N, scale):
    D=N//2
    sums,traces=[],[]
    for b in range(4):
        br,bc=divmod(b,2)
        sums.append(_Form({(br*D+i)*N+bc*D+j:scale for i in range(D) for j in range(D)}))
        traces.append(_Form({(br*D+i)*N+bc*D+i:scale for i in range(D)}))
    return tuple(sums+traces)


def _center_raw(bases,D):
    """Four transformed aggregates, three heptad terms, one mixed correction."""
    a,b,c = bases
    def modes(x, sign):
        S0,S1,S2,S3,T0,T1,T2,T3=x
        return ((4*T0-(D+2)*T3+S3)/2,
                (S1+S2+sign*(S0-2*T0+D*T3))/2,
                (S3-(D-4)*T3-2*T0)/2,
                (S1+S2+sign*(S0+(D-2)*T3))/2)
    aa,bb=modes(a,1),modes(b,-1)
    yield aa[0]/3, bb[0], -(D-4)*c[4]/2+c[7]-c[3]/2
    yield aa[1], bb[1], (c[1]-c[2]-c[0]-(D-2)*c[4])/2
    yield aa[2]/3, bb[2], (c[3]+(D+2)*c[4]+4*c[7])/2
    yield aa[3], bb[3], (c[2]-c[1]+c[0]+D*c[4])/2+c[7]
    yield (a[0]-a[3])/2, b[6]-b[7]+(b[3]+b[0]-b[2]-b[1])/2, -2*(c[4]+c[6])/(D-2)
    yield -a[7]-a[6]+(a[3]+a[0]+a[2]+a[1])/2, b[4]-b[6], -(c[3]+c[0])/(D-2)
    yield a[4]+a[6], (b[0]-b[3])/2, (-2*c[7]+c[3]-c[0]+2*c[6]-c[2]+c[1])/(D-2)
    # Place each denominator on a trace factor, not on the polynomial row.
    n=((D-2)*(D-8)*c[4]-6*(D-2)*c[7]+2*c[0]+D*c[3])/2
    yield (a[4]-a[7])/(D-3), (-b[4]+b[7])/(D-2), n


# ---------- Weighted projection and sparse serialization ----------


def _project(form, N):
    """Project K*F using L=[I;-1.T/w] and R=[I-J/2,-1/2].

    For Lambda=diag(1,...,1,w), R*Lambda*L=I. The closing row
    coefficients are divisible by w. The final numerators are integral on K=6*(D-2)*(D-3).
    """
    D, d, M, weight = N//2, N//2+1, N+2, 2-N//2
    result, row_sums = {}, {}
    for index, value in form.items():
        row, column = divmod(index,M)
        br,i = divmod(row,d)
        bc,j = divmod(column,d)
        totals = row_sums.setdefault((br,bc),{})
        if i == D:
            value,remainder = divmod(value,weight)
            if remainder:
                raise ArithmeticError("nonintegral closing-row projection")
        totals[i] = totals.get(i,0)+value
        if j < D:
            rows = ((i,1),) if i < D else ((k,-1) for k in range(D))
            for k,sign in rows:
                target = (br*D+k)*N+bc*D+j
                result[target] = result.get(target,0)+2*sign*value
    for (br,bc),totals in row_sums.items():
        closing = totals.get(D,0)
        rows = range(D) if closing else (i for i in totals if i < D)
        for i in rows:
            shift = closing-totals.get(i,0)
            if shift:
                start = (br*D+i)*N+bc*D
                for j in range(D):
                    result[start+j] = result.get(start+j,0)+shift
    result = _Form({i:v for i,v in result.items() if v})/2
    return result, 6*(D-2)*(D-3)



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
            "construction": "LITA weighted closed-field aggregation",
        }, separators=(",", ":")))
        np.savez_compressed(path, **arrays)


def lita(N):
    """Construct the explicit rational weighted LITA scheme over Q."""
    _check_dimension(N)
    coordinates=_Coordinates(N)
    scheme=Scheme(N)
    for family in (_triangle_terms,_edge_terms,_boundary_terms):
        for term in family(coordinates):
            rows=tuple(_project(f,N) for f in term)
            if not all(f for f,d in rows):
                raise RuntimeError("unexpected zero projected factor")
            scheme.append(*rows)
    basis=_shared_basis(N,coordinates.grid)
    for term in _center_raw((basis,basis,basis),coordinates.D):
        if not all(term):
            raise RuntimeError("unexpected zero central factor")
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
