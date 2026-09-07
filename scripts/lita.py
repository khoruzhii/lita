"""Explicit rational LITA schemes for even square matrix multiplication.

API:
    rank = lita_rank(N)
    scheme = lita(N)
    scheme.save("scheme.npz")

N must be even and at least 4. Coefficients in the saved CSR arrays must
fit signed int64; overflow is reported, never rounded or wrapped.
"""

from array import array
from fractions import Fraction
from itertools import combinations, permutations
import json
from math import gcd, lcm
from pathlib import Path
import sys

import numpy as np


# ---------- Sparse linear forms ----------


def _check_dimension(N):
    if type(N) is not int or N < 4 or N % 2:
        raise ValueError("N must be an even integer at least 4")


def lita_rank(N):
    _check_dimension(N)
    D = N // 2
    return 16 * (D + 1) * D * (D - 1) // 6 + 25 * D * (D - 1) // 2 + 28 * D + 22


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


def _entry(N, i, j):
    return {i * N + j: 1}


def _linear(forms, coefficients):
    result = {}
    for form, coefficient in zip(forms, coefficients):
        _add_scaled(result, form, coefficient)
    return result


def _sum(forms):
    result = {}
    for form in forms:
        _add_scaled(result, form, 1)
    return result


def _scaled_form(form, scale):
    return form if scale == 1 else _linear((form,), (scale,))


def _integer_row(form):
    denominator = lcm(*(value.denominator for value in form.values()))
    return {i: value.numerator * (denominator // value.denominator)
            for i, value in form.items()}, denominator


class _Form(dict):
    """Sparse exact linear forms with the usual arithmetic operations."""

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
        return self * (Fraction(1) / scalar)


def _basis(size):
    return tuple(_Form({i: 1}) for i in range(size))


# ---------- Closed blocks and shared forms ----------


class _Coordinates:
    """Sparse forms on the lifted blocks; the Pan map is applied only at output.

    The closing row and column have zero sums. Thus T=Y_DD,
    row_i(Y)=-Y_iD and col_i(Y)=-Y_Di need no expanded sums.
    """

    def __init__(self, N):
        self.N, self.D = N, N // 2
        D, d = self.D, self.D + 1
        self.shared = tuple(form for b in range(4) for form in (
            self.entry(b, D, D), _sum(self.entry(b, i, i) for i in range(D))))
        Q = 3*D*D + D - 8
        g, h = Fraction(d, Q), Fraction((D-5)*d, 2*Q)
        self.common = (
            _linear((self.shared[3], self.shared[5], self.shared[7]), (g, g, -(D-7)*g)),
            _linear((self.shared[3], self.shared[5]), (h, h)),
            _linear((self.shared[3], self.shared[5]), (h, -h)),
        )

    def entry(self, block, i, j):
        br, bc = divmod(block, 2)
        d = self.D + 1
        return _entry(self.N + 2, br*d + i, bc*d + j)

    def edge(self, i, j, axis):
        diagonal = tuple(self.entry(b, v, v) for v in (i, j) for b in range(4))
        cross = tuple(self.entry(b, u, v) for b in range(4) for u, v in ((i, j), (j, i)))
        return (self.common[axis],) + diagonal + cross


# ---------- Cycles and mixed aggregates ----------


_BLOCK_SIGNS = ((1, 1, 1, 1), (-1, 1, 1, -1), (-1, 1, -1, 1))


class _BlockView:
    """Signed blocks with the closing diagonal continued by minus trace."""

    def __init__(self, coordinates, axis):
        self.coordinates = coordinates
        self.signs = _BLOCK_SIGNS[axis]
        self.D = coordinates.D
        self.G = coordinates.common[axis]

    def entry(self, block, i, j):
        return _scaled_form(self.coordinates.entry(block, i, j), self.signs[block])

    def diagonal(self, block, i):
        if i == self.D:
            return _scaled_form(self.coordinates.shared[2*block+1], -self.signs[block])
        return self.entry(block, i, i)


def _cycle(view, i, j, k, barred):
    block = 3 * barred
    forms = [view.entry(block, u, v) for u, v in ((i, j), (j, k), (k, i))]
    forms += [view.diagonal(block, v) for v in (i, j, k)]
    forms.append(view.G)
    closing = sum(v == view.D for v in (i, j, k))
    return _scaled_form(_linear(forms, (1, 1, 1, -1, -1, -1, closing)),
                        view.signs[block])


def _mixed(view, i, j, k, barred):
    Y, z = view.entry, view.diagonal
    u, v = (j, i) if barred else (i, j)
    block = 3 * barred
    # The same diagonal field h=z00+z10+z11 handles both parities.
    forms = (Y(2-barred, j, k), Y(1+barred, k, i), Y(block, i, j),
             z(0, u), z(2, u), z(3, u), z(2, v), z(block, k))
    result = _linear(forms, (1, 1, -1, 1, 1, 1, -1, 1))
    return _add_scaled(result, view.G, -sum(v == view.D for v in (i, j, k)))


def _mixed_factors(views, i, j, k, barred):
    A, B, C = views
    return (_mixed(A, i, j, k, barred), _mixed(B, k, i, j, barred),
            _scaled_form(_mixed(C, j, k, i, barred), -1 if barred else 1))


def _triangle_terms(coordinates):
    views = tuple(_BlockView(coordinates, axis) for axis in range(3))
    for vertices in combinations(range(coordinates.D + 1), 3):
        for i, j, k in permutations(vertices):
            for barred in (0, 1):
                if i < j < k or k < j < i:
                    yield tuple(_cycle(view, i, j, k, barred) for view in views)
                yield _mixed_factors(views, i, j, k, barred)


# ---------- Six paired edge blocks and their reversals ----------


def _edge_half(x, D):
    """Six rank-two blocks, with one shared factor in each pair.

    (a,b,c,e), (A,B,C,E) are the diagonal blocks at the two vertices.
    (p,q,r,s), (P,Q,R,S) are their forward and reverse entries.
    All four-tuples use block order 00, 01, 10, 11.
    """
    G = x[0]
    a, b, c, e = x[1:5]
    A, B, C, E = x[5:9]
    p, P, q, Q, r, R, s, S = x[9:17]

    # Shared first factor: the two outgoing/incoming row differences.
    u = a + c - p - r
    v = D*c - e - A + B + (D+1)*C - 2*E - P + Q - D*r + (1-D)*R + s
    yield u, v, -a + C + p - R
    yield u, v + G + (D-1)*(A-C+R) + 2*P, -C + E - p + q

    u = c + e - R - S
    w = a + D*c + (1-D)*A - B - E + D*P - q + (1-D)*r + R - s
    yield u, -e + A - 2*C - Q + r + 2*S, w
    yield u, A - C - Q + S, -2*w + G + (D-1)*(C-E-r) - 2*s

    # Shared second factor: two signed edge differences.
    v = -C + E + r - s
    u = 2*a + b + 2*c + e + A + D*C + (D-1)*E - p + Q - D*r + (1-D)*s + S
    yield u, v, -a + E + q - R
    yield G + (1-D)*(e+E-s) - 2*S, v, -a - c + q + s

    v = a - c - p + r
    yield C + E + p + q, v, -G + (D-1)*(c-A-r) - 2*P
    yield a + E + q + R, v, c - e + A - B + P - Q + r - s

    # Shared third factor: two incoming/outgoing row differences.
    w = -c + e + R - S
    u = A + C + Q + S
    v = a - c - B + E + P - q - R + s
    yield u, -2*v - G + (1-D)*c + (1-D)*E + (D-1)*R - 2*s, w
    yield 2*u + e + A + Q + r, v, w

    w = A + C - P - R
    u = -a - b + (D-1)*(c+e-A+P) - D*C - E - p - q - R - D*S
    yield u, -e - A + Q + r, w
    yield G + (1-D)*(a+A) - 2*p + (D-1)*P, -c + e + P - Q, w


def _edge_recipe(D):
    x = _basis(17)
    # Reverse the two vertices: diagonals exchange, and Y_ij becomes Y_ji.
    reverse = (x[0],) + x[5:9] + x[1:5] + tuple(x[j] for i in range(9, 17, 2) for j in (i+1, i))
    direct = tuple(_edge_half(x, D))
    reflected = tuple(_edge_half(reverse, D))
    d, Q = D+1, 3*D*D+D-8
    q = Fraction
    scales = (((1, 1), (1, 1)), ((1, 1), (2, 1)),
              ((1, 1), (q(Q, d), q(d, Q))), ((q(d, 2), 1), (d, 1)),
              ((q(3, 2), q(2, 3)), (1, 1)), ((q(1, d), d), (1, 1)))
    for pair, normalizations in enumerate(scales):
        for half in (direct, reflected):
            for term, scale in zip(half[2*pair:2*pair+2], normalizations):
                yield _balance(term, *scale)
    # The last product uses symmetric second differences on the edge.
    delta = tuple(x[9+2*b]+x[10+2*b]-x[1+b]-x[5+b] for b in range(4))
    yield (D-1)*(delta[0]-delta[3]), delta[2], delta[2]


def _edge_terms(coordinates):
    recipe = tuple(_edge_recipe(coordinates.D))
    for i, j in combinations(range(coordinates.D), 2):
        bases = tuple(coordinates.edge(i, j, axis) for axis in range(3))
        for term in recipe:
            yield tuple(_linear((bases[a][j] for j in row), row.values()) for a, row in enumerate(term))


# ---------- One oriented heptad rule for every edge ----------


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


def _edge_seeds(view, i, j):
    """One oriented rule for both ordinary and closing heptads."""
    x = tuple(_Form(view.diagonal(b, i)) for b in range(4))
    y = tuple(_Form(view.diagonal(b, j)) for b in range(4))
    p = tuple(_Form(view.entry(b, i, j)) for b in range(4))
    closing = (i == view.D) + (j == view.D)
    H = _Form(view.G) * (closing + Fraction(1, view.D+1))
    return (x[0]+x[2]-y[2]-y[3]-p[0]+p[3],
            y[2]+y[3]-p[2]-p[3], -x[0]-x[2]-p[1]-p[3],
            y[2]-x[0]-x[2]-x[3]+p[0]+H,
            y[0]+y[2]+y[3]-x[2]-p[3]-H)


def _edge_aggregates(coordinates, i, j):
    """The 28-term edge identity: 2 cycles, 12 mixed products, 2 heptads.

    For two ordinary vertices, _edge_recipe is its 25-term replacement.
    Reversing a heptad simply exchanges its endpoints.
    """
    views = tuple(_BlockView(coordinates, a) for a in range(3))
    for barred in (0, 1):
        mean = tuple(_linear((_cycle(v, i, i, j, barred), _cycle(v, j, j, i, barred)),
                             (Fraction(1, 2), Fraction(1, 2))) for v in views)
        yield mean[0], mean[1], _scaled_form(mean[2], 2)
        yield _mixed_factors(views, i, i, j, barred)
    for ids in ((i, j, i), (i, j, j), (j, i, i), (j, i, j), (j, j, i)):
        for barred in (0, 1):
            yield _mixed_factors(views, *ids, barred)
    for left, right in ((i, j), (j, i)):
        seeds = tuple(_edge_seeds(v, left, right) for v in views)
        for u, v, w in _heptad(seeds):
            yield u, v, _scaled_form(w, coordinates.D+1)


def _boundary_scales(D):
    q = Fraction
    d, Q = D + 1, 3*D*D + D - 8
    return (
        (2, 1), (1, 1), (2, 1), (2, 1), (1, 1), (1, 1), (1, 1),
        (q(2, 3), q(1, 2)), (1, 1), (2, q(1, 2)), (1, 1), (1, 1), (1, 1), (1, 1),
        (2, 2), (q(d, 2), q(D, 3*d)), (D, q(Q, 4*D)), (q(Q, 3*D), 3),
        (q(Q, D), d), (d, q(D, 6*d)), (q(4*D, Q), q(Q, 4*D)),
        (2, 2), (q(d, 2), q(D, 3*d)), (D, q(Q, 4*D)), (q(Q, D), 1),
        (q(Q*d, 2*D), 2), (d, q(4*D, Q)), (q(4*D, Q), q(Q, 4*D)),
    )


def _boundary_terms(coordinates):
    scales = _boundary_scales(coordinates.D)
    for vertex in range(coordinates.D):
        for term, scale in zip(_edge_aggregates(coordinates, vertex, coordinates.D), scales):
            yield _balance(term, *scale)


# ---------- Three shared products and six residual matrix slices ----------


def _center_coordinates(coordinates):
    """One shared-coordinate rule, read in the three signed block views."""
    D = coordinates.D
    h = Fraction(1, D+2)
    result = []
    for axis in range(3):
        view = _BlockView(coordinates, axis)
        s = view.signs
        T0, t0, T1, t1, T2, t2, T3, t3 = (
            _Form(f) for b in range(4) for f in (
                view.entry(b, D, D), _scaled_form(view.diagonal(b, D), -1)))
        G = _Form(view.G)
        result.append((s[0]*(T0+h*t3), s[0]*(t0-t3), T1-t2-(1+h)*t3,
                       (1 if axis == 0 else 2)*(G+(D+1)*h*t3),
                       s[2]*(T2+t2+(1-h)*t3), s[3]*(T3+h*t3)))
    return tuple(result)


def _center_raw(bases, D):
    """Three common products leave slices of lengths 3, 4, 2, 5, 2, 3."""
    a, b, c = (tuple(map(_Form, axis)) for axis in bases)
    b0, b1, b2, b3, b4, b5 = b
    c0, c1, c2, c3, c4, c5 = c
    u = b0+b1-b3/2
    v = b1+b2-b5
    x = c0+c1-c3/2
    y = c0-c2
    z = c0+c1+c5

    yield (-a[2]-a[4]+a[5])/4, 2*(b1+b2+b4+b5)-3*b3, 2*(c1+c2-c4-c5)-3*c3
    yield (-a[0]+a[2]+a[4])/4, 2*(b0+2*b1+b2+b4)-3*b3, 2*(c0+2*c1+c2-c4)-3*c3
    yield -(D+1)*a[1]-(2*D+3)*a[3], 2*b0-b2-b4, c1+c2+c5

    # Each remaining pair is a rank-one summand of one matrix slice.
    slices = (
        ((Fraction(3, 4), (D+22)*b0/3+(D+8)*b1-(3*D+26)*b3/6, x),
         (6, -(D+1)*(b0+3*b1)/2+(3*D+5)*b3/4, y/6),
         (Fraction(3, 2), b0+b1+b3/2-2*b4, -(D+1)*(2*c1+c2)/3+(2*D+3)*c3/6)),
        ((Fraction(1, 2), -u, (D+6)*c0/2+(4-9*D)*c1/2-4*D*c2+(9*D-2)*c3/4+4*D*(c4-c5)),
         (1, -(D+2)*(b1+b2)-b4+b5, -y-c4+c5),
         (1, (D-4)*b0/2+b1/2+3*(b2+b4-b5), -x),
         (1, 2*b0+(1-D)*b1-2*b5, z)),
        ((1, (b0+b1-b4)/2, 2*(D+1)*(c4-c1)+(2*D+3)*c3),
         (1, (2*D+3)*b3/2-(D+1)*b4, c5-c4)),
        ((2, b0+b1-b5, (1-3*D)*c0+4*c1+(3*D+3)*(c2-c4)+(2-9*D)*c3/16+(17*D+62)*c5/8),
         (2, b0+3*b1+b2-2*b5, D*c0-3*c1/2-(2*D+3)*(c2-c4)/2-(D+3)*c5),
         (2, -u/2, (4-7*D)*(c0+c5)/4+(2-9*D)*c1/4),
         (2, D*b0/4-b1/4+3*b4/2-b5, -z),
         (2, -b0+b5, -D*(c0+c1)-(D+1)*c3/8-(2*D+3)*c4/2+(5-D)*c5/4)),
        ((1, -(D+1)*(2*b1+b2)+(2*D+3)*b3/2, y),
         (1, -v/2, 2*(D+1)*(c1+c2)-(2*D+3)*c3)),
        ((1, -v, 5*c3-c4+11*c5),
         (Fraction(1, 2), 2*(b1+b2-b3), (2*D+13)*c3/2+D*c4+11*c5),
         (Fraction(1, 2), 3*b3-2*b5, (3*D+24)*c3/8-c4+(D+24)*c5/4)),
    )
    for first, pairs in zip(a, slices):
        for scale, second, third in pairs:
            yield scale*first, second, third


def _balance(term, s, t):
    """Redistribute a scalar between axes; the tensor is unchanged."""
    u, v, w = term
    return (_scaled_form(u, s), _scaled_form(v, t),
            _scaled_form(w, Fraction(1) / (s*t)))


def _center22_scales(D):
    q = Fraction
    d, Q = D + 1, 3*D*D + D - 8
    return (
        (4, q(1, 2)), (4, q(1, 2)), (q(3*Q, D*d), 1),
        (q(D+2, 3), q(3, D+2)), (q(1, 3), q(1, d)), (q(d, 3), 1),
        (3, q(2, d)), (1, 1), (q(d, 2), 1), (d, q(1, d)),
        (q(1, 2), 2), (q(1, 2), 2), (q(D+2, 2), q(1, D+2)),
        (3, q(1, d)), (q(D+2, 4), 2), (q(D+2, 2), q(2, D+2)),
        (q(D+2, 2), q(d, D+2)), (1, q(1, d)), (1, 2*d),
        (q(D+2, d), d), (D+2, q(d, 2*(D+2))), (1, q(1, 2)),
    )


def _center_terms(coordinates):
    factors = _center_raw(_center_coordinates(coordinates), coordinates.D)
    for term, scale in zip(factors, _center22_scales(coordinates.D)):
        yield _balance(term, *scale)


def _terms(N):
    coordinates = _Coordinates(N)
    yield from _triangle_terms(coordinates)
    yield from _edge_terms(coordinates)
    yield from _boundary_terms(coordinates)
    yield from _center_terms(coordinates)


# ---------- Pan projection and sparse serialization ----------


def _project(form, N):
    """Apply the block map L.T * F * R.T using integer row sums.

    L=[I;-1.T], R=[I-J/d,-1/d], and R*L=I. For one lifted block F,
    its output coefficient (i,j) is F_ij-F_Dj-(s_i-s_D)/d,
    where s_i=sum_k F_ik. A common denominator is cleared first.
    """
    D, d, M = N // 2, N // 2 + 1, N + 2
    coefficients, denominator = _integer_row(form)
    result, row_sums = {}, {}
    for index, value in coefficients.items():
        row, column = divmod(index, M)
        br, i = divmod(row, d)
        bc, j = divmod(column, d)
        totals = row_sums.setdefault((br, bc), {})
        totals[i] = totals.get(i, 0) + value
        if j < D:
            rows = ((i, 1),) if i < D else ((k, -1) for k in range(D))
            for k, sign in rows:
                target = (br*D+k)*N + bc*D+j
                result[target] = result.get(target, 0) + d*sign*value
    for (br, bc), totals in row_sums.items():
        closing = totals.get(D, 0)
        rows = range(D) if closing else (i for i in totals if i < D)
        for i in rows:
            shift = closing - totals.get(i, 0)
            if shift:
                base = (br*D+i)*N + bc*D
                for j in range(D):
                    result[base+j] = result.get(base+j, 0) + shift
    return {i: v for i, v in result.items() if v}, denominator*d


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
            "construction": "LITA rational edge and central replacement",
        }, separators=(",", ":")))
        np.savez_compressed(path, **arrays)


def lita(N):
    """Construct the explicit rational LITA scheme over Q."""
    _check_dimension(N)
    scheme = Scheme(N)
    for u, v, w in _terms(N):
        u, v, w = (_project(f, N) for f in (u, v, w))
        if not u[0] or not v[0] or not w[0]:
            raise RuntimeError("unexpected zero factor after the Pan map")
        scheme.append(u, v, w)
    expected = lita_rank(N)
    if scheme.rank != expected:
        raise RuntimeError(f"internal LITA rank mismatch: {scheme.rank} != {expected}")
    return scheme


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python lita.py N [output.npz]")
    dimension = int(sys.argv[1])
    output = sys.argv[2] if len(sys.argv) == 3 else None
    result = lita(dimension)
    if output is not None:
        result.save(output)
    print(f"N={dimension} rank={result.rank}")
    if output is not None:
        print(Path(output))
