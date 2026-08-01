"""Explicit rational LITA4 schemes for even square matrix multiplication.

The construction combines Pan's lifted trilinear aggregation with a source
gauge, a rational diagonal split, and one affine simplex gauge. All factors
are emitted directly from closed formulas, with no reduction passes.

API:
    rank = lita_rank(N)
    scheme = lita(N)
    materialize(N, "scheme.npz")

N must be even and at least 18.
"""

from array import array
from fractions import Fraction
import json
from math import gcd, lcm
from pathlib import Path
import sys

import numpy as np


A_AXIS, B_AXIS, C_AXIS = range(3)

OFF_SIGNS = (
    (1, 1, 1, 1, 1),
    (1, -1, -1, 1, -1),
    (1, 1, -1, -1, 1),
)
LOCAL_SIGNS = (
    (1, 1, 1, 1),
    (1, -1, -1, 1),
    (1, 1, -1, -1),
)
OFF_SEEDS = (
    (1, 0, 0, -1, 1),       # e
    (0, -1, 0, -1, 1),      # a
    (0, 0, -1, -1, 0),      # b
    (1, 0, 0, 0, 1),        # c
    (0, 0, 0, -1, 1),       # a'
    (1, 0, 1, 0, 1),        # b'
    (1, 1, 0, 0, 0),        # c'
)


def _check_dimension(N):
    if not isinstance(N, int) or N < 18 or N % 2:
        raise ValueError("N must be an even integer at least 18")


def _raw_rank(N):
    _check_dimension(N)
    return (4 * N**3 + 45 * N**2 + 116 * N + 84) // 12


def lita_rank(N):
    _check_dimension(N)
    return _raw_rank(N) - 3 * N


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


def _entry(M, i, j, coefficient=1):
    return {} if not coefficient else {i * M + j: coefficient}


def _linear(forms, coefficients):
    result = {}
    for form, coefficient in zip(forms, coefficients):
        _add_scaled(result, form, coefficient)
    return result


def _sum(forms):
    return _linear(forms, (1,) * len(forms))


def _scaled_form(form, scale):
    return form if scale == 1 else _linear((form,), (scale,))


def _quad(M, d, i, j, coefficients):
    return _linear(
        (
            _entry(M, i, j),
            _entry(M, i + d, j),
            _entry(M, i, j + d),
            _entry(M, i + d, j + d),
        ),
        coefficients,
    )


def _diagonal_fields(N):
    """Return the centered rational diagonal source fields."""
    _check_dimension(N)
    D = N // 2
    d = D + 1
    M = N + 2
    skew = [
        _quad(M, d, i, i, (0, Fraction(1, 2), Fraction(-1, 2), 0))
        for i in range(D)
    ]
    symmetric = [
        _quad(M, d, i, i, (0, Fraction(1, 2), Fraction(1, 2), 0))
        for i in range(D)
    ]

    def close(field):
        closure = _linear(field, (-1,) * len(field))
        return field + [closure]

    skew = close(skew)
    return skew, list(skew), close(symmetric)


class _SourceGauge:
    """Five coordinates of one lifted block and one centered source field."""

    def __init__(self, M, d, field, upper_sign):
        self.M = M
        self.d = d
        self.field = field
        self.upper_sign = upper_sign
        self.cache = {}

    def block(self, i, j):
        key = (i, j)
        block = self.cache.get(key)
        if block is not None:
            return block

        phi_i = self.field[i]
        lower = _entry(self.M, i + self.d, j)
        upper = _entry(self.M, i, j + self.d)
        _add_scaled(lower, phi_i, -1)
        _add_scaled(upper, phi_i, self.upper_sign)

        gradient = dict(self.field[j])
        _add_scaled(gradient, phi_i, -1)
        block = (
            _entry(self.M, i, j),
            lower,
            upper,
            _entry(self.M, i + self.d, j + self.d),
            gradient,
        )
        self.cache[key] = block
        return block


class _SimplexGauge:
    """One weighted-centered vertex field and its face/edge incidences.

    If p_i are the ordinary pivot sections and S=sum_i p_i, set

        tau_i = p_i - d*S/Q,          i < D,
        tau_D = -(sum_{i<D} tau_i)/(d+1).

    Thus tau is centered with weights (1,...,1,d+1).  Every aggregate
    correction is tau_i+tau_j+tau_k, while every
    off-diagonal correction is tau_i+tau_j-tau_D.  Thus one vertex field,
    rather than separate face and edge fields, controls the whole second
    gauge.  It satisfies sum_{i<D} tau_i + (d+1)*tau_D = 0.
    """

    def __init__(self, d, ordinary_sections):
        Q = 3 * d * d - 5 * d - 6
        total = _sum(ordinary_sections)
        shift = _scaled_form(total, Fraction(-d, Q))
        ordinary = [_sum((section, shift)) for section in ordinary_sections]
        infinity = _scaled_form(_sum(ordinary), Fraction(-1, d + 1))
        self.vertex = tuple(ordinary) + (infinity,)
        self.infinity = infinity
        self.face_cache = {}
        self.edge_cache = {}

    def face(self, i, j, k):
        key = (i, j, k)
        result = self.face_cache.get(key)
        if result is None:
            result = _sum((self.vertex[i], self.vertex[j], self.vertex[k]))
            self.face_cache[key] = result
        return result

    def edge(self, i, j):
        key = (i, j)
        result = self.edge_cache.get(key)
        if result is None:
            result = _linear(
                (self.vertex[i], self.vertex[j], self.infinity),
                (1, 1, -1),
            )
            self.edge_cache[key] = result
        return result


def _source_gauges(N, fields):
    d = N // 2 + 1
    M = N + 2
    return (
        _SourceGauge(M, d, fields[0], 1),
        _SourceGauge(M, d, fields[1], 1),
        _SourceGauge(M, d, fields[2], -1),
    )


def _simplex_gauges(N, source):
    D = N // 2
    d = D + 1
    q = d - 4
    eta = Fraction(d - 6, 2 * (d - 3))
    rows = (
        (Fraction(-1, d - 3), Fraction(-q, 2 * (d - 3)), 0, 0, 0),
        (0, 0, 0, eta, 0),
        (eta, 0, 0, 0, 0),
    )
    return tuple(
        _SimplexGauge(d, [
            _linear(source[axis].block(i, i), rows[axis])
            for i in range(D)
        ])
        for axis in range(3)
    )


def _cycle_factor(M, d, i, j, k, barred):
    shift = d if barred else 0
    return _sum((
        _entry(M, i + shift, j + shift),
        _entry(M, j + shift, k + shift),
        _entry(M, k + shift, i + shift),
    ))


def _mixed_factors(source, i, j, k, barred):
    A, B, C = source
    Aij, Ajk, Aki = A.block(i, j), A.block(j, k), A.block(k, i)
    Bij, Bjk, Bki = B.block(i, j), B.block(j, k), B.block(k, i)
    Cij, Cjk, Cki = C.block(i, j), C.block(j, k), C.block(k, i)
    low, high, diagonal, orbit = (2, 1, 3, -1) if barred else (1, 2, 0, 1)
    return (
        _linear((Ajk[low], Aki[high], Aij[diagonal], Aki[4]), (1, 1, -1, orbit)),
        _linear((Bjk[high], Bki[diagonal], Bij[low], Bjk[4]), (1, 1, 1, orbit)),
        _linear((Cij[high], Cjk[diagonal], Cki[low], Cij[4]), (1, 1, -1, -1)),
    )


def _local_seed_vectors(d):
    e = tuple(map(Fraction, (-1, 0, 0, 1)))
    u = (
        Fraction(d * (8 - d), d - 6),
        Fraction(-2 * d, d - 6),
        Fraction(d * (d - 2), d - 6),
        Fraction(0),
    )
    v = tuple(map(Fraction, (0, -1, 1, 0)))
    w = (Fraction(1, 2), Fraction(0), Fraction(1, 2), Fraction(0))

    def combine(vector, scale):
        return tuple(vector[t] + scale * e[t] for t in range(4))

    return (
        e,
        combine(u, Fraction(d, d - 6)),
        combine(v, Fraction(-(d - 6), d)),
        combine(w, Fraction(d - 3, 2 * d)),
        combine(tuple(-value for value in u), Fraction(d * (d - 7), d - 6)),
        combine(w, Fraction(3, 2 * d)),
        combine(v, Fraction(d - 6, d)),
    )


def _seed_forms(block, vectors, signs, scale=1):
    coordinates = tuple(_scaled_form(form, sign) for form, sign in zip(block, signs))
    return [
        _scaled_form(_linear(coordinates, vector), scale)
        for vector in vectors
    ]


def _heptad(A, B, C, central):
    e, a, b, c, ap, bp, cp = range(7)
    yield _scaled_form(A[e], central), B[e], C[e]
    yield A[a], B[b], C[c]
    yield A[b], B[c], C[a]
    yield A[c], B[a], C[b]
    yield A[ap], B[bp], C[cp]
    yield A[bp], B[cp], C[ap]
    yield A[cp], B[ap], C[bp]


def _off_heptad(source, simplex, i, j, d):
    blocks = tuple(gauge.block(i, j) for gauge in source)
    seeds = []
    for axis in range(3):
        scale = d if axis == C_AXIS else 1
        mode = _seed_forms(blocks[axis], OFF_SEEDS, OFF_SIGNS[axis], scale)
        edge = _scaled_form(simplex[axis].edge(i, j), scale)
        _add_scaled(mode[3], edge, -1)  # c
        _add_scaled(mode[4], edge, 1)   # a'
        seeds.append(mode)
    yield from _heptad(*seeds, central=-1)


def _extra_local_heptad(source, simplex, D, d):
    vectors = _local_seed_vectors(d)
    blocks = tuple(gauge.block(D, D)[:4] for gauge in source)
    seeds = []
    for axis in range(3):
        mode = _seed_forms(blocks[axis], vectors, LOCAL_SIGNS[axis])
        edge = simplex[axis].edge(D, D)
        _add_scaled(mode[1], edge, 2 * d)   # a
        _add_scaled(mode[4], edge, -2 * d)  # a'
        seeds.append(mode)
    central = Fraction(d * d - 11 * d + 27, d)
    yield from _heptad(*seeds, central=central)


def _ordinary_local(source, i, d):
    q = d - 4
    A, B, C = (gauge.block(i, i) for gauge in source)
    return (
        _linear(A, (Fraction(-4, q), 0, 0, Fraction(4, q), 0)),
        _linear(B, (0, 2, 0, -2, 0)),
        _linear(C, (Fraction(q * (d - 6), 8), Fraction(q * q, 8), 0, Fraction(-q, 4), 0)),
    )


def _lifted_terms(N):
    """Yield the final rank R0(N) - 3N identity before the Pan map."""
    _check_dimension(N)
    D = N // 2
    d = D + 1
    M = N + 2
    source = _source_gauges(N, _diagonal_fields(N))
    simplex = _simplex_gauges(N, source)

    for i in range(d):
        for j in range(d):
            for k in range(d):
                if i <= j < k or k < j <= i:
                    face = tuple(gauge.face(i, j, k) for gauge in simplex)
                    for barred in (0, 1):
                        orbit = -1 if barred else 1
                        factor = _cycle_factor(M, d, i, j, k, barred)
                        yield (
                            _linear((factor, face[0]), (1, -1)),
                            _linear((factor, face[1]), (1, -1)),
                            _linear((factor, face[2]), (1, -orbit)),
                        )

    for i in range(d):
        for j in range(d):
            for k in range(d):
                if i == j == k:
                    continue
                face = tuple(gauge.face(i, j, k) for gauge in simplex)
                for barred in (0, 1):
                    orbit = -1 if barred else 1
                    u, v, w = _mixed_factors(source, i, j, k, barred)
                    yield (
                        _linear((u, face[0]), (1, 1)),
                        _linear((v, face[1]), (1, -1)),
                        _linear((w, face[2]), (1, -orbit)),
                    )

    for i in range(d):
        for j in range(d):
            if i < D and i == j:
                yield _ordinary_local(source, i, d)
            elif i == j:
                yield from _extra_local_heptad(source, simplex, D, d)
            else:
                yield from _off_heptad(source, simplex, i, j, d)


def _project(factor, N):
    n = N // 2
    d = n + 1
    M = N + 2
    denominator = lcm(*(value.denominator for value in factor.values()))
    result = {}

    for index, value in factor.items():
        source_row, source_column = divmod(index, M)
        block_row, local_row = divmod(source_row, d)
        block_column, local_column = divmod(source_column, d)
        numerator = value.numerator * (denominator // value.denominator)

        if local_row < n:
            rows = ((block_row * n + local_row, 1),)
        else:
            rows = ((block_row * n + row, -1) for row in range(n))

        for target_row, row_sign in rows:
            base = target_row * N + block_column * n
            signed = numerator * row_sign
            for column in range(n):
                column_scale = d - 1 if local_column < n and column == local_column else -1
                target = base + column
                updated = result.get(target, 0) + signed * column_scale
                if updated:
                    result[target] = updated
                elif target in result:
                    del result[target]

    return result, denominator * d


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
        if transpose:
            N = transpose
            items = (((index % N) * N + index // N, value) for index, value in row.items())
        else:
            items = row.items()
        for index, numerator in sorted(items):
            common = gcd(abs(numerator), denominator)
            self.indices.append(index)
            self.numerators.append(numerator // common)
            self.denominators.append(denominator // common)
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
        }, separators=(",", ":")))
        np.savez_compressed(path, **arrays)


def lita(N):
    """Construct the affine-simplex LITA4 scheme over Q."""
    _check_dimension(N)
    scheme = Scheme(N)
    for u, v, w in _lifted_terms(N):
        if not u or not v or not w:
            raise RuntimeError("unexpected zero factor in emitted lifted term")
        u, v, w = _project(u, N), _project(v, N), _project(w, N)
        if not u[0] or not v[0] or not w[0]:
            raise RuntimeError("unexpected zero factor after the Pan map")
        scheme.append(u, v, w)
    expected = lita_rank(N)
    if scheme.rank != expected:
        raise RuntimeError(f"internal LITA rank mismatch: {scheme.rank} != {expected}")
    return scheme


def materialize(N, output=None):
    scheme = lita(N)
    if output is not None:
        scheme.save(output)
    return scheme


if __name__ == "__main__":
    if len(sys.argv) not in (2, 3):
        raise SystemExit("usage: python lita.py N [output.npz]")
    dimension = int(sys.argv[1])
    output = sys.argv[2] if len(sys.argv) == 3 else None
    result = materialize(dimension, output)
    print(f"N={dimension} rank={result.rank}")
    if output is not None:
        print(Path(output))
