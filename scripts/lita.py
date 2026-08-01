"""Explicit rational LITA3 schemes for even square matrix multiplication.

The construction combines Pan's lifted trilinear aggregation with three
centered fields and a universal seven-product tensor. All factors are emitted
directly from closed formulas.

API:
    rank = lita3_rank(N)
    scheme = lita3(N)
    scheme.save("scheme.npz")

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

# Two signs contain all A/B/C asymmetry in the gauge coordinates.
FACTOR_SIGNS = (
    (1, 1),      # A
    (-1, -1),    # B
    (1, -1),     # C
)

# Unimodular map from gauge coordinates to heptad coordinates.
HEPTAD_COORDS = (
    (-1, 0, 0, 1, -1),
    (0, -1, 0, -1, 1),
    (0, 0, -1, -1, 0),
    (1, 0, 0, 0, 1),
    (0, 0, 0, -1, 1),
)

# (lower coordinate, upper coordinate, diagonal coordinate, gradient sign).
MIXED_ORBITS = (
    (1, 2, 0, 1),
    (2, 1, 3, -1),
)


def _check_dimension(N):
    if not isinstance(N, int) or N < 18 or N % 2:
        raise ValueError("N must be an even integer at least 18")


def _raw_rank(N):
    _check_dimension(N)
    return (4 * N**3 + 45 * N**2 + 116 * N + 84) // 12


def lita3_rank(N):
    _check_dimension(N)
    return _raw_rank(N) - (9 * (N // 2)) // 2


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


def _sum_forms(left, right):
    return _linear((left, right), (1, 1))


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


def _evaluate(coefficients, coordinates):
    return _linear(coordinates, coefficients)


def _heptad(a, b, c, *, central=1, output_scale=1):
    """One factor-cyclic heptad: center, 3-cycle, and corner orbit."""
    c = tuple(_scaled_form(form, output_scale) for form in c)
    yield _scaled_form(a[0], central), b[0], c[0]
    yield a[1], b[2], c[3]
    yield a[2], b[3], c[1]
    yield a[3], b[1], c[2]
    yield a[4], _sum_forms(b[0], b[2]), _sum_forms(c[0], c[1])
    yield _sum_forms(a[0], a[1]), b[4], _sum_forms(c[0], c[2])
    yield _sum_forms(a[0], a[2]), _sum_forms(b[0], b[1]), c[4]


def _colored_edges(D):
    """A sparse 1-factorization of a Mobius ladder on the active vertices."""
    active = D - (D % 2)
    half = active // 2
    return (
        tuple((r, r + half) for r in range(half)),
        tuple((2 * r + 1, (2 * r + 2) % active) for r in range(half)),
        tuple((2 * r, 2 * r + 1) for r in range(half)),
    )


def _tail_form(axis, d):
    if axis == A_AXIS:
        return (
            Fraction(-1, d), Fraction(2, d),
            Fraction(-(d - 2), d), Fraction(d - 7, d),
        )
    if axis == B_AXIS:
        return (
            Fraction(-(d - 6), 2 * d), Fraction(1, 2),
            Fraction(-1, 2), Fraction(d - 6, 2 * d),
        )
    return (
        Fraction(-(d - 3), d), Fraction(0),
        Fraction(1), Fraction(3, d),
    )


def _canonical_fields(N):
    """The gauge-fixing fields obtained from one section and three matchings."""
    D = N // 2
    d = D + 1
    M = N + 2
    fields = tuple([{} for _ in range(d)] for _ in range(3))

    # On a -> b, z0=z1=0 gives
    # F_a = alpha*x00 + x10,  F_b = x10 + beta*x11.
    for axis, (edges, signs) in enumerate(zip(_colored_edges(D), FACTOR_SIGNS)):
        alpha, beta = signs
        for a, b in edges:
            fields[axis][a] = _quad(M, d, a, b, (alpha, 1, 0, 0))
            fields[axis][b] = _quad(M, d, a, b, (0, 1, 0, beta))

    if D % 2:
        x = D - 1
        for axis in range(3):
            fields[axis][x] = _quad(M, d, x, x, _tail_form(axis, d))

    # The closure coordinate makes each field sum to zero.
    for axis in range(3):
        closure = {}
        for vertex in range(D):
            _add_scaled(closure, fields[axis][vertex], -1)
        fields[axis][D] = closure

    return fields


class Gauge:
    """Five gauge coordinates of a lifted block with one vertex field."""

    def __init__(self, M, d, field, alpha, beta):
        self.M = M
        self.d = d
        self.field = field
        self.alpha = alpha
        self.beta = beta
        self.upper_sign = alpha * beta
        self.coordinate_signs = (1, alpha, beta, alpha * beta, alpha)
        self.cache = {}

    def block(self, i, j):
        key = (i, j)
        cached = self.cache.get(key)
        if cached is not None:
            return cached

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


def _canonical_coordinates(gauge, block):
    """The same unimodular coordinates in all three factor modes."""
    signed = tuple(
        _scaled_form(form, sign)
        for form, sign in zip(block, gauge.coordinate_signs)
    )
    return tuple(_linear(signed, row) for row in HEPTAD_COORDS)


def _diagonal_coordinates(coordinates, d):
    """One common integer map from canonical to diagonal coordinates."""
    transform = (
        (2 * (d - 6), 0, 0, 0, 0),
        (2 * (d - 5), -4, 2 * (d - 2), 4 * (d - 6), 0),
        (-(d - 6), d, -d, 0, 0),
        (6, 0, 2 * d, 0, 0),
        (-2 * (d - 3), 0, -2 * d, 0, 0),
    )
    return tuple(
        tuple(_linear(mode, row) for row in transform)
        for mode in coordinates
    )


def _diagonal_heptad(coordinates, d):
    coordinates = _diagonal_coordinates(coordinates, d)
    central = Fraction(d * d - 11 * d + 27, (d - 6) * (d - 6))
    output_scale = Fraction(1, 8 * d * (d - 6))
    yield from _heptad(
        *coordinates, central=central, output_scale=output_scale
    )


def _boundary_triad(coordinates, d):
    """The singular diagonal specialization at the unmatched odd vertex."""
    x, y, z = _diagonal_coordinates(coordinates, d)
    output_scale = Fraction(1, 8 * d * (d - 6))
    boundary_central = Fraction(d - 9, (d - 6) * (d - 6))
    yield _scaled_form(x[0], boundary_central), y[0], _scaled_form(z[0], output_scale)
    yield x[2], y[3], _scaled_form(z[1], output_scale)
    yield x[3], y[1], _scaled_form(z[2], output_scale)


def _cycle_factor(M, d, i, j, k, barred):
    shift = d if barred else 0
    return {
        (i + shift) * M + j + shift: 1,
        (j + shift) * M + k + shift: 1,
        (k + shift) * M + i + shift: 1,
    }


def _mixed_factors(gauges, i, j, k, orbit):
    lower, upper, diagonal, gradient_sign = MIXED_ORBITS[orbit]
    A, B, C = gauges
    Aij, Ajk, Aki = A.block(i, j), A.block(j, k), A.block(k, i)
    Bij, Bjk, Bki = B.block(i, j), B.block(j, k), B.block(k, i)
    Cij, Cjk, Cki = C.block(i, j), C.block(j, k), C.block(k, i)

    return (
        _linear(
            (Ajk[lower], Aki[upper], Aij[diagonal], Aki[4]),
            (1, 1, -1, gradient_sign),
        ),
        _linear(
            (Bjk[upper], Bki[diagonal], Bij[lower], Bjk[4]),
            (1, 1, 1, gradient_sign),
        ),
        _linear(
            (Cij[upper], Cjk[diagonal], Cki[lower], Cij[4]),
            (1, 1, -1, -1),
        ),
    )


def _terms(N, fields, *, exceptional=None, omit_zero=False):
    D = N // 2
    d = D + 1
    M = N + 2
    gauges = tuple(
        Gauge(M, d, field, *signs)
        for field, signs in zip(fields, FACTOR_SIGNS)
    )

    for i in range(d):
        for j in range(d):
            for k in range(d):
                if i <= j < k or k < j <= i:
                    for barred in (False, True):
                        factor = _cycle_factor(M, d, i, j, k, barred)
                        yield factor, factor, factor

    for i in range(d):
        for j in range(d):
            for k in range(d):
                if i == j == k:
                    continue
                yield _mixed_factors(gauges, i, j, k, 0)
                yield _mixed_factors(gauges, i, j, k, 1)

    for i in range(d):
        for j in range(d):
            blocks = tuple(gauge.block(i, j) for gauge in gauges)
            coordinates = tuple(
                _canonical_coordinates(gauges[axis], blocks[axis])
                for axis in range(3)
            )
            if i == j:
                if i == exceptional:
                    yield from _boundary_triad(coordinates, d)
                else:
                    yield from _diagonal_heptad(coordinates, d)
                continue

            for term in _heptad(*coordinates, output_scale=d):
                if not omit_zero or (term[0] and term[1] and term[2]):
                    yield term


def _lifted_terms(N):
    """Yield the final lifted LITA3 terms selected by the three matchings."""
    _check_dimension(N)
    D = N // 2
    exceptional = D - 1 if D % 2 else None
    yield from _terms(
        N,
        _canonical_fields(N),
        exceptional=exceptional,
        omit_zero=True,
    )


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
            items = (
                ((index % N) * N + index // N, value)
                for index, value in row.items()
            )
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


def lita3(N):
    """Construct the universal-heptad LITA3 scheme over Q."""
    _check_dimension(N)
    scheme = Scheme(N)

    for u, v, w in _lifted_terms(N):
        if not u or not v or not w:
            raise RuntimeError("unexpected zero factor in emitted lifted term")
        u, v, w = _project(u, N), _project(v, N), _project(w, N)
        if not u[0] or not v[0] or not w[0]:
            raise RuntimeError("unexpected zero factor after the Pan map")
        scheme.append(u, v, w)

    expected = lita3_rank(N)
    if scheme.rank != expected:
        raise RuntimeError(
            f"internal LITA3 rank mismatch: {scheme.rank} != {expected}"
        )
    return scheme


def materialize(N, output=None):
    scheme = lita3(N)
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
