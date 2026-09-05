"""Explicit rational LITA schemes for even square matrix multiplication.

API:
    rank = lita_rank(N)
    scheme = lita(N)
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

# (alpha, beta) for the signed A, B, and C block matrices.
FACTOR_SIGNATURES = (
    (1, 1),
    (-1, -1),
    (1, -1),
)

# Rows in normalized Walsh coordinates (h0, h1, h2, h3, gradient).
OFF_WALSH_SEEDS = (
    (0, 2, 2, 0, 1),       # e
    (-2, 0, 2, 0, 1),      # a
    (-2, 2, 0, 0, 0),      # b
    (1, 1, 1, 1, 1),       # c
    (-1, 1, 1, -1, 1),     # a'
    (2, 0, 2, 0, 1),       # b'
    (2, 2, 0, 0, 0),       # c'
)


# ---------- Sparse linear forms ----------


def _check_dimension(N):
    if not isinstance(N, int) or N < 18 or N % 2:
        raise ValueError("N must be an even integer at least 18")


def _raw_rank(N):
    _check_dimension(N)
    return (4 * N**3 + 45 * N**2 + 116 * N + 84) // 12


def lita_rank(N):
    _check_dimension(N)
    return _raw_rank(N) - 7 * (N // 2) + 1


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


# ---------- Source and residual vertex fields ----------


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
        return tuple(field) + (_linear(field, (-1,) * len(field)),)

    skew = close(skew)
    return skew, tuple(skew), close(symmetric)


def _residual_field(d, ordinary_sections):
    """Project ordinary sections to the weighted-centered simplex field."""
    Q = 3 * d * d - 5 * d - 6
    total = _sum(ordinary_sections)
    shift = _scaled_form(total, Fraction(-d, Q))
    ordinary = tuple(_sum((section, shift)) for section in ordinary_sections)
    infinity = _scaled_form(_sum(ordinary), Fraction(-1, d + 1))
    return ordinary + (infinity,)


class _WalshGauge:
    """Source block, Walsh coordinates, and one residual vertex field."""

    def __init__(self, M, d, source_field, residual_field, alpha, beta):
        self.M = M
        self.d = d
        self.source_field = source_field
        self.residual_field = residual_field
        self.alpha = alpha
        self.beta = beta
        self.source_cache = {}
        self.walsh_cache = {}
        self.face_cache = {}
        self.edge_cache = {}

    def source_block(self, i, j):
        """Five source-gauge coordinates in unsigned block order."""
        key = (i, j)
        block = self.source_cache.get(key)
        if block is not None:
            return block

        phi_i = self.source_field[i]
        lower = _entry(self.M, i + self.d, j)
        upper = _entry(self.M, i, j + self.d)
        _add_scaled(lower, phi_i, -1)
        _add_scaled(upper, phi_i, self.alpha * self.beta)

        gradient = dict(self.source_field[j])
        _add_scaled(gradient, phi_i, -1)

        block = (
            _entry(self.M, i, j),
            lower,
            upper,
            _entry(self.M, i + self.d, j + self.d),
            gradient,
        )
        self.source_cache[key] = block
        return block

    def edge(self, i, j):
        """Residual edge incidence tau_i + tau_j - tau_D."""
        if self.residual_field is None:
            return {}
        key = (i, j)
        value = self.edge_cache.get(key)
        if value is None:
            value = _linear(
                (
                    self.residual_field[i],
                    self.residual_field[j],
                    self.residual_field[-1],
                ),
                (1, 1, -1),
            )
            self.edge_cache[key] = value
        return value

    def face(self, i, j, k):
        """Residual face incidence tau_i + tau_j + tau_k."""
        if self.residual_field is None:
            return {}
        key = (i, j, k)
        value = self.face_cache.get(key)
        if value is None:
            value = _sum(
                (
                    self.residual_field[i],
                    self.residual_field[j],
                    self.residual_field[k],
                )
            )
            self.face_cache[key] = value
        return value

    def walsh_block(self, i, j):
        """Final signed block in normalized Walsh coordinates."""
        key = (i, j)
        block = self.walsh_cache.get(key)
        if block is not None:
            return block

        source = self.source_block(i, j)
        z = (
            source[0],
            _scaled_form(source[1], self.alpha),
            _scaled_form(source[2], self.beta),
            _scaled_form(source[3], self.alpha * self.beta),
        )
        quarter = Fraction(1, 4)
        h0 = _linear(z, (quarter, quarter, quarter, quarter))
        h1 = _linear(z, (quarter, quarter, -quarter, -quarter))
        h2 = _linear(z, (quarter, -quarter, quarter, -quarter))
        h3 = _linear(z, (quarter, -quarter, -quarter, quarter))
        _add_scaled(h3, self.edge(i, j), -1)
        gradient = _scaled_form(source[4], self.alpha)

        block = (h0, h1, h2, h3, gradient)
        self.walsh_cache[key] = block
        return block


def _source_gauges(N, fields):
    d = N // 2 + 1
    M = N + 2
    return tuple(
        _WalshGauge(M, d, fields[axis], None, *FACTOR_SIGNATURES[axis])
        for axis in range(3)
    )


def _lita_fields(N):
    """Return ((phi_A, tau_A), (phi_B, tau_B), (phi_C, tau_C))."""
    _check_dimension(N)
    D = N // 2
    d = D + 1
    source_fields = _diagonal_fields(N)
    source = _source_gauges(N, source_fields)
    denominator = 2 * (d - 3)
    a_row = tuple(Fraction(x, denominator) for x in (d - 10, -2 * (d - 8), d - 6))
    bc_row = tuple(Fraction(x, denominator) for x in (-(d - 6), 0, d - 6))
    section_rows = (a_row, bc_row, bc_row)

    result = []
    for axis in range(3):
        ordinary = []
        for i in range(D):
            h = source[axis].walsh_block(i, i)
            ordinary.append(_linear((h[0], h[1], h[3]), section_rows[axis]))
        result.append((source_fields[axis], _residual_field(d, ordinary)))
    return tuple(result)


def _lita_gauges(N):
    d = N // 2 + 1
    M = N + 2
    return tuple(
        _WalshGauge(M, d, phi, tau, *FACTOR_SIGNATURES[axis])
        for axis, (phi, tau) in enumerate(_lita_fields(N))
    )


# ---------- Universal aggregates and heptads ----------

def _cycle_factor(M, d, i, j, k, barred):
    shift = d if barred else 0
    return _sum(
        (
            _entry(M, i + shift, j + shift),
            _entry(M, j + shift, k + shift),
            _entry(M, k + shift, i + shift),
        )
    )


def _mixed_factors(gauges, i, j, k, barred):
    A, B, C = gauges
    Aij, Ajk, Aki = A.source_block(i, j), A.source_block(j, k), A.source_block(k, i)
    Bij, Bjk, Bki = B.source_block(i, j), B.source_block(j, k), B.source_block(k, i)
    Cij, Cjk, Cki = C.source_block(i, j), C.source_block(j, k), C.source_block(k, i)
    low, high, diagonal, orbit = (2, 1, 3, -1) if barred else (1, 2, 0, 1)
    return (
        _linear((Ajk[low], Aki[high], Aij[diagonal], Aki[4]), (1, 1, -1, orbit)),
        _linear((Bjk[high], Bki[diagonal], Bij[low], Bjk[4]), (1, 1, 1, orbit)),
        _linear((Cij[high], Cjk[diagonal], Cki[low], Cij[4]), (1, 1, -1, -1)),
    )


def _local_walsh_seeds(d):
    """The extra-local heptad rows in normalized Walsh coordinates."""
    return (
        (0, -2, -2, 0, 0),
        (
            Fraction(4 * d, d - 6),
            Fraction(-2 * d * (d - 3), d - 6),
            Fraction(6 * d, d - 6),
            -2 * d,
            0,
        ),
        (0, Fraction(-12, d), Fraction(4 * (d - 3), d), 0, 0),
        (1, Fraction(-(d - 3), d), Fraction(3, d), 0, 0),
        (
            Fraction(-4 * d, d - 6),
            Fraction(6 * d, d - 6),
            Fraction(-2 * d * (d - 3), d - 6),
            2 * d,
            0,
        ),
        (1, Fraction(-3, d), Fraction(d - 3, d), 0, 0),
        (0, Fraction(-4 * (d - 3), d), Fraction(12, d), 0, 0),
    )


def _seed_forms(coordinates, rows, scale=1):
    return [
        _scaled_form(_linear(coordinates, row), scale)
        for row in rows
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


def _off_heptad(gauges, i, j, d):
    coordinates = tuple(gauge.walsh_block(i, j) for gauge in gauges)
    seeds = tuple(
        _seed_forms(coordinates[axis], OFF_WALSH_SEEDS, d if axis == C_AXIS else 1)
        for axis in range(3)
    )
    yield from _heptad(*seeds, central=-1)


def _local_heptad(gauges, i, d):
    rows = _local_walsh_seeds(d)
    seeds = tuple(
        _seed_forms(gauge.walsh_block(i, i), rows)
        for gauge in gauges
    )
    central = Fraction(d * d - 11 * d + 27, d)
    yield from _heptad(*seeds, central=central)


# ---------- Pair-factor dependency ----------


def _pair_field(gauges, d):
    """Return psi with sum(psi[:-1]) = 0 and psi[-1] = 0."""
    tail = tuple(gauges[A_AXIS].walsh_block(i, i)[1] for i in range(1, d - 1))
    return (_scaled_form(_sum(tail), -1),) + tail + ({},)


def _pair_cycle(field, i, j, k, barred, d):
    numerator = 1 if barred else -(2 * d - 7)
    return _scaled_form(_sum((field[i], field[j], field[k])), Fraction(numerator, d - 3))


def _pair_mixed(field, i, j, k, barred, d):
    weights = (d - 4, d - 4, -1 if barred else 2 * d - 7)
    return _scaled_form(
        _linear((field[i], field[j], field[k]), weights),
        Fraction(1, d - 3),
    )


def _pair_off(field, i, j, d):
    edge = _sum((field[i], field[j]))
    minus_i = _scaled_form(field[i], -1)
    minus_j = _scaled_form(field[j], -1)
    corner = _scaled_form(edge, Fraction(d - 4, d - 3))
    # Heptad order: e, a, b, c, a', b', c'.
    return edge, minus_j, minus_i, _scaled_form(corner, -1), corner, minus_j, minus_i


def _pair_local(gauges, d):
    """Return the single ordinary-local product left by the pair dependency."""
    A, B, C = gauges
    A_local = _sum(tuple(A.walsh_block(i, i)[1] for i in range(d - 1)))
    B_local = B.walsh_block(0, 0)
    C_local = C.walsh_block(0, 0)
    return (
        _scaled_form(A_local, -16 * (d - 8)),
        _sum((B_local[0], B_local[1])),
        _sum((C_local[0], C_local[1])),
    )


def _terms(N, gauges, reduced):
    D = N // 2
    d = D + 1
    M = N + 2
    pair_field = _pair_field(gauges, d) if reduced else None
    pair_scale = Fraction(d - 8, d - 4) if reduced else None

    for i in range(d):
        for j in range(d):
            for k in range(d):
                if i <= j < k or k < j <= i:
                    face = tuple(gauge.face(i, j, k) for gauge in gauges) if reduced else None
                    for barred in (0, 1):
                        orbit = -1 if barred else 1
                        factor = _cycle_factor(M, d, i, j, k, barred)
                        if not reduced:
                            yield factor, factor, factor
                        else:
                            correction = _pair_cycle(
                                pair_field, i, j, k, barred, d
                            )
                            yield (
                                _linear(
                                    (factor, face[A_AXIS], correction),
                                    (1, -1, pair_scale),
                                ),
                                _linear((factor, face[B_AXIS]), (1, -1)),
                                _linear((factor, face[C_AXIS]), (1, -orbit)),
                            )

    for i in range(d):
        for j in range(d):
            for k in range(d):
                if i == j == k:
                    continue
                face = tuple(gauge.face(i, j, k) for gauge in gauges) if reduced else None
                for barred in (0, 1):
                    orbit = -1 if barred else 1
                    u, v, w = _mixed_factors(gauges, i, j, k, barred)
                    if not reduced:
                        yield u, v, w
                    else:
                        correction = _pair_mixed(
                            pair_field, i, j, k, barred, d
                        )
                        yield (
                            _linear(
                                (u, face[A_AXIS], correction),
                                (1, 1, pair_scale),
                            ),
                            _linear((v, face[B_AXIS]), (1, -1)),
                            _linear((w, face[C_AXIS]), (1, -orbit)),
                        )

    for i in range(d):
        for j in range(d):
            if i == j:
                if not reduced:
                    yield from _local_heptad(gauges, i, d)
                elif i == 0:
                    yield _pair_local(gauges, d)
                elif i == D:
                    yield from _local_heptad(gauges, i, d)
                continue

            if not reduced:
                yield from _off_heptad(gauges, i, j, d)
            else:
                corrections = _pair_off(pair_field, i, j, d)
                for term, correction in zip(
                    _off_heptad(gauges, i, j, d), corrections
                ):
                    yield (
                        _linear((term[A_AXIS], correction), (1, pair_scale)),
                        term[B_AXIS],
                        term[C_AXIS],
                    )


def _family_terms(N, fields):
    """Yield the raw-rank source-gauge identity for centered fields."""
    _check_dimension(N)
    d = N // 2 + 1
    if len(fields) != 3 or any(len(field) != d for field in fields):
        raise ValueError("fields must be a triple of length-(N/2+1) sequences")
    for field in fields:
        if _sum(field):
            raise ValueError("each source field must be centered")
    yield from _terms(N, _source_gauges(N, fields), reduced=False)


def _lifted_terms(N):
    """Yield the final LITA identity before the Pan map."""
    _check_dimension(N)
    yield from _terms(N, _lita_gauges(N), reduced=True)


# ---------- Pan projection and sparse serialization ----------


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
    """Construct the Walsh-simplex LITA scheme over Q."""
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
