# Local Improvements to Trilinear Aggregation

This repository contains explicit rational schemes for square matrix
multiplication and a direct implementation of the LITA3 construction. The
table lists the materialized schemes in `schemes/`. A scheme
⟨N×N×N : R⟩ has exponent ω = log_N R.

| N | rank R | ω |
|---:|---:|---:|
| 13 | 1420 | 2.82985 |
| 19 | 4002 | 2.81702 |
| 20 | 4297 | 2.79253 |
| 21 | 5183 | 2.80935 |
| 22 | 5519 | 2.78739 |
| 23 | 6570 | 2.80347 |
| 24 | 6949 | 2.78358 |
| 25 | 8180 | 2.79894 |
| 26 | 8633 | 2.78179 |
| 27 | 10027 | 2.79536 |
| 28 | 10514 | 2.77908 |
| 29 | 12128 | 2.79253 |
| 30 | 12605 | 2.77604 |
| 31 | 14499 | 2.79029 |
| 32 | 15055 | 2.77559 |
| 34 | 17696 | 2.77371 |
| 36 | 20686 | 2.77303 |
| 38 | 23995 | 2.77261 |
| 40 | 27637 | 2.77236 |
| 42 | 31630 | **2.77228** |
| 44 | 36054 | 2.77279 |

For `N=42`, rank `31630` gives the smallest exponent in the catalogue. For
`N=44`, rank `36054` improves the exponent `2.77320` reported by Schwartz and
Zwecher in [arXiv:2508.01748](https://arxiv.org/abs/2508.01748) to `2.77279`.

## Repository Contents

- `schemes/` contains the materialized rational decompositions.
- `scripts/lita.py` constructs the even-dimensional LITA3 family.
- `scripts/verify.py` checks the catalogue by random matrix multiplication.
- `src/scheme.h` defines sparse rational and prime-field schemes.
- `src/reduce.h` implements sparse 2-reduction.

## Scheme Format

Each file is named `schemes/{N}x{N}x{N}_r{rank}.npz` and stores three sparse
rational matrices `U`, `V`, and `W`. Their rows define

```math
T_{ijk} = \sum_{q=1}^{R} U_{qi} V_{qj} W_{qk},
```

where `T` is the matrix multiplication tensor. For flattened input matrices
`A` and `B`, the product is

```math
C_k = \sum_{i,j} T_{ijk} A_i B_j.
```

For each lowercase axis name `u`, `v`, or `w`, the archive contains CSR arrays
`{axis}_indptr`, `{axis}_indices`, `{axis}_numerators`, and
`{axis}_denominators`. `metadata_json` records the tensor dimensions, rank,
coefficient field, and format identifier.

The following example loads a scheme and uses it to multiply two matrices:

```python
import json
import numpy as np

path = "schemes/19x19x19_r4002.npz"

def read_axis(npz, name, rows, cols):
    indptr = npz[f"{name}_indptr"]
    indices = npz[f"{name}_indices"]
    values = npz[f"{name}_numerators"] / npz[f"{name}_denominators"]
    row = np.repeat(np.arange(rows), np.diff(indptr))
    axis = np.zeros((rows, cols), dtype=np.float64)
    np.add.at(axis, (row, indices), values)
    return axis

with np.load(path, allow_pickle=False) as npz:
    metadata = json.loads(str(npz["metadata_json"].tolist()))
    N = metadata["tensor"][0]
    R = metadata["rank"]
    U = read_axis(npz, "u", R, N * N)
    V = read_axis(npz, "v", R, N * N)
    W = read_axis(npz, "w", R, N * N)

A = np.random.normal(size=(N, N))
B = np.random.normal(size=(N, N))
C = np.einsum(
    "qi,i,qj,j,qk->k",
    U, A.reshape(-1), V, B.reshape(-1), W,
    optimize=True,
).reshape(N, N)

# C = A @ B
```

## LITA3

`scripts/lita.py` is a self-contained implementation of LITA3 for every even
`N >= 18`. It combines Pan's lifted trilinear aggregation with three centered
fields and a universal seven-product tensor. Every rational factor is emitted
directly from closed formulas. NumPy is used only to write the compressed NPZ
archive.

```python
from scripts.lita import lita3, lita3_rank

print(lita3_rank(18))
scheme = lita3(18)
scheme.save("18x18x18_r3300.npz")
```

The generator can also be run directly:

```bash
python scripts/lita.py 18 18x18x18_r3300.npz
```

Its rank is

```text
R_even(N) = (4*N^3 + 45*N^2 + 116*N + 84)/12 - floor(9*N/4).
```

The catalogue also contains dimension-specific schemes whose ranks need not
equal this uniform formula.

## Verification

`scripts/verify.py` loads every file in `schemes/` and compares scheme-based
matrix multiplication with `A @ B`. It performs ten `float64` trials and ten
trials modulo each of `1000003`, `1000033`, and `1000037`.

```bash
python scripts/verify.py
```

## 2-Reducibility

The following property was the main guide in searching for improvements to the
construction.

```text
Let T = Σₜ uₜ⊗vₜ⊗wₜ, Xₜ = uₜ⊗vₜ and yₜ = wₜ.
The decomposition is 2-reducible if, for some p, Xₚ = Σ_{t≠p} αₜ Xₜ.
In this case the p-th term can be removed: T = Σ_{t≠p} Xₜ⊗(yₜ + αₜyₚ).
```

A fast 2-reducibility check is implemented in `src/reduce.h`. It searches for
linear dependencies among the `UV`, `UW`, or `VW` pair factors, lifts a
single-prime certificate to small rational coefficients, and applies the
resulting reduction to a rational scheme. The supporting header `src/scheme.h`
defines `SchemeQ` for rational coefficients and `SchemeP` for coefficients
modulo a prime. The principal functions are
`find_two_reductions()`, `lift_two_reductions()`, and `two_reduce()`.

This is a strict generalization of reducibility: every decomposition reducible in the sense of Kauers and Moosbauer [arXiv:2212.01175](https://arxiv.org/abs/2212.01175) is 2-reducible, but the converse does not hold. For example, consider

```text
T = (1,0)×(1,0)×(1,0) + (0,1)×(0,1)×(0,1) + (1,1)×(1,1)×(1,1) + (1,−1)×(1,−1)×(1,−1).
```

This decomposition is not reducible, while its pair factors satisfy `(1,1)×(1,1)+(1,−1)×(1,−1) = 2(1,0)×(1,0)+2(0,1)×(0,1)`, so it is 2-reducible to

```text
T = (1,0)×(1,0)×(3,2) + (0,1)×(0,1)×(2,3) + (1,−1)×(1,−1)×(0,−2).
```

## Citation

```bibtex
@misc{khoruzhii2026lita,
  author       = {Kirill Khoruzhii and Luzian Serafin and Patrick Gel{\ss} and Sebastian Pokutta},
  title        = {Local Improvements to Trilinear Aggregation},
  year         = {2026},
  url          = {https://github.com/khoruzhii/lita}
}
```

An associated manuscript, *Local Improvements to Trilinear Aggregation*, is
in preparation.
