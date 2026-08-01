# Local Improvements to Trilinear Aggregation

This repository contains explicit rational schemes for square matrix
multiplication and their construction. The table lists the `⟨N×N×N : R⟩` schemes in `schemes/` with exponent `ω = log_N R`.

| N | rank R | ω |
|---:|---:|---:|
| 13 | 1420 | 2.82985 |
| 19 | 4002 | 2.81702 |
| 20 | 4297 | 2.79253 |
| 21 | 5183 | 2.80935 |
| 22 | 5518 | 2.78733 |
| 23 | 6570 | 2.80347 |
| 24 | 6935 | 2.78294 |
| 25 | 8180 | 2.79894 |
| 26 | 8574 | 2.77969 |
| 27 | 10027 | 2.79536 |
| 28 | 10451 | 2.77728 |
| 29 | 12128 | 2.79253 |
| 30 | 12582 | 2.77550 |
| 31 | 14499 | 2.79029 |
| 32 | 14983 | 2.77421 |
| 34 | 17670 | 2.77329 |
| 36 | 20659 | 2.77267 |
| 38 | 23966 | 2.77228 |
| 40 | 27607 | 2.77207 |
| 42 | 31598 | **2.77201** |
| 44 | 35955 | 2.77207 |

For `N=42`, rank `31598` gives the smallest exponent in the catalogue. For
`N=44`, rank `35955` improves the exponent `2.77320` reported by Schwartz and
Zwecher in [arXiv:2508.01748](https://arxiv.org/abs/2508.01748) to `2.77207`.

## Repository Contents

- `schemes/` contains the rational decompositions.
- `scripts/lita.py` constructs the even-dimensional LITA family.
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

# C = A @ B
A = np.random.normal(size=(N, N))
B = np.random.normal(size=(N, N))
C = np.einsum(               
    "qi,i,qj,j,qk->k",
    U, A.reshape(-1), V, B.reshape(-1), W,
    optimize=True,
).reshape(N, N)
```

## LITA

`scripts/lita.py` is a LITA construction for even
`N >= 18`:

```bash
python scripts/lita.py 22 schemes/22x22x22_r5518.npz
```

Its rank is

```text
R_even(N) = N^3/3 + 15*N^2/4 + 20*N/3 + 7.
```

This construction gives all even-dimensional schemes in `schemes/` except
the scheme for `N=20`.

## Verification

`scripts/verify.py` loads every file in `schemes/` and compares scheme-based
matrix multiplication with `A @ B`. It performs ten `float64` trials and ten
trials modulo each of `1000003`, `1000033`, and `1000037`.

```bash
python scripts/verify.py
```

## 2-Reducibility

The following property was the main guide in searching for improvements to trilinear aggregation.

```text
Let T = Σₜ uₜ⊗vₜ⊗wₜ, Xₜ = uₜ⊗vₜ and yₜ = wₜ.
The decomposition is 2-reducible if, for some p, Xₚ = Σ_{t≠p} αₜ Xₜ.
In this case the p-th term can be removed: T = Σ_{t≠p} Xₜ⊗(yₜ + αₜyₚ).
```

A fast 2-reducibility check is implemented in `src/reduce.h`. It searches for
linear dependencies among the `UV`, `UW`, or `VW` pair factors and applies the
resulting reduction to a scheme. The supporting header `src/scheme.h`
defines `SchemeQ` for rational coefficients and `SchemeP` for coefficients
modulo a prime.

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

An associated manuscript, *Local Improvements to Trilinear Aggregation*, is in preparation.
