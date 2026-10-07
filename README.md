# Local Improvements to Trilinear Aggregation

This repository contains rational schemes for square matrix multiplication and their construction. These constructions build on Victor Pan’s trilinear aggregation techniques ([1978](https://doi.org/10.1109/SFCS.1978.34), [1982](https://doi.org/10.1016/0898-1221(82)90037-2)). The table lists the even-dimensional `⟨N×N×N : R⟩` schemes in `schemes/` with exponent `ω = log_N R`. Odd-dimensional schemes are listed in [LITA odd](#lita-odd).

| N | rank R | ω |
|---:|---:|---:|
| 14 | 1593 | 2.79394 |
| 16 | 2236 | 2.78168 |
| 18 | 3031 | 2.77357 |
| 20 | 3994 | 2.76812 |
| 22 | 5141 | 2.76444 |
| 24 | 6488 | 2.76198 |
| 26 | 8051 | 2.76037 |
| 28 | 9846 | 2.75938 |
| 30 | 11889 | 2.75884 |
| 32 | 14196 | **2.75864** |

For `N=32`, rank `14196` gives the smallest exponent in this catalogue and improves the `N=44` exponent `2.77320` reported by Schwartz and Zwecher in [arXiv:2508.01748](https://arxiv.org/abs/2508.01748) to `2.75864`.

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

path = "schemes/19x19x19_r3981.npz"

def read_axis(npz, name, rows, cols):
    indptr = npz[f"{name}_indptr"]
    indices = npz[f"{name}_indices"]
    values = npz[f"{name}_numerators"] / npz[f"{name}_denominators"]
    row = np.repeat(np.arange(rows), np.diff(indptr))
    axis = np.zeros((rows, cols), dtype=np.float64)
    np.add.at(axis, (row, indices), values)
    return axis

with np.load(path, allow_pickle=False) as npz:
    metadata = json.loads(npz["metadata_json"].item())
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
`N ≥ 8`:

```bash
python scripts/lita.py 22 schemes/22x22x22_r5141.npz
```

Its rank is

```text
R_even(N) = N^3/3 + 3*N^2 + 37*N/6 + 4.
```

For this family, `ω(N) = log_N R_even(N)` is minimized over even `N ≥ 8`
at `N=32`, with `R=14196` and `ω ≈ 2.75864`.

This construction gives all even-dimensional schemes in `schemes/`.

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

Let T = Σₜ uₜ⊗vₜ⊗wₜ , Xₜ = uₜ⊗vₜ and yₜ = wₜ.
The decomposition has block flip if, for some u' and v', u'⊗v' = -Xₚ + Σ_{t≠p} αₜ Xₜ.
In this case the p-th term can be replaced: T = u'⊗v'⊗yₚ + Σ_{t≠p} Xₜ⊗(yₜ + αₜyₚ).
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

## LITA odd

As a complement to the even-dimensional construction, schemes for odd `N` are also included in `schemes/`. `scripts/lita_odd.py` constructs rational schemes for odd `9 <= N < 32`, with rank

```text
R_odd(N) = N^3/3 + 7*N^2/2 + 14*N/3 - 11/2.
```

| N | rank R | ω |
|---:|---:|---:|
| 13 | 1379 | 2.81842 |
| 15 | 1977 | 2.80251 |
| 17 | 2723 | 2.79170 |
| 19 | 3633 | 2.78417 |
| 21 | 4723 | 2.77883 |
| 23 | 6009 | 2.77501 |
| 25 | 7507 | 2.77227 |
| 27 | 9233 | 2.77033 |
| 29 | 11203 | 2.76897 |
| 31 | 13433 | 2.76806 |

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
