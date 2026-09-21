# Local Improvements to Trilinear Aggregation

This repository contains rational schemes for square matrix multiplication and their construction. These constructions build on Victor Pan’s trilinear aggregation techniques ([1978](https://doi.org/10.1109/SFCS.1978.34), [1982](https://doi.org/10.1016/0898-1221(82)90037-2)). The table lists the even-dimensional `⟨N×N×N : R⟩` schemes in `schemes/` with exponent `ω = log_N R`. Odd-dimensional schemes are listed in [LITA odd](#lita-odd).

| N | rank R | ω |
|---:|---:|---:|
| 14 | 1594 | 2.79418 |
| 16 | 2237 | 2.78184 |
| 18 | 3032 | 2.77368 |
| 20 | 3995 | 2.76820 |
| 22 | 5142 | 2.76450 |
| 24 | 6489 | 2.76202 |
| 26 | 8052 | 2.76041 |
| 28 | 9847 | 2.75941 |
| 30 | 11890 | 2.75887 |
| 32 | 14197 | **2.75866** |

For `N=32`, rank `14197` gives the smallest exponent in this catalogue and improves the `N=44` exponent `2.77320` reported by Schwartz and Zwecher in [arXiv:2508.01748](https://arxiv.org/abs/2508.01748) to `2.75866`.

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
python scripts/lita.py 22 schemes/22x22x22_r5142.npz
```

Its rank is

```text
R_even(N) = N^3/3 + 3*N^2 + 37*N/6 + 5.
```

For this family, `ω(N) = log_N R_even(N)` is minimized over even `N ≥ 8`
at `N=32`, with `R=14197` and `ω ≈ 2.75866`.

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

The main focus has been on even dimensions. Nevertheless, the following
schemes were also found for odd `N` and are included in `schemes/`.

`scripts/lita_odd.py` constructs rational schemes for odd `7 <= N < 32`, with

```text
R_odd(N) = N^3/3 + 15*N^2/4 + 14*N/3 + 13/4.
```

It generates the entries below for `N >= 15`. For `N=13`, the rank-1420
scheme was obtained by applying a 2-reduction to a previously known rank-1421 scheme.

| N | rank R | ω |
|---:|---:|---:|
| 13 | 1420 | 2.82985 |
| 15 | 2042 | 2.81445 |
| 17 | 2804 | 2.80205 |
| 19 | 3732 | 2.79330 |
| 21 | 4842 | 2.78700 |
| 23 | 6150 | 2.78240 |
| 25 | 7672 | 2.77902 |
| 27 | 9424 | 2.77654 |
| 29 | 11422 | 2.77472 |
| 31 | 13682 | 2.77340 |

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
