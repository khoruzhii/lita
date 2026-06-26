# Local Improvements to Trilinear Aggregation

This directory contains explicit rational square matrix multiplication schemes obtained from local improvements to trilinear aggregation. For the dimensions listed below, their ranks improve the corresponding previously reported upper bounds. 

The table lists schemes ⟨N×N×N : R⟩, with ω = log_N R.

| N | rank R | ω |
|---:|---:|---|
| 19 | 4016  | 2.81821 |
| 21 | 5198  | 2.81030 |
| 23 | 6586  | 2.80425 |
| 25 | 8196  | 2.79955 |
| 26 | 8652  | 2.78247 |
| 27 | 10045 | 2.79590 |
| 28 | 10535 | 2.77968 |
| 29 | 12147 | 2.79299 |
| 30 | 12672 | 2.77760 |
| 31 | 14519 | 2.79070 |
| 32 | 15079 | 2.77605 |
| 44 | 36087 | **2.77303** |

For `N=44`, the rank `36087` scheme gives ω = 2.77***303***.  This improves on the 2.77***320*** exponent reported by Schwartz and Zwecher in [arXiv:2508.01748](https://arxiv.org/abs/2508.01748) for the same regime.

## Tensor Convention and Loading

Each file is `schemes/{N}x{N}x{N}_r{rank}.npz` and stores sparse rational factors `U`, `V`, and `W`. They define the matrix multiplication tensor by

```math
T_{ijk} = \sum_{q=1}^{R} U_{qi} V_{qj} W_{qk}
```

and the product is decoded as

```math
C_k = \sum_{i,j} T_{ijk} A_i B_j.
```

Here `i`, `j`, and `k` are row-major flattened coordinates of `A`, `B`, and
`C`.

The following Python snippet reads one `.npz` file and expands the sparse
factors to dense `float64` arrays:

```python
import json
import numpy as np

path = "data/npz/schemes/19x19x19_r4016.npz"

def read_factor(npz, name, rows, cols):
    indptr = npz[f"{name}_indptr"]
    indices = npz[f"{name}_indices"]
    values = npz[f"{name}_numerators"] / npz[f"{name}_denominators"]
    row = np.repeat(np.arange(rows), np.diff(indptr))
    factor = np.zeros((rows, cols), dtype=np.float64)
    np.add.at(factor, (row, indices), values)
    return factor

with np.load(path, allow_pickle=False) as npz:
    meta = json.loads(str(npz["metadata_json"].tolist()))
    N = meta["tensor"][0]
    R = meta["rank"]
    U = read_factor(npz, "u", R, N * N)
    V = read_factor(npz, "v", R, N * N)
    W = read_factor(npz, "w", R, N * N)

# usage example
A = np.random.normal(size=(N, N))
B = np.random.normal(size=(N, N))
C = np.einsum(
    "qi,i,qj,j,qk->k", 
    U, A.reshape(-1), V, B.reshape(-1), W).reshape(N, N
) # C = A @ B
```

For a complete compact loading example, see `verify.py`.

## Maple Generators

The directory `scripts/` contains Maple generators for the two construction families.  The file `KGP2026_even.mpl` implements the even-dimensional construction `N > 18` with

```text
Rₑ(N) = N³/3 + 15N²/4 + 29N/3 + 7,
```

and `KGP2026_odd.mpl` implements the odd-dimensional construction `N > 18`, with

```text
Rₒ(N) = (4N³ + 57N² + 14N − 15)/12 − floor(3(N − 1)/8).
```


## Verification Example

The schemes were verified exactly over `Q`.  The small NumPy script
`verify.py` is included as a simple sanity check for these files: it multiplies random matrices using the stored scheme and compares the result with ordinary matrix multiplication over both `float64` and a fixed prime field.

Run from the repository root:

```bash
python data\npz\verify.py
```

## Citation

If you use these schemes, please cite this repository:

```bibtex
@misc{khoruzhii2026lita,
  author       = {Kirill Khoruzhii and Patrick Gel{\ss} and Sebastian Pokutta},
  title        = {Local Improvements to Trilinear Aggregation},
  year         = {2026},
  url          = {https://github.com/khoruzhii/lita}
}
```

An associated manuscript, *Local Improvements to Trilinear Aggregation*, is in preparation.
