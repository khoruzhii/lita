import json
from pathlib import Path
import numpy as np

SCHEME_DIR = Path(__file__).resolve().with_name("schemes")

PRIME = 1_000_003
FLOAT_TRIALS = 10
MOD_TRIALS = 10
FLOAT_ATOL = 1e-7
FLOAT_RTOL = 1e-7

def load_meta(npz):
    meta = json.loads(str(npz["metadata_json"].tolist()))
    n1, n2, n3 = map(int, meta.get("tensor", meta.get("n")))
    return {"n1": n1, "n2": n2, "n3": n3, "rank": int(meta["rank"])}

def inv_mod(values):
    values = np.asarray(values, dtype=np.int64) % PRIME
    unique, inverse = np.unique(values, return_inverse=True)
    if np.any(unique == 0):
        raise ValueError(f"denominator divisible by PRIME={PRIME}")
    unique_inv = np.array([pow(int(x), PRIME - 2, PRIME) for x in unique], dtype=np.int64)
    return unique_inv[inverse]

def dense_axis(npz, axis, rows, cols, mod_prime):
    indptr = npz[f"{axis}_indptr"]
    indices = npz[f"{axis}_indices"]
    numerators = npz[f"{axis}_numerators"]
    denominators = npz[f"{axis}_denominators"]

    if indptr.shape != (rows + 1,) or int(indptr[0]) != 0 or int(indptr[-1]) != len(indices):
        raise ValueError(f"bad {axis}_indptr")
    if indices.shape != numerators.shape or indices.shape != denominators.shape:
        raise ValueError(f"bad {axis} CSR lengths")
    if np.any(indices < 0) or np.any(indices >= cols):
        raise ValueError(f"{axis}_indices out of bounds")

    row = np.repeat(np.arange(rows, dtype=np.int64), np.diff(indptr))
    out = np.zeros((rows, cols), dtype=np.int64 if mod_prime else np.float64)
    if mod_prime:
        values = ((numerators.astype(np.int64) % PRIME) * inv_mod(denominators)) % PRIME
    else:
        values = numerators.astype(np.float64) / denominators.astype(np.float64)
    np.add.at(out, (row, indices), values)
    return out % PRIME if mod_prime else out

def load_factors(path, mod_prime=False):
    with np.load(path, allow_pickle=False) as npz:
        meta = load_meta(npz)
        n1, n2, n3, rank = meta["n1"], meta["n2"], meta["n3"], meta["rank"]
        u = dense_axis(npz, "u", rank, n1 * n2, mod_prime)
        v = dense_axis(npz, "v", rank, n2 * n3, mod_prime)
        w = dense_axis(npz, "w", rank, n1 * n3, mod_prime)
    return meta, u, v, w

def check_float(meta, u, v, w):
    n1, n2, n3 = meta["n1"], meta["n2"], meta["n3"]
    for _ in range(FLOAT_TRIALS):
        a = np.random.normal(size=(n1, n2))
        b = np.random.normal(size=(n2, n3))
        gamma = (u @ a.reshape(-1)) * (v @ b.reshape(-1))
        got = np.einsum("r,ro->o", gamma, w, optimize=True).reshape(n1, n3)
        expected = a @ b
        if not np.allclose(got, expected, atol=FLOAT_ATOL, rtol=FLOAT_RTOL):
            err = float(np.max(np.abs(got - expected)))
            raise AssertionError(f"float64 check failed; max_abs_error={err:g}")

def check_mod(meta, u, v, w):
    n1, n2, n3 = meta["n1"], meta["n2"], meta["n3"]
    for _ in range(MOD_TRIALS):
        a = np.random.randint(0, PRIME, size=(n1, n2), dtype=np.int64)
        b = np.random.randint(0, PRIME, size=(n2, n3), dtype=np.int64)
        alpha = (u @ a.reshape(-1)) % PRIME
        beta = (v @ b.reshape(-1)) % PRIME
        got = (((alpha * beta) % PRIME) @ w % PRIME).reshape(n1, n3)
        expected = (a @ b) % PRIME
        if not np.array_equal(got, expected):
            raise AssertionError("mod-prime check failed")

paths = sorted(SCHEME_DIR.glob("*.npz"))
if not paths:
    raise SystemExit(f"no .npz schemes found in {SCHEME_DIR}")

for path in paths:
    meta, u, v, w = load_factors(path, mod_prime=False)
    check_float(meta, u, v, w)
    del u, v, w

    meta, u, v, w = load_factors(path, mod_prime=True)
    check_mod(meta, u, v, w)

    print(f"OK {path.name}")

print(f"All checks passed for {len(paths)} schemes.")
