#include "lita.h"

#include <exception>
#include <iostream>
#include <random>
#include <vector>

static U64 mod_pow_test(U64 a, U64 e, U64 p) {
    U64 out = 1;
    while (e != 0) {
        if ((e & 1) != 0) out = out * a % p;
        a = a * a % p;
        e >>= 1;
    }
    return out;
}

static U64 rat_mod(Rat x, U64 p) {
    I64 n = x.num % static_cast<I64>(p);
    I64 d = x.den % static_cast<I64>(p);
    if (n < 0) n += static_cast<I64>(p);
    if (d < 0) d += static_cast<I64>(p);
    return static_cast<U64>(n) * mod_pow_test(static_cast<U64>(d), p - 2, p) % p;
}

static U64 dot_sparse(const SparseMatrixP& m, U32 row, const std::vector<U64>& x, U64 p) {
    U64 out = 0;
    for (U64 e = m.ptr[row]; e < m.ptr[row + 1]; ++e) {
        out = (out + m.val[e] * x[m.col[e]]) % p;
    }
    return out;
}

static bool same_mod(const SparseMatrixQ& q, const SparseMatrixP& pmat, U64 p) {
    if (q.rows != pmat.rows || q.cols != pmat.cols || q.ptr != pmat.ptr || q.col != pmat.col) return false;
    if (q.val.size() != pmat.val.size()) return false;
    for (size_t i = 0; i < q.val.size(); ++i) {
        if (rat_mod(q.val[i], p) != pmat.val[i]) return false;
    }
    return true;
}

static bool verify_lita_q(U32 N, U64 p) {
    const SchemeQ q = lita_q(N);
    const SchemeP pm = lita_p(N, p);
    if (q.U.rows != lita_rank(N) || q.V.rows != lita_rank(N) || q.W.rows != lita_rank(N)) {
        std::cerr << "Q rank mismatch for N=" << N << "\n";
        return false;
    }
    if (!same_mod(q.U, pm.U, p) || !same_mod(q.V, pm.V, p) || !same_mod(q.W, pm.W, p)) {
        std::cerr << "Q-to-mod mismatch for N=" << N << " p=" << p << "\n";
        return false;
    }
    return true;
}

static bool verify_lita(U32 N, U64 p, int trials) {
    const SchemeP scheme = lita_p(N, p);
    if (scheme.U.rows != lita_rank(N) || scheme.V.rows != lita_rank(N) || scheme.W.rows != lita_rank(N)) {
        std::cerr << "rank mismatch for N=" << N << "\n";
        return false;
    }

    std::mt19937_64 rng(20260707ull + N + p);
    for (int trial = 0; trial < trials; ++trial) {
        std::vector<U64> A(static_cast<size_t>(N) * N);
        std::vector<U64> B(static_cast<size_t>(N) * N);
        std::vector<U64> C(static_cast<size_t>(N) * N, 0);
        std::vector<U64> want(static_cast<size_t>(N) * N, 0);

        for (U64& x : A) x = rng() % p;
        for (U64& x : B) x = rng() % p;

        for (U32 q = 0; q < scheme.U.rows; ++q) {
            const U64 a = dot_sparse(scheme.U, q, A, p);
            const U64 b = dot_sparse(scheme.V, q, B, p);
            const U64 ab = a * b % p;
            if (ab == 0) continue;
            for (U64 e = scheme.W.ptr[q]; e < scheme.W.ptr[q + 1]; ++e) {
                const U32 k = scheme.W.col[e];
                C[k] = (C[k] + ab * scheme.W.val[e]) % p;
            }
        }

        for (U32 i = 0; i < N; ++i) {
            for (U32 k = 0; k < N; ++k) {
                U64 s = 0;
                for (U32 j = 0; j < N; ++j) {
                    s = (s + A[static_cast<size_t>(i) * N + j] * B[static_cast<size_t>(j) * N + k]) % p;
                }
                want[static_cast<size_t>(i) * N + k] = s;
            }
        }

        if (C != want) {
            std::cerr << "multiplication mismatch for N=" << N << " p=" << p
                      << " trial=" << trial << "\n";
            return false;
        }
    }

    return true;
}

int main() {
    try {
        const U64 primes[] = {1000003, 1000033, 1000037};
        const U32 dimensions[] = {18, 20, 22, 24, 26, 28, 30, 32, 44};
        for (U32 N : dimensions) {
            if (!verify_lita_q(N, primes[0])) return 1;
            for (U64 p : primes) {
                if (!verify_lita(N, p, 2)) return 1;
            }
            std::cout << "N=" << N << " rank=" << lita_rank(N) << " ok\n";
        }
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "error: " << e.what() << "\n";
        return 1;
    }
}
