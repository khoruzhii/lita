// ta.h
//
// Raw Schwartz-Zwecher 2025 trilinear aggregation.
//
// This header materializes scripts/SchwartzZwecher2025.mpl in the same
// direct-output sparse factor format used by lita.h.  The construction is
// rational and is singular at N = 16.
//
// API:
//     U64 r = ta_rank(N);
//     SchemeQ q = ta_q(N);
//     SchemeP p = ta_p(N, prime);
//
// ta_rank()
//     returns N^3/3 + 15*N^2/4 + 61*N/6 + 8.
//
// ta_q()
//     returns the rational direct-output scheme for even N, N != 16.
//
// ta_p()
//     returns the same scheme reduced modulo prime.

#pragma once

#include "lita.h"

#include <stdexcept>

inline U64 ta_rank(U32 N) {
    const U64 n = N;
    return (4 * n * n * n + 45 * n * n + 122 * n + 96) / 12;
}

template<class Field>
inline LitaLocal7<Field> ta_local7(int d, const Field& field) {
    using S = typename Field::Scalar;
    auto F = [&](I64 x) -> S { return field.from_i64(x); };
    auto Q = [&](I64 n, I64 den) -> S { return field.frac(n, den); };
    auto neg = [&](S x) -> S { return field.neg(x); };
    auto mul = [&](S a, S b) -> S { return field.mul(a, b); };

    const S invg = Q(d, d - 9);
    const S one_plus_gamma = Q(2 * I64(d) - 9, d);
    const S one_minus_gamma = Q(9, d);
    const S gamma_minus_one = Q(-9, d);
    const S d_over_gamma = mul(F(d), invg);
    const S d_over_gamma2 = mul(d_over_gamma, invg);

    LitaLocal7<Field> out{};
    auto set = [&](int r, std::array<S, 4> u, std::array<S, 4> v, std::array<S, 4> w) {
        out[static_cast<size_t>(r)][0] = u;
        out[static_cast<size_t>(r)][1] = v;
        out[static_cast<size_t>(r)][2] = w;
    };

    set(0,
        {F(-1), F(1), F(1), F(0)},
        {F(1), F(1), F(1), F(0)},
        {
            F(1 - d),
            Q(I64(d) * d - d + 9, d - 9),
            Q(-(I64(d) * d - d + 9), d - 9),
            mul(one_minus_gamma, d_over_gamma),
        }
    );
    set(1,
        {F(0), F(0), F(1), F(0)},
        {
            gamma_minus_one,
            F(-1),
            mul(mul(gamma_minus_one, one_plus_gamma), invg),
            neg(one_plus_gamma),
        },
        {d_over_gamma, d_over_gamma, d_over_gamma2, d_over_gamma}
    );
    set(2,
        {Q(d - 9, d), F(0), F(1), F(0)},
        {F(1), one_plus_gamma, invg, one_plus_gamma},
        {d_over_gamma, F(0), d_over_gamma2, F(0)}
    );
    set(3,
        {neg(one_plus_gamma), F(1), F(0), F(0)},
        {F(1), F(1), invg, F(1)},
        {mul(one_plus_gamma, d_over_gamma), F(0), d_over_gamma2, F(0)}
    );
    set(4,
        {F(-1), F(1), neg(invg), F(1)},
        {F(0), F(0), invg, one_plus_gamma},
        {F(0), d_over_gamma, F(0), neg(mul(gamma_minus_one, d_over_gamma))}
    );
    set(5,
        {F(1), F(-1), F(0), F(0)},
        {F(1), F(1), mul(one_plus_gamma, invg), one_plus_gamma},
        {d_over_gamma, d_over_gamma, mul(one_minus_gamma, d_over_gamma2), mul(one_minus_gamma, d_over_gamma)}
    );
    set(6,
        {F(0), F(0), neg(mul(one_plus_gamma, invg)), F(1)},
        {F(0), F(0), mul(one_minus_gamma, invg), F(1)},
        {F(0), neg(mul(one_plus_gamma, d_over_gamma)), F(0), neg(d_over_gamma)}
    );
    return out;
}

template<class Field>
inline LitaBuilder<Field> ta_build(U32 N, const Field& field) {
    if (N % 2 != 0) throw std::runtime_error("TA is implemented for even N");
    if (N == 16) throw std::runtime_error("TA is singular for N = 16");

    using S = typename Field::Scalar;
    const int n = static_cast<int>(N / 2);
    const int d = n + 1;
    const int D = d - 1;
    const int M = static_cast<int>(N) + 2;
    auto foo = [](int i) { return i; };
    auto bar = [&](int i) { return (i + d) % M; };
    auto F = [&](I64 x) -> S { return field.from_i64(x); };

    LitaBuilder<Field> builder;
    builder.N = N;
    builder.U.reserve(static_cast<size_t>(ta_rank(N)));
    builder.V.reserve(static_cast<size_t>(ta_rank(N)));
    builder.W.reserve(static_cast<size_t>(ta_rank(N)));

    const LitaPhiCache<Field> cache = lita_make_phi_cache(N, field);
    const LitaLocal7<Field> local7 = ta_local7(d, field);

    auto emit = [&](LitaRow<Field> u, LitaRow<Field> v, LitaRow<Field> w) {
        builder.add(std::move(u), std::move(v), std::move(w));
    };

    for (int i = 0; i <= D; ++i) {
        for (int r = 0; r < 7; ++r) {
            emit(
                lita_form4(cache, M, local7[r][0], i, d, field),
                lita_form4(cache, M, local7[r][1], i, d, field),
                lita_form4(cache, M, local7[r][2], i, d, field)
            );
        }
    }

    auto add_agg1 = [&](int i, int j, int k, bool barred) {
        auto f = [&](int x) { return barred ? bar(x) : foo(x); };
        emit(
            lita_star_form<Field>(cache, M, {{F(1), f(i), f(j)}, {F(1), f(j), f(k)}, {F(1), f(k), f(i)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), f(j), f(k)}, {F(1), f(k), f(i)}, {F(1), f(i), f(j)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), f(k), f(i)}, {F(1), f(i), f(j)}, {F(1), f(j), f(k)}}, field)
        );
    };

    for (int k = 0; k <= D; ++k)
        for (int j = 0; j < k; ++j)
            for (int i = 0; i <= j; ++i)
                add_agg1(i, j, k, false);
    for (int k = 0; k <= D; ++k)
        for (int j = 0; j < k; ++j)
            for (int i = 0; i <= j; ++i)
                add_agg1(i, j, k, true);
    for (int i = 0; i <= D; ++i)
        for (int j = 0; j <= i; ++j)
            for (int k = 0; k < j; ++k)
                add_agg1(i, j, k, false);
    for (int i = 0; i <= D; ++i)
        for (int j = 0; j <= i; ++j)
            for (int k = 0; k < j; ++k)
                add_agg1(i, j, k, true);

    auto add_agg2a = [&](int i, int j, int k) {
        emit(
            lita_star_form<Field>(cache, M, {{F(1), bar(j), foo(k)}, {F(1), foo(k), bar(i)}, {F(-1), foo(i), foo(j)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), foo(j), bar(k)}, {F(1), foo(k), foo(i)}, {F(1), bar(i), foo(j)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), foo(i), bar(j)}, {F(1), foo(j), foo(k)}, {F(-1), bar(k), foo(i)}}, field)
        );
    };
    auto add_agg2b = [&](int i, int j, int k) {
        emit(
            lita_star_form<Field>(cache, M, {{F(1), foo(j), bar(k)}, {F(1), bar(k), foo(i)}, {F(-1), bar(i), bar(j)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), bar(j), foo(k)}, {F(1), bar(k), bar(i)}, {F(1), foo(i), bar(j)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), bar(i), foo(j)}, {F(1), bar(j), bar(k)}, {F(-1), foo(k), bar(i)}}, field)
        );
    };

    for (int k = 0; k <= D; ++k) {
        for (int j = 0; j <= D; ++j) {
            for (int i = 0; i <= D; ++i) {
                if (!(i == j && j == k)) add_agg2a(i, j, k);
            }
        }
    }
    for (int k = 0; k <= D; ++k)
        for (int j = 0; j <= D; ++j)
            for (int i = 0; i <= D; ++i)
                add_agg2b(i, j, k);

    for (int i = 0; i <= D; ++i) {
        for (int j = 0; j <= D; ++j) {
            if (i == j) continue;

            const int fi = foo(i), fj = foo(j), bi = bar(i), bj = bar(j);
            const std::array<std::pair<int, int>, 4> coords = {{
                {fi, fj}, {bi, fj}, {fi, bj}, {bi, bj}
            }};
            const std::array<int, 4> wscale = {{
                -d * LITA_CSIGN[0], -d * LITA_CSIGN[1],
                -d * LITA_CSIGN[2], -d * LITA_CSIGN[3],
            }};

            for (int t = 0; t < 7; ++t) {
                emit(
                    lita_table_form<Field>(cache, M, coords, LITA_USTR[static_cast<size_t>(t)], LITA_ONE4, field),
                    lita_table_form<Field>(cache, M, coords, LITA_VSTR[static_cast<size_t>(t)], LITA_ONE4, field),
                    lita_table_form<Field>(cache, M, coords, LITA_WSTR[static_cast<size_t>(t)], wscale, field)
                );
            }
        }
    }

    if (builder.U.size() != ta_rank(N)) {
        throw std::runtime_error("internal TA rank mismatch");
    }
    return builder;
}

inline SchemeP ta_p(U32 N, U64 prime) {
    const LitaPField field{prime};
    const LitaBuilder<LitaPField> rows = ta_build(N, field);
    SchemeP out;
    out.U = lita_to_sparse_p(rows.U, N * N);
    out.V = lita_to_sparse_p(rows.V, N * N);
    out.W = lita_to_sparse_p(lita_transpose_square_rows(rows.W, N, field), N * N);
    return out;
}

inline SchemeQ ta_q(U32 N) {
    const LitaQField field{};
    const LitaBuilder<LitaQField> rows = ta_build(N, field);
    SchemeQ out;
    out.U = lita_to_sparse_q(rows.U, N * N);
    out.V = lita_to_sparse_q(rows.V, N * N);
    out.W = lita_to_sparse_q(lita_transpose_square_rows(rows.W, N, field), N * N);
    return out;
}
