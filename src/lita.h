// lita.h
//
// Even-dimensional Local Improvements to Trilinear Aggregation.
//
// This header materializes the direct pair-reduced LITA construction matching
// scripts/KGP2026.mpl.  It is an exact rational scheme over Q, with a modular
// companion constructor for finite-field checks.
//
// API:
//     U64 r = lita_rank(N);
//     SchemeQ q = lita_q(N);
//     SchemeP p = lita_p(N, prime);
//
// lita_rank()
//     returns N^3/3 + 15*N^2/4 + 55*N/6 + 7.
//
// lita_q()
//     returns the rational direct-output scheme for even N >= 18.
//
// lita_p()
//     returns the same scheme reduced modulo prime.

#pragma once

#include "scheme.h"

#include <algorithm>
#include <array>
#include <initializer_list>
#include <numeric>
#include <stdexcept>
#include <utility>
#include <vector>

inline U64 lita_rank(U32 N) {
    const U64 n = N;
    return (4 * n * n * n + 45 * n * n + 110 * n + 84) / 12;
}

inline Rat rat(I64 num, I64 den = 1) {
    if (den == 0) throw std::runtime_error("zero rational denominator");
    if (den < 0) {
        num = -num;
        den = -den;
    }
    const I64 g = std::gcd(num < 0 ? -num : num, den);
    return {num / g, den / g};
}

inline Rat operator-(Rat x) {
    return {-x.num, x.den};
}

inline Rat operator+(Rat a, Rat b) {
    const I64 g = std::gcd(a.den, b.den);
    const I64 ad = a.den / g;
    const I64 bd = b.den / g;
    return rat(a.num * bd + b.num * ad, ad * b.den);
}

inline Rat operator-(Rat a, Rat b) {
    return a + (-b);
}

inline Rat operator*(Rat a, Rat b) {
    I64 an = a.num;
    I64 ad = a.den;
    I64 bn = b.num;
    I64 bd = b.den;

    I64 g = std::gcd(an < 0 ? -an : an, bd);
    an /= g;
    bd /= g;
    g = std::gcd(bn < 0 ? -bn : bn, ad);
    bn /= g;
    ad /= g;
    return rat(an * bn, ad * bd);
}

inline bool operator==(Rat a, Rat b) {
    return a.num == b.num && a.den == b.den;
}

inline bool operator!=(Rat a, Rat b) {
    return !(a == b);
}

struct LitaPField {
    using Scalar = U64;

    U64 prime = 1000003;

    U64 mod_i64(I64 x) const {
        I64 r = x % static_cast<I64>(prime);
        if (r < 0) r += static_cast<I64>(prime);
        return static_cast<U64>(r);
    }

    U64 add(U64 a, U64 b) const {
        const U64 s = a + b;
        return (s >= prime || s < a) ? (s - prime) : s;
    }

    U64 neg(U64 a) const {
        return a == 0 ? 0 : prime - a;
    }

    U64 mul(U64 a, U64 b) const {
        return (a * b) % prime;
    }

    U64 pow(U64 a, U64 e) const {
        U64 out = 1;
        U64 base = a % prime;
        while (e != 0) {
            if ((e & 1) != 0) out = mul(out, base);
            base = mul(base, base);
            e >>= 1;
        }
        return out;
    }

    U64 inv(U64 a) const {
        if (a == 0) throw std::runtime_error("zero inverse modulo prime");
        return pow(a, prime - 2);
    }

    U64 from_i64(I64 x) const {
        return mod_i64(x);
    }

    U64 frac(I64 num, I64 den) const {
        return mul(mod_i64(num), inv(mod_i64(den)));
    }

    bool is_zero(U64 x) const {
        return x == 0;
    }
};

struct LitaQField {
    using Scalar = Rat;

    Rat add(Rat a, Rat b) const { return a + b; }
    Rat neg(Rat a) const { return -a; }
    Rat mul(Rat a, Rat b) const { return a * b; }
    Rat from_i64(I64 x) const { return rat(x); }
    Rat frac(I64 num, I64 den) const { return rat(num, den); }
    bool is_zero(Rat x) const { return x.num == 0; }
};

template<class Field>
struct LitaEntry {
    U32 index = 0;
    typename Field::Scalar value{};
};

template<class Field>
using LitaRow = std::vector<LitaEntry<Field>>;

template<class Field>
struct LitaTerm {
    typename Field::Scalar coeff{};
    int row = 0;
    int col = 0;
};

template<class Field>
inline void lita_normalize_row(LitaRow<Field>& row, const Field& field) {
    row.erase(
        std::remove_if(row.begin(), row.end(), [&](const auto& entry) { return field.is_zero(entry.value); }),
        row.end()
    );
    std::sort(row.begin(), row.end(), [](const auto& a, const auto& b) {
        return a.index < b.index;
    });

    size_t out = 0;
    for (const auto& entry : row) {
        if (out != 0 && row[out - 1].index == entry.index) {
            row[out - 1].value = field.add(row[out - 1].value, entry.value);
            if (field.is_zero(row[out - 1].value)) --out;
        } else {
            row[out++] = entry;
        }
    }
    row.resize(out);
}

template<class Field>
inline LitaRow<Field> lita_add_scaled(
    LitaRow<Field> row,
    const LitaRow<Field>& other,
    typename Field::Scalar scale,
    const Field& field
) {
    if (field.is_zero(scale)) return row;
    row.reserve(row.size() + other.size());
    for (const auto& entry : other) {
        row.push_back({entry.index, field.mul(entry.value, scale)});
    }
    lita_normalize_row(row, field);
    return row;
}

template<class Field>
struct LitaBuilder {
    U32 N = 0;
    std::vector<LitaRow<Field>> U;
    std::vector<LitaRow<Field>> V;
    std::vector<LitaRow<Field>> W;

    void add(LitaRow<Field> u, LitaRow<Field> v, LitaRow<Field> w) {
        U.push_back(std::move(u));
        V.push_back(std::move(v));
        W.push_back(std::move(w));
    }
};

enum class LitaFamily {
    Agg1F,
    Agg1B,
    Agg1Fsym,
    Agg1Bsym,
    Agg2a,
    Agg2b,
    Local7,
    Off,
};

struct LitaLabel {
    LitaFamily family = LitaFamily::Agg1F;
    int a = 0;
    int b = 0;
    int t = 0;
};

struct LitaEndpoints {
    std::vector<std::pair<int, int>> agg2a;
    std::vector<std::pair<int, int>> agg2b;
};

template<class Field>
using LitaPhiCache = std::vector<LitaRow<Field>>;

template<class Field>
using LitaLocal7 = std::array<std::array<std::array<typename Field::Scalar, 4>, 3>, 7>;

template<class Field>
struct LitaRemovedCfactors {
    std::vector<LitaRow<Field>> R;
    LitaRow<Field> T;
};

inline constexpr std::array<std::array<int, 4>, 7> LITA_USTR = {{
    {{1, 0, 0, 1}}, {{0, 0, 0, 1}}, {{0, 1, 0, 1}}, {{1, -1, 0, 0}},
    {{1, 0, 1, 0}}, {{1, 0, 0, 0}}, {{0, 0, 1, -1}},
}};

inline constexpr std::array<std::array<int, 4>, 7> LITA_VSTR = {{
    {{1, 0, 0, 1}}, {{1, 0, 1, 0}}, {{0, 0, -1, 1}}, {{0, 0, 0, 1}},
    {{1, -1, 0, 0}}, {{0, -1, 0, -1}}, {{1, 0, 0, 0}},
}};

inline constexpr std::array<std::array<int, 4>, 7> LITA_WSTR = {{
    {{1, 0, 0, 1}}, {{-1, 1, 0, 0}}, {{-1, 0, 0, 0}}, {{-1, 0, -1, 0}},
    {{0, 0, 0, -1}}, {{0, 0, -1, 1}}, {{0, 1, 0, 1}},
}};

inline constexpr std::array<int, 4> LITA_CSIGN = {{1, -1, -1, 1}};
inline constexpr std::array<int, 4> LITA_ONE4 = {{1, 1, 1, 1}};

inline int lita_key(int i, int j, int k, int d) {
    return (k * d + j) * d + i;
}

inline int lita_vertex(int s, int d) {
    const int t = s % d;
    return t == 0 ? d - 1 : t - 1;
}

inline std::pair<int, int> lita_edge(int r, int d, int offset) {
    const int s = (r + offset) % (d * d);
    return {lita_vertex(s / d, d), lita_vertex(s % d, d)};
}

inline LitaEndpoints lita_agg2_endpoints(int d) {
    LitaEndpoints out;
    out.agg2a.assign(static_cast<size_t>(d * d * d), {-1, -1});
    out.agg2b.assign(static_cast<size_t>(d * d * d), {-1, -1});

    for (int family = 0; family < 2; ++family) {
        std::vector<std::pair<int, int>>& table = family == 0 ? out.agg2a : out.agg2b;
        int offset = 1;
        if (family == 1) {
            if (d % 2 == 0) {
                offset = 0;
            } else {
                const int h = (d - 1) / 2;
                offset = h * (d + 1) + 1;
            }
        }

        int r = 0;
        int used = 0;
        std::pair<int, int> cur = lita_edge(r, d, offset);
        for (int k = 0; k < d; ++k) {
            for (int j = 0; j < d; ++j) {
                for (int i = 0; i < d; ++i) {
                    const bool zero = cur.first == cur.second;
                    if (i == j && j == k) {
                        bool consume = false;
                        if (family == 0) consume = zero;
                        else if (d % 2 == 0) consume = zero && (i % 2 == 1);
                        else consume = zero && i != 0;
                        if (consume) {
                            ++r;
                            used = 0;
                            cur = lita_edge(r, d, offset);
                        }
                    } else {
                        table[static_cast<size_t>(lita_key(i, j, k, d))] = cur;
                        ++used;
                        if (used == 2) {
                            ++r;
                            used = 0;
                            cur = lita_edge(r, d, offset);
                        }
                    }
                }
            }
        }
    }

    return out;
}

template<class Field>
inline LitaRow<Field> lita_phi_entry(U32 N, int r, int c, const Field& field) {
    const int n = static_cast<int>(N / 2);
    const int d = n + 1;
    const int br = r / d;
    const int lr = r % d;
    const int bc = c / d;
    const int lc = c % d;

    std::vector<std::pair<int, typename Field::Scalar>> prow;
    if (lr < n) {
        prow.push_back({br * n + lr, field.from_i64(1)});
    } else {
        for (int t = 0; t < n; ++t) prow.push_back({br * n + t, field.from_i64(-1)});
    }

    std::vector<std::pair<int, typename Field::Scalar>> qcol;
    for (int t = 0; t < n; ++t) {
        typename Field::Scalar coeff = field.frac(-1, d);
        if (lc < n && t == lc) coeff = field.add(coeff, field.from_i64(1));
        if (!field.is_zero(coeff)) qcol.push_back({bc * n + t, coeff});
    }

    LitaRow<Field> out;
    out.reserve(prow.size() * qcol.size());
    for (const auto& [i, ci] : prow) {
        const U32 base = static_cast<U32>(i * static_cast<int>(N));
        for (const auto& [j, cj] : qcol) {
            out.push_back({base + static_cast<U32>(j), field.mul(ci, cj)});
        }
    }
    lita_normalize_row(out, field);
    return out;
}

template<class Field>
inline LitaPhiCache<Field> lita_make_phi_cache(U32 N, const Field& field) {
    const int M = static_cast<int>(N) + 2;
    LitaPhiCache<Field> cache;
    cache.reserve(static_cast<size_t>(M * M));
    for (int r = 0; r < M; ++r) {
        for (int c = 0; c < M; ++c) {
            cache.push_back(lita_phi_entry(N, r, c, field));
        }
    }
    return cache;
}

template<class Field>
inline const LitaRow<Field>& lita_phi(const LitaPhiCache<Field>& cache, int M, int r, int c) {
    return cache[static_cast<size_t>(r * M + c)];
}

template<class Field>
inline LitaRow<Field> lita_star_form(
    const LitaPhiCache<Field>& cache,
    int M,
    std::initializer_list<LitaTerm<Field>> entries,
    const Field& field
) {
    LitaRow<Field> out;
    for (const auto& entry : entries) {
        out = lita_add_scaled(std::move(out), lita_phi(cache, M, entry.row, entry.col), entry.coeff, field);
    }
    return out;
}

template<class Field>
inline LitaRow<Field> lita_table_form(
    const LitaPhiCache<Field>& cache,
    int M,
    const std::array<std::pair<int, int>, 4>& coords,
    const std::array<int, 4>& coeffs,
    const std::array<int, 4>& scale,
    const Field& field
) {
    LitaRow<Field> out;
    for (int z = 0; z < 4; ++z) {
        const int coeff = coeffs[static_cast<size_t>(z)] * scale[static_cast<size_t>(z)];
        if (coeff != 0) {
            out = lita_add_scaled(
                std::move(out),
                lita_phi(cache, M, coords[static_cast<size_t>(z)].first, coords[static_cast<size_t>(z)].second),
                field.from_i64(coeff),
                field
            );
        }
    }
    return out;
}

template<class Field>
inline LitaLocal7<Field> lita_local7(int d, const Field& field) {
    using S = typename Field::Scalar;
    auto F = [&](I64 x) -> S { return field.from_i64(x); };
    auto Q = [&](I64 n, I64 den) -> S { return field.frac(n, den); };

    std::array<S, 4> e = {F(-1), F(0), F(0), F(1)};
    std::array<S, 4> u = {
        Q(I64(d) * (8 - d), d - 6),
        Q(-2 * I64(d), d - 6),
        Q(I64(d) * (d - 2), d - 6),
        F(0),
    };
    std::array<S, 4> v = {F(0), F(-1), F(1), F(0)};
    std::array<S, 4> w = {Q(1, 2), F(0), Q(1, 2), F(0)};
    const S A = Q(d, d - 6);
    const S B = Q(-(d - 6), d);
    const S C = Q(d - 3, 2 * d);
    const S Ap = Q(I64(d) * (d - 7), d - 6);
    const S Bp = Q(3, 2 * d);
    const S Cp = Q(d - 6, d);
    const S lam = Q(I64(d) * d - 11 * I64(d) + 27, d);

    std::array<S, 4> a, b, c, ap, bp, cp;
    for (int t = 0; t < 4; ++t) {
        a[t] = field.add(u[t], field.mul(A, e[t]));
        b[t] = field.add(v[t], field.mul(B, e[t]));
        c[t] = field.add(w[t], field.mul(C, e[t]));
        ap[t] = field.add(field.neg(u[t]), field.mul(Ap, e[t]));
        bp[t] = field.add(w[t], field.mul(Bp, e[t]));
        cp[t] = field.add(v[t], field.mul(Cp, e[t]));
    }

    const std::array<I64, 4> D0 = {1, -1, -1, 1};
    const std::array<I64, 4> D2 = {1, 1, -1, -1};
    std::array<std::array<std::array<S, 4>, 3>, 7> raw = {{
        {{{field.mul(lam, e[0]), field.mul(lam, e[1]), field.mul(lam, e[2]), field.mul(lam, e[3])}, e, e}},
        {{a, b, c}},
        {{b, c, a}},
        {{c, a, b}},
        {{ap, bp, cp}},
        {{bp, cp, ap}},
        {{cp, ap, bp}},
    }};

    LitaLocal7<Field> out{};
    for (int r = 0; r < 7; ++r) {
        out[r][0] = raw[r][0];
        for (int t = 0; t < 4; ++t) {
            out[r][1][t] = field.mul(F(D0[t]), raw[r][1][t]);
            out[r][2][t] = field.mul(F(D2[t]), raw[r][2][t]);
        }
    }
    return out;
}

template<class Field>
inline LitaRow<Field> lita_form4(
    const LitaPhiCache<Field>& cache,
    int M,
    const std::array<typename Field::Scalar, 4>& coeff,
    int i,
    int d,
    const Field& field
) {
    const int foo = i;
    const int bar = (i + d) % M;
    return lita_star_form<Field>(
        cache,
        M,
        {
            {coeff[0], foo, foo},
            {coeff[1], bar, foo},
            {coeff[2], foo, bar},
            {coeff[3], bar, bar},
        },
        field
    );
}

inline U64 lita_base_rank(U32 N) {
    const U64 n = N;
    return (4 * n * n * n + 45 * n * n + 116 * n + 84) / 12;
}

inline bool lita_is_removed_label(const LitaLabel& label, int d) {
    return label.family == LitaFamily::Off && label.b == d - 1 && label.t == 6;
}

template<class Field>
inline LitaRemovedCfactors<Field> lita_removed_cfactors(
    const LitaPhiCache<Field>& cache,
    int M,
    const LitaLocal7<Field>& coeffs,
    int d,
    const Field& field
) {
    LitaRemovedCfactors<Field> out;
    const int D = d - 1;
    out.R.reserve(static_cast<size_t>(D));
    for (int m = 0; m < D; ++m) {
        // Matches LITA_REMOVED_CFACTORS: the removed positional rows are local7(m+1, 7).
        LitaRow<Field> r = lita_form4(cache, M, coeffs[6][2], m + 1, d, field);
        out.T = lita_add_scaled(std::move(out.T), r, field.from_i64(1), field);
        out.R.push_back(std::move(r));
    }
    return out;
}

template<class Field>
inline LitaRow<Field> lita_add_removed(
    LitaRow<Field> row,
    const LitaRemovedCfactors<Field>& rt,
    int a,
    int d,
    typename Field::Scalar scale,
    const Field& field
) {
    if (a == d - 1) {
        return lita_add_scaled(std::move(row), rt.T, field.neg(scale), field);
    }
    return lita_add_scaled(std::move(row), rt.R[static_cast<size_t>(a)], scale, field);
}

template<class Field>
inline LitaRow<Field> lita_cert_cfactor(
    const LitaLabel& label,
    LitaRow<Field> c0,
    const LitaRemovedCfactors<Field>& rt,
    int d,
    const LitaEndpoints& endpoints,
    const Field& field
) {
    using S = typename Field::Scalar;
    const int D = d - 1;
    const I64 L = 2 * I64(d);
    auto F = [&](I64 x) -> S { return field.from_i64(x); };
    auto add = [&](LitaRow<Field> row, const LitaRow<Field>& other, S scale) {
        return lita_add_scaled(std::move(row), other, scale, field);
    };
    auto add_i64 = [&](LitaRow<Field> row, const LitaRow<Field>& other, I64 scale) {
        return lita_add_scaled(std::move(row), other, F(scale), field);
    };

    if (label.family == LitaFamily::Agg1F || label.family == LitaFamily::Agg1B ||
        label.family == LitaFamily::Agg1Fsym || label.family == LitaFamily::Agg1Bsym) {
        return c0;
    }

    if (label.family == LitaFamily::Agg2a || label.family == LitaFamily::Agg2b) {
        const std::vector<std::pair<int, int>>& table =
            label.family == LitaFamily::Agg2a ? endpoints.agg2a : endpoints.agg2b;
        const std::pair<int, int> ep = table[static_cast<size_t>(lita_key(label.a, label.b, label.t, d))];
        c0 = lita_add_removed(std::move(c0), rt, ep.first, d, F(2), field);
        c0 = lita_add_removed(std::move(c0), rt, ep.second, d, F(-2), field);
        return c0;
    }

    if (label.family == LitaFamily::Local7) {
        const int a = label.a;
        const int r = label.b;
        if (a == D) {
            const I64 v0[7] = {-2, 1, 2, 1, 2, -1, -1};
            const I64 v1[7] = {-1, 0, 1, 1, 1, 0, -1};
            c0 = add_i64(std::move(c0), rt.R[0], L * (v0[r] - v1[r]));
            c0 = add_i64(std::move(c0), rt.T, L * v1[r]);
            return c0;
        }
        const I64 v0[7] = {2, -1, -2, -1, -2, 1, 1};
        const I64 v1[7] = {1, -1, -1, 0, -1, 1, 0};
        c0 = add_i64(std::move(c0), rt.R[static_cast<size_t>(a)], L * (v0[r] - v1[r]));
        c0 = add_i64(std::move(c0), rt.T, L * v1[r]);
        return c0;
    }

    if (label.family == LitaFamily::Off) {
        const int a = label.a;
        const int b = label.b;
        const int t = label.t;
        const S S2 = field.frac(2 * I64(d) * d, d - 6);
        if (a == D && b == D - 1) {
            const std::array<S, 7> vec = {
                F(0), F(1), S2, F(4), F(4), field.neg(S2), F(1),
            };
            return add(std::move(c0), rt.T, vec[static_cast<size_t>(t)]);
        }
        if (a <= D - 1 && b == D) {
            const std::array<S, 7> vec = {
                F(0), F(-1), field.neg(S2), F(-4), F(-4), S2, F(0),
            };
            return add(std::move(c0), rt.R[static_cast<size_t>(a)], vec[static_cast<size_t>(t)]);
        }
        if (a == D && 0 <= b && b <= D - 2) {
            const I64 v0[7] = {-2, 1, 2, 1, 2, -1, -1};
            const I64 v1[7] = {-1, 0, 1, 1, 1, 0, -1};
            const size_t r = static_cast<size_t>(b + 1);
            c0 = add_i64(std::move(c0), rt.R[r], L * (v0[t] - v1[t]));
            c0 = add_i64(std::move(c0), rt.T, L * v1[t]);
            return c0;
        }
        if (0 <= b && b <= D - 1 && 0 <= a && a <= D - 1 && a != b) {
            const I64 vb[7] = {-1, 1, 1, 0, 1, -1, 0};
            const I64 va[7] = {1, 0, -1, -1, -1, 0, 1};
            c0 = add_i64(std::move(c0), rt.R[static_cast<size_t>(b)], L * vb[t]);
            c0 = add_i64(std::move(c0), rt.R[static_cast<size_t>(a)], L * va[t]);
            return c0;
        }
    }

    return c0;
}

inline std::vector<LitaLabel> lita_certificate_labels(int d) {
    const int D = d - 1;
    std::vector<LitaLabel> labels;
    labels.reserve(static_cast<size_t>(
        4 * (d * (d - 1) * (d + 1) / 6) + 2 * (d * d * d - d) + 7 * d * d
    ));

    for (int k = 0; k <= D; ++k)
        for (int j = 0; j < k; ++j)
            for (int i = 0; i <= j; ++i)
                labels.push_back({LitaFamily::Agg1F, i, j, k});
    for (int k = 0; k <= D; ++k)
        for (int j = 0; j < k; ++j)
            for (int i = 0; i <= j; ++i)
                labels.push_back({LitaFamily::Agg1B, i, j, k});
    for (int i = 0; i <= D; ++i)
        for (int j = 0; j <= i; ++j)
            for (int k = 0; k < j; ++k)
                labels.push_back({LitaFamily::Agg1Fsym, i, j, k});
    for (int i = 0; i <= D; ++i)
        for (int j = 0; j <= i; ++j)
            for (int k = 0; k < j; ++k)
                labels.push_back({LitaFamily::Agg1Bsym, i, j, k});
    for (int k = 0; k <= D; ++k)
        for (int j = 0; j <= D; ++j)
            for (int i = 0; i <= D; ++i)
                if (!(i == j && j == k))
                    labels.push_back({LitaFamily::Agg2a, i, j, k});
    for (int k = 0; k <= D; ++k)
        for (int j = 0; j <= D; ++j)
            for (int i = 0; i <= D; ++i)
                if (!(i == j && j == k))
                    labels.push_back({LitaFamily::Agg2b, i, j, k});
    for (int i = 0; i <= D; ++i)
        for (int r = 0; r < 7; ++r)
            labels.push_back({LitaFamily::Local7, i, r, 0});
    for (int j = 0; j <= D; ++j)
        for (int i = 0; i <= D; ++i)
            if (i != j)
                for (int t = 0; t < 7; ++t)
                    labels.push_back({LitaFamily::Off, i, j, t});
    return labels;
}

template<class Field>
inline LitaBuilder<Field> lita_build(U32 N, const Field& field) {
    if (N % 2 != 0 || N < 18) throw std::runtime_error("LITA is implemented for even N >= 18");

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
    builder.U.reserve(static_cast<size_t>(lita_rank(N)));
    builder.V.reserve(static_cast<size_t>(lita_rank(N)));
    builder.W.reserve(static_cast<size_t>(lita_rank(N)));

    const LitaPhiCache<Field> cache = lita_make_phi_cache(N, field);
    const LitaLocal7<Field> coeffs = lita_local7(d, field);
    const LitaRemovedCfactors<Field> rt = lita_removed_cfactors(cache, M, coeffs, d, field);
    const LitaEndpoints endpoints = lita_agg2_endpoints(d);
    const std::vector<LitaLabel> labels = lita_certificate_labels(d);
    if (labels.size() != static_cast<size_t>(lita_base_rank(N))) {
        throw std::runtime_error("internal LITA certificate label count mismatch");
    }

    size_t q = 0;
    auto emit = [&](LitaRow<Field> u, LitaRow<Field> v, LitaRow<Field> c0) {
        if (q >= labels.size()) throw std::runtime_error("internal LITA certificate overrun");
        const LitaLabel cert = labels[q++];
        if (lita_is_removed_label(cert, d)) return;
        builder.add(
            std::move(u),
            std::move(v),
            lita_cert_cfactor<Field>(cert, std::move(c0), rt, d, endpoints, field)
        );
    };

    // Archive/materialization order used by LITA_ROW_CERT_TABLE in scripts/KGP2026.mpl.
    auto add_agg1 = [&](int i, int j, int k, bool barred) {
        auto f = [&](int x) { return barred ? bar(x) : foo(x); };
        emit(
            lita_star_form<Field>(cache, M, {{F(1), f(i), f(j)}, {F(1), f(j), f(k)}, {F(1), f(k), f(i)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), f(j), f(k)}, {F(1), f(k), f(i)}, {F(1), f(i), f(j)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), f(k), f(i)}, {F(1), f(i), f(j)}, {F(1), f(j), f(k)}}, field)
        );
    };

    for (int i = 0; i <= D; ++i) {
        for (int j = 0; j <= D; ++j) {
            for (int k = 0; k <= D; ++k) {
                const bool forward = i <= j && j < k;
                const bool backward = k < j && j <= i;
                if (!forward && !backward) continue;
                for (int barred = 0; barred < 2; ++barred) {
                    add_agg1(i, j, k, barred != 0);
                }
            }
        }
    }

    auto add_agg2a = [&] (int i, int j, int k) {
        emit(
            lita_star_form<Field>(cache, M, {{F(1), bar(j), foo(k)}, {F(1), foo(k), bar(i)}, {F(-1), foo(i), foo(j)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), foo(j), bar(k)}, {F(1), foo(k), foo(i)}, {F(1), bar(i), foo(j)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), foo(i), bar(j)}, {F(1), foo(j), foo(k)}, {F(-1), bar(k), foo(i)}}, field)
        );
    };
    auto add_agg2b = [&] (int i, int j, int k) {
        emit(
            lita_star_form<Field>(cache, M, {{F(1), foo(j), bar(k)}, {F(1), bar(k), foo(i)}, {F(-1), bar(i), bar(j)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), bar(j), foo(k)}, {F(1), bar(k), bar(i)}, {F(1), foo(i), bar(j)}}, field),
            lita_star_form<Field>(cache, M, {{F(1), bar(i), foo(j)}, {F(1), bar(j), bar(k)}, {F(-1), foo(k), bar(i)}}, field)
        );
    };

    for (int i = 0; i <= D; ++i) {
        for (int j = 0; j <= D; ++j) {
            for (int k = 0; k <= D; ++k) {
                if (i == j && j == k) continue;
                add_agg2a(i, j, k);
                add_agg2b(i, j, k);
            }
        }
    }

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

    for (int i = 0; i <= D; ++i) {
        for (int r = 0; r < 7; ++r) {
            emit(
                lita_form4(cache, M, coeffs[r][0], i, d, field),
                lita_form4(cache, M, coeffs[r][1], i, d, field),
                lita_form4(cache, M, coeffs[r][2], i, d, field)
            );
        }
    }

    if (q != labels.size()) {
        throw std::runtime_error("internal LITA certificate underrun");
    }
    if (builder.U.size() != lita_rank(N)) {
        throw std::runtime_error("internal LITA reduced rank mismatch");
    }
    return builder;
}

template<class Field>
inline SparseMatrixP lita_to_sparse_p(const std::vector<LitaRow<Field>>& rows, U32 cols) {
    SparseMatrixP out;
    out.rows = static_cast<U32>(rows.size());
    out.cols = cols;
    out.ptr.reserve(rows.size() + 1);
    out.ptr.push_back(0);
    for (const auto& row : rows) {
        for (const auto& entry : row) {
            out.col.push_back(entry.index);
            out.val.push_back(static_cast<U64>(entry.value));
        }
        out.ptr.push_back(static_cast<U64>(out.col.size()));
    }
    return out;
}

template<class Field>
inline SparseMatrixQ lita_to_sparse_q(const std::vector<LitaRow<Field>>& rows, U32 cols) {
    SparseMatrixQ out;
    out.rows = static_cast<U32>(rows.size());
    out.cols = cols;
    out.ptr.reserve(rows.size() + 1);
    out.ptr.push_back(0);
    for (const auto& row : rows) {
        for (const auto& entry : row) {
            out.col.push_back(entry.index);
            out.val.push_back(entry.value);
        }
        out.ptr.push_back(static_cast<U64>(out.col.size()));
    }
    return out;
}

template<class Field>
inline std::vector<LitaRow<Field>> lita_transpose_square_rows(
    std::vector<LitaRow<Field>> rows,
    U32 N,
    const Field& field
) {
    for (LitaRow<Field>& row : rows) {
        for (auto& entry : row) {
            const U32 i = entry.index / N;
            const U32 j = entry.index % N;
            entry.index = j * N + i;
        }
        lita_normalize_row(row, field);
    }
    return rows;
}

inline SchemeP lita_p(U32 N, U64 prime) {
    const LitaPField field{prime};
    const LitaBuilder<LitaPField> rows = lita_build(N, field);
    SchemeP out;
    out.U = lita_to_sparse_p(rows.U, N * N);
    out.V = lita_to_sparse_p(rows.V, N * N);
    out.W = lita_to_sparse_p(lita_transpose_square_rows(rows.W, N, field), N * N);
    return out;
}

inline SchemeQ lita_q(U32 N) {
    const LitaQField field{};
    const LitaBuilder<LitaQField> rows = lita_build(N, field);
    SchemeQ out;
    out.U = lita_to_sparse_q(rows.U, N * N);
    out.V = lita_to_sparse_q(rows.V, N * N);
    out.W = lita_to_sparse_q(lita_transpose_square_rows(rows.W, N, field), N * N);
    return out;
}
