// reduce.h
//
// Sparse 2-reduction tools for matrix multiplication schemes.
//
// A scheme is stored as three sparse factor matrices U, V, W whose rows are
// rank-one summands.  A 2-reduction is a linear dependency among one pair of
// factors, for example U_q tensor V_q.  The dependent summand can be removed
// and its complementary factor can be absorbed into the remaining rows.
//
// API:
//     SchemeP p = reduce::mod_scheme(q, prime);
//     auto rp = reduce::find_two_reductions(p, prime, reduce::UV);
//     auto rq = reduce::lift_two_reductions(rp, prime);
//     bool changed = reduce::two_reduce(q, reduce::UV, rq);
//
// find_two_reductions()
//     works over F_p and returns sparse modular dependency certificates.
//
// lift_two_reductions()
//     reconstructs coefficients over Q from one prime, using the fixed bound
//     kLiftMaxNumDen.
//
// two_reduce()
//     applies lifted certificates exactly to a SchemeQ.

#pragma once

#include "scheme.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace reduce {

struct Entry {
    U32 index = 0;
    U64 value = 0;
};

struct AxisRows {
    U32 cols = 0;
    std::vector<std::vector<Entry>> rows;
};

struct CscAxis {
    U32 rows = 0;
    U32 cols = 0;
    std::vector<std::vector<Entry>> columns;
};

struct PairSpec {
    int a = 0;
    int b = 1;

    int complement() const {
        return 3 - a - b;
    }
};

enum Pair : U32 {
    UV = 0,
    UW = 1,
    VW = 2,
};

inline PairSpec pair_spec(Pair pair) {
    if (pair == UV) return {0, 1};
    if (pair == UW) return {0, 2};
    if (pair == VW) return {1, 2};
    throw std::runtime_error("bad pair id");
}

enum class ColumnOrder {
    Asc,
    Desc,
    Natural,
    Random,
};

enum class PivotRule {
    First,
    MinDegree,
    MaxDegree,
};

struct Options {
    std::vector<PairSpec> order{{0, 1}, {0, 2}, {1, 2}};
    ColumnOrder column_order = ColumnOrder::Asc;
    PivotRule pivot_rule = PivotRule::First;
    U64 random_seed = 0x3243f6a8885a308dull;
};

struct PairScan {
    bool full_rank = false;
    U32 rank = 0;
    U32 nullity = 0;
    U64 generated_columns = 0;
    U64 nonzero_columns = 0;
    U64 processed_columns = 0;
    size_t max_input_nnz = 0;
    size_t max_pivot_nnz = 0;
    std::vector<U32> free_rows;
    std::vector<std::vector<U64>> coeff_basis;
};

struct SchemeRows {
    AxisRows axis[3];
};

struct ColumnSpec {
    U32 a_col = 0;
    U32 b_col = 0;
    U32 nnz = 0;
};

struct PivotColumn {
    U32 pivot_row = 0;
    std::vector<Entry> entries;
};

struct TwoReductionP {
    U32 removed = 0;
    std::vector<std::pair<U32, U64>> terms;
};

struct TwoReductionQ {
    U32 removed = 0;
    std::vector<std::pair<U32, Rat>> terms;
};

inline constexpr I64 kLiftMaxNumDen = 4096;

inline U64 mul_mod(U64 a, U64 b, U64 prime) {
    return (a * b) % prime;
}

inline U64 add_mod(U64 a, U64 b, U64 prime) {
    const U64 s = a + b;
    return (s >= prime || s < a) ? (s - prime) : s;
}

inline U64 mod_pow(U64 a, U64 e, U64 prime) {
    U64 out = 1;
    U64 base = a % prime;
    while (e != 0) {
        if ((e & 1) != 0) out = mul_mod(out, base, prime);
        base = mul_mod(base, base, prime);
        e >>= 1;
    }
    return out;
}

inline U64 inv_mod(U64 a, U64 prime) {
    if (a == 0) throw std::runtime_error("attempted to invert zero modulo prime");
    return mod_pow(a, prime - 2, prime);
}

inline void normalize_row(std::vector<Entry>& row, U64 prime) {
    for (Entry& entry : row) entry.value %= prime;
    row.erase(
        std::remove_if(row.begin(), row.end(), [](const Entry& entry) { return entry.value == 0; }),
        row.end()
    );
    std::sort(row.begin(), row.end(), [](const Entry& a, const Entry& b) {
        return a.index < b.index;
    });

    size_t out = 0;
    for (const Entry& entry : row) {
        if (out != 0 && row[out - 1].index == entry.index) {
            row[out - 1].value = add_mod(row[out - 1].value, entry.value, prime);
            if (row[out - 1].value == 0) --out;
        } else {
            row[out++] = entry;
        }
    }
    row.resize(out);
}

inline void scale_row(std::vector<Entry>& row, U64 factor, U64 prime) {
    if (factor == 1) return;
    for (Entry& entry : row) entry.value = mul_mod(entry.value, factor, prime);
}

inline std::vector<Entry> subtract_scaled(
    const std::vector<Entry>& row,
    const std::vector<Entry>& pivot,
    U64 factor,
    U64 prime
) {
    if (factor == 0) return row;

    std::vector<Entry> out;
    out.reserve(row.size() + pivot.size());
    size_t i = 0;
    size_t j = 0;
    while (i < row.size() || j < pivot.size()) {
        if (j >= pivot.size() || (i < row.size() && row[i].index < pivot[j].index)) {
            out.push_back(row[i++]);
        } else if (i >= row.size() || pivot[j].index < row[i].index) {
            const U64 sub = mul_mod(factor, pivot[j].value, prime);
            if (sub != 0) out.push_back({pivot[j].index, prime - sub});
            ++j;
        } else {
            const U64 sub = mul_mod(factor, pivot[j].value, prime);
            U64 value = row[i].value;
            value = (value >= sub) ? (value - sub) : (value + prime - sub);
            if (value != 0) out.push_back({row[i].index, value});
            ++i;
            ++j;
        }
    }
    return out;
}

inline AxisRows to_axis_rows(const SparseMatrixP& matrix, U64 prime) {
    if (matrix.ptr.size() != static_cast<size_t>(matrix.rows) + 1) {
        throw std::runtime_error("CSR ptr size does not match row count");
    }
    if (matrix.col.size() != matrix.val.size()) {
        throw std::runtime_error("CSR col and val sizes differ");
    }
    if (!matrix.ptr.empty() && matrix.ptr.front() != 0) {
        throw std::runtime_error("CSR ptr must start at zero");
    }
    if (!matrix.ptr.empty() && matrix.ptr.back() != matrix.col.size()) {
        throw std::runtime_error("CSR ptr back does not match nonzero count");
    }

    AxisRows out;
    out.cols = matrix.cols;
    out.rows.resize(matrix.rows);
    for (U32 row = 0; row < matrix.rows; ++row) {
        const U64 begin = matrix.ptr[row];
        const U64 end = matrix.ptr[row + 1];
        if (begin > end || end > matrix.col.size()) throw std::runtime_error("CSR ptr is not monotone");

        std::vector<Entry>& entries = out.rows[row];
        entries.reserve(static_cast<size_t>(end - begin));
        for (U64 offset = begin; offset < end; ++offset) {
            const U32 col = matrix.col[static_cast<size_t>(offset)];
            if (col >= matrix.cols) throw std::runtime_error("CSR column index out of bounds");
            entries.push_back({col, matrix.val[static_cast<size_t>(offset)] % prime});
        }
        normalize_row(entries, prime);
    }
    return out;
}

inline SparseMatrixP from_axis_rows(const AxisRows& axis) {
    SparseMatrixP out;
    if (axis.rows.size() > std::numeric_limits<U32>::max()) {
        throw std::runtime_error("too many rows for SparseMatrixP");
    }
    out.rows = static_cast<U32>(axis.rows.size());
    out.cols = axis.cols;
    out.ptr.reserve(axis.rows.size() + 1);
    out.ptr.push_back(0);
    for (const std::vector<Entry>& row : axis.rows) {
        for (const Entry& entry : row) {
            out.col.push_back(entry.index);
            out.val.push_back(entry.value);
        }
        out.ptr.push_back(static_cast<U64>(out.col.size()));
    }
    return out;
}

inline SchemeRows to_scheme_rows(const SchemeP& scheme, U64 prime) {
    SchemeRows out;
    out.axis[0] = reduce::to_axis_rows(scheme.U, prime);
    out.axis[1] = reduce::to_axis_rows(scheme.V, prime);
    out.axis[2] = reduce::to_axis_rows(scheme.W, prime);
    if (out.axis[0].rows.size() != out.axis[1].rows.size() ||
        out.axis[0].rows.size() != out.axis[2].rows.size()) {
        throw std::runtime_error("U, V, W row counts differ");
    }
    return out;
}

inline U32 rank_rows(const SchemeRows& scheme) {
    return static_cast<U32>(scheme.axis[0].rows.size());
}

inline void assign_scheme(SchemeP& scheme, const SchemeRows& rows) {
    scheme.U = from_axis_rows(rows.axis[0]);
    scheme.V = from_axis_rows(rows.axis[1]);
    scheme.W = from_axis_rows(rows.axis[2]);
}

inline CscAxis build_csc(const AxisRows& axis) {
    CscAxis out;
    out.rows = static_cast<U32>(axis.rows.size());
    out.cols = axis.cols;
    out.columns.resize(axis.cols);
    for (U32 row = 0; row < out.rows; ++row) {
        for (const Entry& entry : axis.rows[row]) {
            out.columns[entry.index].push_back({row, entry.value});
        }
    }
    return out;
}

inline std::vector<U32> pair_row_degrees(const AxisRows& a, const AxisRows& b) {
    if (a.rows.size() != b.rows.size()) throw std::runtime_error("axis row count mismatch");
    std::vector<U32> out(a.rows.size(), 0);
    for (size_t row = 0; row < a.rows.size(); ++row) {
        const U64 degree = U64(a.rows[row].size()) * U64(b.rows[row].size());
        out[row] = degree > std::numeric_limits<U32>::max()
            ? std::numeric_limits<U32>::max()
            : static_cast<U32>(degree);
    }
    return out;
}

inline std::vector<ColumnSpec> build_pair_column_specs(
    const AxisRows& a,
    const AxisRows& b,
    const Options& options
) {
    const U64 flat_size = U64(a.cols) * U64(b.cols);
    if (flat_size > static_cast<U64>(std::numeric_limits<size_t>::max())) {
        throw std::runtime_error("pair column space is too large");
    }

    std::vector<U32> counts(static_cast<size_t>(flat_size), 0);
    for (size_t row = 0; row < a.rows.size(); ++row) {
        for (const Entry& ea : a.rows[row]) {
            const size_t base = static_cast<size_t>(ea.index) * static_cast<size_t>(b.cols);
            for (const Entry& eb : b.rows[row]) {
                ++counts[base + static_cast<size_t>(eb.index)];
            }
        }
    }

    std::vector<ColumnSpec> specs;
    for (size_t flat = 0; flat < counts.size(); ++flat) {
        if (counts[flat] == 0) continue;
        specs.push_back({
            static_cast<U32>(flat / static_cast<size_t>(b.cols)),
            static_cast<U32>(flat % static_cast<size_t>(b.cols)),
            counts[flat],
        });
    }

    if (options.column_order == ColumnOrder::Asc) {
        std::sort(specs.begin(), specs.end(), [](const ColumnSpec& x, const ColumnSpec& y) {
            if (x.nnz != y.nnz) return x.nnz < y.nnz;
            if (x.a_col != y.a_col) return x.a_col < y.a_col;
            return x.b_col < y.b_col;
        });
    } else if (options.column_order == ColumnOrder::Desc) {
        std::sort(specs.begin(), specs.end(), [](const ColumnSpec& x, const ColumnSpec& y) {
            if (x.nnz != y.nnz) return x.nnz > y.nnz;
            if (x.a_col != y.a_col) return x.a_col < y.a_col;
            return x.b_col < y.b_col;
        });
    } else if (options.column_order == ColumnOrder::Random) {
        std::mt19937_64 rng(options.random_seed ^ (U64(a.cols) << 32) ^ U64(b.cols) ^ U64(a.rows.size()));
        std::shuffle(specs.begin(), specs.end(), rng);
    }

    return specs;
}

inline std::vector<Entry> make_pair_column(
    const CscAxis& a,
    const CscAxis& b,
    const ColumnSpec& spec,
    U64 prime
) {
    const std::vector<Entry>& ca = a.columns[spec.a_col];
    const std::vector<Entry>& cb = b.columns[spec.b_col];
    std::vector<Entry> out;
    out.reserve(std::min(ca.size(), cb.size()));

    size_t i = 0;
    size_t j = 0;
    while (i < ca.size() && j < cb.size()) {
        if (ca[i].index == cb[j].index) {
            const U64 value = mul_mod(ca[i].value, cb[j].value, prime);
            if (value != 0) out.push_back({ca[i].index, value});
            ++i;
            ++j;
        } else if (ca[i].index < cb[j].index) {
            ++i;
        } else {
            ++j;
        }
    }
    return out;
}

inline int find_existing_pivot_position(const std::vector<Entry>& column, const std::vector<int>& pivot_for_row) {
    for (int pos = 0; pos < static_cast<int>(column.size()); ++pos) {
        if (pivot_for_row[column[static_cast<size_t>(pos)].index] >= 0) return pos;
    }
    return -1;
}

inline int choose_new_pivot_position(
    const std::vector<Entry>& column,
    const std::vector<U32>& row_degrees,
    PivotRule rule
) {
    if (column.empty()) throw std::runtime_error("cannot choose pivot in empty column");
    int best = 0;
    if (rule == PivotRule::First) return best;

    U32 best_degree = row_degrees[column[0].index];
    for (int pos = 1; pos < static_cast<int>(column.size()); ++pos) {
        const U32 degree = row_degrees[column[static_cast<size_t>(pos)].index];
        if ((rule == PivotRule::MinDegree && degree < best_degree) ||
            (rule == PivotRule::MaxDegree && degree > best_degree)) {
            best_degree = degree;
            best = pos;
        }
    }
    return best;
}

inline std::vector<U64> build_dependency_from_free_row(
    U32 rows,
    U32 free_row,
    const std::vector<PivotColumn>& pivots,
    U64 prime
) {
    std::vector<U64> coeffs(rows, 0);
    coeffs[free_row] = 1;

    for (int pivot_index = static_cast<int>(pivots.size()) - 1; pivot_index >= 0; --pivot_index) {
        const PivotColumn& pivot = pivots[static_cast<size_t>(pivot_index)];
        U64 sum = 0;
        for (const Entry& entry : pivot.entries) {
            if (entry.index == pivot.pivot_row) continue;
            const U64 c = coeffs[entry.index];
            if (c != 0) sum += mul_mod(entry.value, c, prime);
            if (sum > (U64(1) << 62)) sum %= prime;
        }
        const U64 reduced = sum % prime;
        coeffs[pivot.pivot_row] = reduced == 0 ? 0 : prime - reduced;
    }
    return coeffs;
}

inline PairScan scan_pair(const SchemeRows& scheme, PairSpec pair, U64 prime, const Options& options = {}) {
    const U32 rows = rank_rows(scheme);
    const AxisRows& a = scheme.axis[pair.a];
    const AxisRows& b = scheme.axis[pair.b];

    const CscAxis csc_a = build_csc(a);
    const CscAxis csc_b = build_csc(b);
    const std::vector<ColumnSpec> specs = build_pair_column_specs(a, b, options);
    const std::vector<U32> row_degrees = pair_row_degrees(a, b);

    PairScan scan;
    scan.generated_columns = U64(a.cols) * U64(b.cols);
    scan.nonzero_columns = specs.size();

    std::vector<int> pivot_for_row(rows, -1);
    std::vector<PivotColumn> pivots;
    pivots.reserve(rows);

    for (const ColumnSpec& spec : specs) {
        ++scan.processed_columns;
        std::vector<Entry> column = make_pair_column(csc_a, csc_b, spec, prime);
        scan.max_input_nnz = std::max(scan.max_input_nnz, column.size());

        while (!column.empty()) {
            const int existing_pos = find_existing_pivot_position(column, pivot_for_row);
            if (existing_pos >= 0) {
                const Entry pivot_entry = column[static_cast<size_t>(existing_pos)];
                const int pivot_index = pivot_for_row[pivot_entry.index];
                column = subtract_scaled(column, pivots[static_cast<size_t>(pivot_index)].entries, pivot_entry.value, prime);
                continue;
            }

            const int new_pos = choose_new_pivot_position(column, row_degrees, options.pivot_rule);
            const U32 pivot_row = column[static_cast<size_t>(new_pos)].index;
            const U64 pivot_value = column[static_cast<size_t>(new_pos)].value;
            scale_row(column, inv_mod(pivot_value, prime), prime);
            scan.max_pivot_nnz = std::max(scan.max_pivot_nnz, column.size());
            pivot_for_row[pivot_row] = static_cast<int>(pivots.size());
            pivots.push_back({pivot_row, std::move(column)});
            break;
        }

        if (static_cast<U32>(pivots.size()) == rows) {
            scan.full_rank = true;
            scan.rank = rows;
            scan.nullity = 0;
            return scan;
        }
    }

    scan.full_rank = false;
    scan.rank = static_cast<U32>(pivots.size());
    scan.nullity = rows - scan.rank;
    for (U32 row = 0; row < rows; ++row) {
        if (pivot_for_row[row] >= 0) continue;
        std::vector<U64> coeffs = build_dependency_from_free_row(rows, row, pivots, prime);
        if (coeffs[row] != 1) throw std::runtime_error("nullspace basis has unexpected free-row coefficient");
        scan.free_rows.push_back(row);
        scan.coeff_basis.push_back(std::move(coeffs));
    }

    if (scan.free_rows.empty() || scan.free_rows.size() != scan.nullity) {
        throw std::runtime_error("failed to construct a dependency from a rank-deficient pair space");
    }
    return scan;
}

inline PairScan scan_pair(const SchemeP& scheme, PairSpec pair, U64 prime, const Options& options = {}) {
    return scan_pair(to_scheme_rows(scheme, prime), pair, prime, options);
}

inline bool two_reducible(const SchemeP& scheme, U64 prime, const Options& options = {}) {
    const SchemeRows rows = to_scheme_rows(scheme, prime);
    for (PairSpec pair : options.order) {
        if (!scan_pair(rows, pair, prime, options).full_rank) return true;
    }
    return false;
}

inline I64 checked_i128(__int128 value, const char* context) {
    if (value < static_cast<__int128>(std::numeric_limits<I64>::min()) ||
        value > static_cast<__int128>(std::numeric_limits<I64>::max())) {
        throw std::overflow_error(std::string("I64 overflow while ") + context);
    }
    return static_cast<I64>(value);
}

inline I64 abs_i64(I64 value) {
    if (value == std::numeric_limits<I64>::min()) {
        throw std::overflow_error("cannot take absolute value of I64 minimum");
    }
    return value < 0 ? -value : value;
}

inline Rat make_rat(I64 num, I64 den = 1) {
    if (den == 0) throw std::runtime_error("zero rational denominator");
    if (den < 0) {
        num = -num;
        den = -den;
    }
    if (num == 0) return {0, 1};
    const I64 g = std::gcd(abs_i64(num), den);
    return {num / g, den / g};
}

inline bool is_zero(Rat x) {
    return x.num == 0;
}

inline Rat neg_rat(Rat x) {
    return {-x.num, x.den};
}

inline Rat add_rat(Rat a, Rat b) {
    const I64 g = std::gcd(a.den, b.den);
    const I64 ad = a.den / g;
    const I64 bd = b.den / g;
    return make_rat(
        checked_i128(static_cast<__int128>(a.num) * bd + static_cast<__int128>(b.num) * ad, "adding rationals"),
        checked_i128(static_cast<__int128>(ad) * b.den, "adding rationals")
    );
}

inline Rat sub_rat(Rat a, Rat b) {
    return add_rat(a, neg_rat(b));
}

inline Rat mul_rat(Rat a, Rat b) {
    I64 an = a.num;
    I64 ad = a.den;
    I64 bn = b.num;
    I64 bd = b.den;

    I64 g = std::gcd(abs_i64(an), bd);
    an /= g;
    bd /= g;
    g = std::gcd(abs_i64(bn), ad);
    bn /= g;
    ad /= g;
    return make_rat(
        checked_i128(static_cast<__int128>(an) * bn, "multiplying rationals"),
        checked_i128(static_cast<__int128>(ad) * bd, "multiplying rationals")
    );
}

inline Rat div_rat(Rat a, Rat b) {
    if (b.num == 0) throw std::runtime_error("division by zero rational");
    return mul_rat(a, make_rat(b.den, b.num));
}

inline U64 mod_i64(I64 value, U64 prime) {
    if (prime > static_cast<U64>(std::numeric_limits<I64>::max())) {
        throw std::runtime_error("prime is too large for I64 modular conversion");
    }
    I64 out = value % static_cast<I64>(prime);
    if (out < 0) out += static_cast<I64>(prime);
    return static_cast<U64>(out);
}

inline U64 rat_mod(Rat value, U64 prime) {
    const U64 num = mod_i64(value.num, prime);
    const U64 den = mod_i64(value.den, prime);
    if (den == 0) throw std::runtime_error("rational denominator is zero modulo prime");
    return mul_mod(num, inv_mod(den, prime), prime);
}

inline Rat reconstruct_mod(U64 value, U64 prime) {
    if (value == 0) return {0, 1};
    if (prime > static_cast<U64>(std::numeric_limits<I64>::max())) {
        throw std::runtime_error("prime is too large for single-prime rational reconstruction");
    }

    for (I64 den = 1; den <= kLiftMaxNumDen; ++den) {
        I64 num = static_cast<I64>((static_cast<unsigned __int128>(value) * static_cast<U64>(den)) % prime);
        if (num > static_cast<I64>(prime / 2)) num -= static_cast<I64>(prime);
        if (abs_i64(num) <= kLiftMaxNumDen) {
            Rat out = make_rat(num, den);
            if (rat_mod(out, prime) == value % prime) return out;
        }
    }
    throw std::runtime_error("failed to lift modular coefficient");
}

inline SparseMatrixP mod_matrix(const SparseMatrixQ& matrix, U64 prime) {
    if (matrix.ptr.size() != static_cast<size_t>(matrix.rows) + 1) {
        throw std::runtime_error("CSR ptr size does not match row count");
    }
    if (matrix.col.size() != matrix.val.size()) {
        throw std::runtime_error("CSR col and val sizes differ");
    }
    if (!matrix.ptr.empty() && matrix.ptr.front() != 0) {
        throw std::runtime_error("CSR ptr must start at zero");
    }
    if (!matrix.ptr.empty() && matrix.ptr.back() != matrix.col.size()) {
        throw std::runtime_error("CSR ptr back does not match nonzero count");
    }

    SparseMatrixP out;
    out.rows = matrix.rows;
    out.cols = matrix.cols;
    out.ptr.reserve(static_cast<size_t>(matrix.rows) + 1);
    out.ptr.push_back(0);
    for (U32 row = 0; row < matrix.rows; ++row) {
        const U64 begin = matrix.ptr[row];
        const U64 end = matrix.ptr[row + 1];
        if (begin > end || end > matrix.col.size()) throw std::runtime_error("CSR ptr is not monotone");
        for (U64 offset = begin; offset < end; ++offset) {
            const U32 col = matrix.col[static_cast<size_t>(offset)];
            if (col >= matrix.cols) throw std::runtime_error("CSR column index out of bounds");
            const U64 value = rat_mod(matrix.val[static_cast<size_t>(offset)], prime);
            if (value == 0) continue;
            out.col.push_back(col);
            out.val.push_back(value);
        }
        out.ptr.push_back(static_cast<U64>(out.col.size()));
    }
    return out;
}

inline SchemeP mod_scheme(const SchemeQ& scheme, U64 prime) {
    SchemeP out;
    out.U = mod_matrix(scheme.U, prime);
    out.V = mod_matrix(scheme.V, prime);
    out.W = mod_matrix(scheme.W, prime);
    return out;
}

inline std::vector<TwoReductionP> find_two_reductions(
    const SchemeP& scheme,
    U64 prime,
    Pair pair,
    const Options& options = {}
) {
    const PairScan scan = scan_pair(scheme, pair_spec(pair), prime, options);
    std::vector<TwoReductionP> out;
    if (scan.full_rank) return out;

    out.reserve(scan.coeff_basis.size());
    for (size_t k = 0; k < scan.coeff_basis.size(); ++k) {
        TwoReductionP reduction;
        reduction.removed = scan.free_rows[k];
        const std::vector<U64>& coeffs = scan.coeff_basis[k];
        if (reduction.removed >= coeffs.size() || coeffs[reduction.removed] != 1) {
            throw std::runtime_error("modular dependency is not normalized at removed row");
        }
        for (U32 row = 0; row < static_cast<U32>(coeffs.size()); ++row) {
            const U64 coeff = coeffs[row];
            if (coeff != 0) reduction.terms.push_back({row, coeff});
        }
        out.push_back(std::move(reduction));
    }
    return out;
}

inline std::vector<TwoReductionQ> lift_two_reductions(
    const std::vector<TwoReductionP>& reductions,
    U64 prime
) {
    std::vector<TwoReductionQ> out;
    out.reserve(reductions.size());
    for (const TwoReductionP& reduction : reductions) {
        TwoReductionQ lifted;
        lifted.removed = reduction.removed;
        lifted.terms.reserve(reduction.terms.size());
        for (const auto& [row, coeff] : reduction.terms) {
            lifted.terms.push_back({row, reconstruct_mod(coeff, prime)});
        }
        out.push_back(std::move(lifted));
    }
    return out;
}

inline bool prune_zero_terms(SchemeRows& scheme) {
    bool changed = false;
    for (int row = static_cast<int>(rank_rows(scheme)) - 1; row >= 0; --row) {
        if (!scheme.axis[0].rows[static_cast<size_t>(row)].empty() &&
            !scheme.axis[1].rows[static_cast<size_t>(row)].empty() &&
            !scheme.axis[2].rows[static_cast<size_t>(row)].empty()) {
            continue;
        }
        for (AxisRows& axis : scheme.axis) {
            axis.rows.erase(axis.rows.begin() + row);
        }
        changed = true;
    }
    return changed;
}

inline void apply_dependencies(SchemeRows& scheme, PairSpec pair, const PairScan& scan, U64 prime) {
    if (scan.coeff_basis.empty()) return;

    const U32 rows = rank_rows(scheme);
    std::vector<char> removed(rows, 0);
    AxisRows& complement = scheme.axis[pair.complement()];
    std::vector<std::vector<Entry>> pivot_rows;
    pivot_rows.reserve(scan.free_rows.size());

    for (U32 free_row : scan.free_rows) {
        if (free_row >= rows) throw std::runtime_error("dependency free row out of bounds");
        if (removed[free_row] != 0) throw std::runtime_error("duplicate dependency free row");
        removed[free_row] = 1;
        pivot_rows.push_back(complement.rows[free_row]);
    }

    for (U32 row = 0; row < rows; ++row) {
        if (removed[row] != 0) continue;

        std::vector<Entry> updated = complement.rows[row];
        for (size_t k = 0; k < scan.coeff_basis.size(); ++k) {
            const U64 coeff = scan.coeff_basis[k][row];
            if (coeff != 0) updated = subtract_scaled(updated, pivot_rows[k], coeff, prime);
        }
        complement.rows[row] = std::move(updated);
    }

    std::vector<U32> free_rows = scan.free_rows;
    std::sort(free_rows.begin(), free_rows.end(), std::greater<U32>());
    for (U32 free_row : free_rows) {
        for (AxisRows& axis : scheme.axis) {
            axis.rows.erase(axis.rows.begin() + static_cast<std::ptrdiff_t>(free_row));
        }
    }
}

inline bool two_reduce(SchemeP& scheme, U64 prime, const Options& options = {}) {
    const U32 old_rank = scheme.U.rows;
    SchemeRows rows = to_scheme_rows(scheme, prime);

    while (true) {
        bool changed = prune_zero_terms(rows);

        for (PairSpec pair : options.order) {
            PairScan scan = scan_pair(rows, pair, prime, options);
            if (scan.full_rank) continue;

            apply_dependencies(rows, pair, scan, prime);
            changed = true;
            break;
        }

        if (!changed) break;
    }

    if (rank_rows(rows) == old_rank) return false;
    assign_scheme(scheme, rows);
    return true;
}

struct EntryQ {
    U32 index = 0;
    Rat value;
};

struct AxisRowsQ {
    U32 cols = 0;
    std::vector<std::vector<EntryQ>> rows;
};

struct SchemeRowsQ {
    AxisRowsQ axis[3];
};

inline void normalize_row_q(std::vector<EntryQ>& row) {
    row.erase(
        std::remove_if(row.begin(), row.end(), [](const EntryQ& entry) { return is_zero(entry.value); }),
        row.end()
    );
    std::sort(row.begin(), row.end(), [](const EntryQ& a, const EntryQ& b) {
        return a.index < b.index;
    });

    size_t out = 0;
    for (const EntryQ& entry : row) {
        if (out != 0 && row[out - 1].index == entry.index) {
            row[out - 1].value = add_rat(row[out - 1].value, entry.value);
            if (is_zero(row[out - 1].value)) --out;
        } else {
            row[out++] = entry;
        }
    }
    row.resize(out);
}

inline AxisRowsQ to_axis_rows_q(const SparseMatrixQ& matrix) {
    if (matrix.ptr.size() != static_cast<size_t>(matrix.rows) + 1) {
        throw std::runtime_error("CSR ptr size does not match row count");
    }
    if (matrix.col.size() != matrix.val.size()) {
        throw std::runtime_error("CSR col and val sizes differ");
    }
    if (!matrix.ptr.empty() && matrix.ptr.front() != 0) {
        throw std::runtime_error("CSR ptr must start at zero");
    }
    if (!matrix.ptr.empty() && matrix.ptr.back() != matrix.col.size()) {
        throw std::runtime_error("CSR ptr back does not match nonzero count");
    }

    AxisRowsQ out;
    out.cols = matrix.cols;
    out.rows.resize(matrix.rows);
    for (U32 row = 0; row < matrix.rows; ++row) {
        const U64 begin = matrix.ptr[row];
        const U64 end = matrix.ptr[row + 1];
        if (begin > end || end > matrix.col.size()) throw std::runtime_error("CSR ptr is not monotone");
        std::vector<EntryQ>& entries = out.rows[row];
        entries.reserve(static_cast<size_t>(end - begin));
        for (U64 offset = begin; offset < end; ++offset) {
            const U32 col = matrix.col[static_cast<size_t>(offset)];
            if (col >= matrix.cols) throw std::runtime_error("CSR column index out of bounds");
            const Rat value = matrix.val[static_cast<size_t>(offset)];
            if (!is_zero(value)) entries.push_back({col, value});
        }
        normalize_row_q(entries);
    }
    return out;
}

inline SparseMatrixQ from_axis_rows_q(const AxisRowsQ& axis) {
    SparseMatrixQ out;
    if (axis.rows.size() > std::numeric_limits<U32>::max()) {
        throw std::runtime_error("too many rows for SparseMatrixQ");
    }
    out.rows = static_cast<U32>(axis.rows.size());
    out.cols = axis.cols;
    out.ptr.reserve(axis.rows.size() + 1);
    out.ptr.push_back(0);
    for (const std::vector<EntryQ>& row : axis.rows) {
        for (const EntryQ& entry : row) {
            if (is_zero(entry.value)) continue;
            out.col.push_back(entry.index);
            out.val.push_back(entry.value);
        }
        out.ptr.push_back(static_cast<U64>(out.col.size()));
    }
    return out;
}

inline SchemeRowsQ to_scheme_rows_q(const SchemeQ& scheme) {
    SchemeRowsQ out;
    out.axis[0] = to_axis_rows_q(scheme.U);
    out.axis[1] = to_axis_rows_q(scheme.V);
    out.axis[2] = to_axis_rows_q(scheme.W);
    if (out.axis[0].rows.size() != out.axis[1].rows.size() ||
        out.axis[0].rows.size() != out.axis[2].rows.size()) {
        throw std::runtime_error("U, V, W row counts differ");
    }
    return out;
}

inline U32 rank_rows(const SchemeRowsQ& scheme) {
    return static_cast<U32>(scheme.axis[0].rows.size());
}

inline void assign_scheme(SchemeQ& scheme, const SchemeRowsQ& rows) {
    scheme.U = from_axis_rows_q(rows.axis[0]);
    scheme.V = from_axis_rows_q(rows.axis[1]);
    scheme.W = from_axis_rows_q(rows.axis[2]);
}

inline std::vector<EntryQ> subtract_scaled_q(
    const std::vector<EntryQ>& row,
    const std::vector<EntryQ>& pivot,
    Rat factor
) {
    if (is_zero(factor)) return row;

    std::vector<EntryQ> out;
    out.reserve(row.size() + pivot.size());
    size_t i = 0;
    size_t j = 0;
    while (i < row.size() || j < pivot.size()) {
        if (j >= pivot.size() || (i < row.size() && row[i].index < pivot[j].index)) {
            out.push_back(row[i++]);
        } else if (i >= row.size() || pivot[j].index < row[i].index) {
            const Rat value = neg_rat(mul_rat(factor, pivot[j].value));
            if (!is_zero(value)) out.push_back({pivot[j].index, value});
            ++j;
        } else {
            const Rat value = sub_rat(row[i].value, mul_rat(factor, pivot[j].value));
            if (!is_zero(value)) out.push_back({row[i].index, value});
            ++i;
            ++j;
        }
    }
    return out;
}

inline bool prune_zero_terms(SchemeRowsQ& scheme) {
    bool changed = false;
    for (int row = static_cast<int>(rank_rows(scheme)) - 1; row >= 0; --row) {
        if (!scheme.axis[0].rows[static_cast<size_t>(row)].empty() &&
            !scheme.axis[1].rows[static_cast<size_t>(row)].empty() &&
            !scheme.axis[2].rows[static_cast<size_t>(row)].empty()) {
            continue;
        }
        for (AxisRowsQ& axis : scheme.axis) {
            axis.rows.erase(axis.rows.begin() + row);
        }
        changed = true;
    }
    return changed;
}

inline Rat coefficient_for_removed(const TwoReductionQ& reduction) {
    for (const auto& [row, coeff] : reduction.terms) {
        if (row == reduction.removed) return coeff;
    }
    throw std::runtime_error("Q reduction does not contain removed row");
}

inline bool two_reduce(SchemeQ& scheme, Pair pair, const std::vector<TwoReductionQ>& reductions) {
    if (reductions.empty()) return false;

    const U32 old_rank = scheme.U.rows;
    SchemeRowsQ rows = to_scheme_rows_q(scheme);
    const U32 row_count = rank_rows(rows);
    const int complement = pair_spec(pair).complement();

    std::vector<char> removed(row_count, 0);
    std::vector<std::vector<std::pair<U32, Rat>>> coeffs;
    std::vector<std::vector<EntryQ>> pivot_rows;
    coeffs.reserve(reductions.size());
    pivot_rows.reserve(reductions.size());

    for (const TwoReductionQ& reduction : reductions) {
        if (reduction.removed >= row_count) throw std::runtime_error("removed row is out of bounds");
        if (removed[reduction.removed] != 0) throw std::runtime_error("duplicate removed row in reduction batch");

        const Rat removed_coeff = coefficient_for_removed(reduction);
        if (is_zero(removed_coeff)) throw std::runtime_error("removed row coefficient is zero");

        std::vector<std::pair<U32, Rat>> normalized;
        normalized.reserve(reduction.terms.size());
        for (const auto& [row, coeff] : reduction.terms) {
            if (row >= row_count) throw std::runtime_error("reduction row is out of bounds");
            const Rat value = div_rat(coeff, removed_coeff);
            if (!is_zero(value)) normalized.push_back({row, value});
        }

        removed[reduction.removed] = 1;
        coeffs.push_back(std::move(normalized));
        pivot_rows.push_back(rows.axis[complement].rows[reduction.removed]);
    }

    for (U32 row = 0; row < row_count; ++row) {
        if (removed[row] != 0) continue;
        std::vector<EntryQ> updated = rows.axis[complement].rows[row];
        for (size_t k = 0; k < coeffs.size(); ++k) {
            Rat coeff{};
            for (const auto& term : coeffs[k]) {
                if (term.first == row) {
                    coeff = term.second;
                    break;
                }
            }
            if (!is_zero(coeff)) updated = subtract_scaled_q(updated, pivot_rows[k], coeff);
        }
        rows.axis[complement].rows[row] = std::move(updated);
    }

    std::vector<U32> removed_rows;
    removed_rows.reserve(reductions.size());
    for (const TwoReductionQ& reduction : reductions) removed_rows.push_back(reduction.removed);
    std::sort(removed_rows.begin(), removed_rows.end(), std::greater<U32>());
    for (U32 row : removed_rows) {
        for (AxisRowsQ& axis : rows.axis) {
            axis.rows.erase(axis.rows.begin() + static_cast<std::ptrdiff_t>(row));
        }
    }
    prune_zero_terms(rows);
    assign_scheme(scheme, rows);
    return scheme.U.rows < old_rank;
}

}  // namespace reduce
