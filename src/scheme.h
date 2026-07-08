#pragma once

#include <cstdint>
#include <vector>

using U32 = std::uint32_t;
using U64 = std::uint64_t;
using I64 = std::int64_t;

struct Rat {
    I64 num = 0;
    I64 den = 1;
};

struct SparseMatrixP {
    U32 rows = 0;
    U32 cols = 0;
    std::vector<U64> ptr;
    std::vector<U32> col;
    std::vector<U64> val;
};

struct SchemeP {
    SparseMatrixP U;
    SparseMatrixP V;
    SparseMatrixP W;
};

struct SparseMatrixQ {
    U32 rows = 0;
    U32 cols = 0;
    std::vector<U64> ptr;
    std::vector<U32> col;
    std::vector<Rat> val;
};

struct SchemeQ {
    SparseMatrixQ U;
    SparseMatrixQ V;
    SparseMatrixQ W;
};
