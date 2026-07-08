#include "reduce.h"
#include "ta.h"

#include <chrono>
#include <iomanip>
#include <iostream>
#include <vector>

namespace {

constexpr U64 kPrime = 1000003;
constexpr U32 kN = 18;

void reduce_all_pairs(SchemeQ& scheme) {
    while (true) {
        bool changed = false;
        const SchemeP mod = reduce::mod_scheme(scheme, kPrime);
        for (reduce::Pair pair : {reduce::UV, reduce::UW, reduce::VW}) {
            const std::vector<reduce::TwoReductionP> reductions_p =
                reduce::find_two_reductions(mod, kPrime, pair);
            if (reductions_p.empty()) continue;

            const std::vector<reduce::TwoReductionQ> reductions_q =
                reduce::lift_two_reductions(reductions_p, kPrime);
            reduce::two_reduce(scheme, pair, reductions_q);
            changed = true;
            break;
        }
        if (!changed) break;
    }
}

void run_two_reduce(const char* name, U32 N, SchemeQ scheme) {
    const U32 before = scheme.U.rows;
    const auto start = std::chrono::steady_clock::now();
    reduce_all_pairs(scheme);
    const auto finish = std::chrono::steady_clock::now();
    const double seconds = std::chrono::duration<double>(finish - start).count();

    std::cout << name << "(N=" << N << ") two_reduce: "
              << before << " -> " << scheme.U.rows << " in "
              << std::fixed << std::setprecision(2) << seconds << " s\n";
}

}  // namespace

int main() {
    run_two_reduce("ta", kN, ta_q(kN));
    run_two_reduce("lita", kN, lita_q(kN));
    return 0;
}
