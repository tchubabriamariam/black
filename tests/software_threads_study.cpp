//
// Software-threads / oversubscription study: naive vs shared, several formulas.
//
// The k-level parallel solver takes a thread count. Those are SOFTWARE threads:
// we may create far more of them than the machine has HARDWARE threads (cores),
// and the OS scheduler multiplexes them onto the cores. For each formula this
// program sweeps the software-thread count from 1 up past the hardware-thread
// count and runs BOTH parallel strategies at every step:
//
//   naive   = solve_parallel()         -- every thread rebuilds the whole
//                                          encoding itself (redundant, racy)
//   shared  = solve_parallel_shared()  -- one encoder builds each formula once,
//                                          workers only read it
//
// Columns (times in milliseconds):
//   threads        number of software threads requested
//   naive_ms       wall time of the naive solver
//   shared_ms      wall time of the shared solver
//   naive_vs_seq   sequential_time / naive_ms   (>1 = faster than sequential)
//   shared_vs_seq  sequential_time / shared_ms  (>1 = faster than sequential)
//   redundancy     naive: unravelings computed / needed (shared builds each once,
//                  so its redundancy is ~1 by design and is not re-measured here)
//   lock_waits     naive: times a thread waited for the shared uniquing lock
//
// Both speedups use the SAME sequential baseline, so naive and shared are
// directly comparable. A '*' marks oversubscription (threads > hardware
// threads); 'MISMATCH' marks a run whose parallel answer disagreed with the
// sequential one (the naive solver has an intermittent correctness bug).
//

#include <black/solver/solver.hpp>
#include <black/logic/logic.hpp>

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <cstdio>
#include <limits>

using namespace black;
using namespace std::chrono;

static constexpr size_t k_unbounded = std::numeric_limits<size_t>::max();

// n nested tomorrows: X(X(...X(f)))
static formula Xn(formula f, int n) {
  for(int i = 0; i < n; ++i) f = X(f);
  return f;
}

static const char* show(tribool t) {
  if(t == true)  return "SAT";
  if(t == false) return "UNSAT";
  return "?";
}

template<class F> static double ms_of(F&& fn) {
  auto a = steady_clock::now();
  fn();
  auto b = steady_clock::now();
  return duration<double, std::milli>(b - a).count();
}

int main() {
  alphabet sigma;
  scope xi{sigma};
  auto p = sigma.proposition("p");
  auto q = sigma.proposition("q");
  auto r = sigma.proposition("r");

  struct Case { std::string label; formula f; };
  std::vector<Case> cases = {
    // Easy (decides at k=1): no real parallelism to exploit.
    {"G(F(p)) & G(F(q)) & G(F(r))",  G(F(p)) && G(F(q)) && G(F(r))},
    // Hard family X^n(p) & X^n(!p): UNSAT, forces the decisive bound K*=n.
    {"X^12(p) & X^12(!p)",           Xn(p,12) && Xn(!p,12)},
    {"X^16(p) & X^16(!p)",           Xn(p,16) && Xn(!p,16)},
    {"X^20(p) & X^20(!p)",           Xn(p,20) && Xn(!p,20)},
  };

  size_t hw     = std::thread::hardware_concurrency();
  size_t pcores = solver::performance_core_count();

  std::cout << "Hardware threads: " << hw
            << "   Performance cores: " << pcores << "\n";
  std::cout << "(software threads > " << hw
            << " => oversubscription: the scheduler must time-slice)\n";

  std::vector<size_t> sweep = {1, 2, 4, 6, 8, 12, 16, 24, 48, 96};

  for(auto& c : cases) {
    // Sequential baseline (also gives us the result and the decisive depth K*).
    black::solver base;
    tribool seq_res{tribool::undef};
    double seq_ms = ms_of([&]{ seq_res = base.solve(xi, c.f); });
    size_t kstar  = base.last_bound();

    std::cout << "\n============================================================"
                 "==================================\n";
    std::cout << c.label << "   (" << show(seq_res) << ", K*=" << kstar
              << ")   sequential = " << std::fixed
              << [&]{ char b[32]; std::snprintf(b,sizeof(b),"%.1f",seq_ms); return std::string(b); }()
              << " ms\n";
    std::cout << std::string(108, '-') << "\n";
    std::printf("%8s %11s %11s %13s %14s %12s %13s\n",
                "threads", "naive_ms", "shared_ms", "naive_vs_seq",
                "shared_vs_seq", "redundancy", "lock_waits");
    std::cout << std::string(108, '-') << "\n";

    for(size_t T : sweep) {
      black::solver sn;
      tribool rn{tribool::undef};
      double naive_ms = ms_of([&]{
        rn = sn.solve_parallel(xi, c.f, false, k_unbounded, {}, false, T);
      });
      auto cnt = sn.last_parallel_counters();

      black::solver ss;
      tribool rs{tribool::undef};
      double shared_ms = ms_of([&]{
        rs = ss.solve_parallel_shared(xi, c.f, false, k_unbounded, {}, false, T, false);
      });

      bool mismatch = (rn != seq_res) || (rs != seq_res);
      std::string note;
      if(T > hw)   note += " *";
      if(mismatch) note += " MISMATCH";

      std::printf("%8zu %11.1f %11.1f %13.2f %14.2f %11.2fx %13zu%s\n",
                  T, naive_ms, shared_ms,
                  naive_ms  > 0 ? seq_ms / naive_ms  : 0.0,
                  shared_ms > 0 ? seq_ms / shared_ms : 0.0,
                  cnt.redundancy, cnt.lock_waits, note.c_str());
    }
  }

  std::cout << "\nnaive_vs_seq / shared_vs_seq = sequential time / that solver's"
               " time (>1 = faster than sequential).\n"
               "'*' = oversubscribed (threads > hardware threads).  redundancy &"
               " lock_waits are for the naive solver.\n";
  return 0;
}
