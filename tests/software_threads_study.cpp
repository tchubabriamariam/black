//
// Software-threads / oversubscription study.
//
// The k-level parallel solver takes a thread count. Those are SOFTWARE threads:
// we may create far more of them than the machine has HARDWARE threads (cores),
// and the OS scheduler multiplexes them onto the cores. This program sweeps the
// software-thread count from 1 up past the hardware-thread count and, for each
// setting, reports both the wall-clock time and the instrumentation counters
// recorded by solve_parallel():
//
//   launched     software threads that actually started
//   aborted      threads that gave up after another thread decided (wasted)
//   needed       k-unravelings on the path to the answer (k = 0..K*)
//   computed     k-unravelings actually built across ALL threads
//   redundancy   computed / needed  (how much extra work the threads did)
//
// What to look for:
//   * redundancy climbs roughly with the thread count (each thread rebuilds the
//     unraveling spine) until it saturates near K*+1 -- past that, extra threads
//     race on k values beyond the decisive one and mostly just abort.
//   * wall time stops improving (and usually worsens) once software threads
//     exceed hardware threads: the scheduler is time-slicing, so more threads
//     buy context-switch overhead, not parallelism.
//
// Run `black::solver::performance_core_count()` and hardware_concurrency() are
// printed so the oversubscription point (threads > cores) is visible.
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

int main(int argc, char** argv) {
  // Hard UNSAT family X^n(p) & X^n(!p): forces the decisive bound K* = n, so
  // there is a real unraveling spine to (re)build. n can be overridden on the
  // command line.
  int n = (argc > 1) ? std::atoi(argv[1]) : 20;

  alphabet sigma;
  scope xi{sigma};
  auto p = sigma.proposition("p");
  formula f = Xn(p, n) && Xn(!p, n);

  size_t hw     = std::thread::hardware_concurrency();
  size_t pcores = solver::performance_core_count();

  std::cout << "Formula: X^" << n << "(p) & X^" << n << "(!p)   (UNSAT, K*=" << n << ")\n";
  std::cout << "Hardware threads: " << hw
            << "   Performance cores: " << pcores << "\n";
  std::cout << "(software threads > " << hw
            << " means the scheduler must time-slice: oversubscription)\n\n";

  std::cout << std::string(84, '-') << "\n";
  std::printf("%8s %10s %9s %9s %8s %8s %10s %11s\n",
              "threads", "wall(ms)", "vs1", "launched",
              "aborted", "needed", "computed", "redundancy");
  std::cout << std::string(84, '-') << "\n";

  std::vector<size_t> sweep = {1, 2, 4, 6, 8, 12, 16, 24, 48, 96};

  double base_ms = 0.0; // 1-thread wall time, for the speedup column

  for(size_t T : sweep) {
    black::solver s;
    auto a = steady_clock::now();
    tribool res = s.solve_parallel(xi, f, false, k_unbounded, {}, false, T);
    auto b = steady_clock::now();
    double ms = duration<double, std::milli>(b - a).count();
    (void)res;

    auto c = s.last_parallel_counters();
    if(T == 1) base_ms = ms;

    const char* over = (T > hw) ? " *" : "";
    std::printf("%8zu %10.1f %9.2f %9zu %8zu %8zu %10zu %10.2fx%s\n",
                T, ms, base_ms > 0 ? base_ms / ms : 0.0,
                c.launched_threads, c.aborted_threads,
                c.unravelings_needed, c.unravelings_computed,
                c.redundancy, over);
  }

  std::cout << std::string(84, '-') << "\n";
  std::cout << "vs1 = speedup vs the 1-thread run.  '*' = oversubscribed"
               " (software threads > hardware threads).\n";
  return 0;
}
