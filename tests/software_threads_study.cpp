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
//   sw_threads       number of software threads requested for the run
//   wall_ms          wall-clock time of the whole solve, in milliseconds
//   speedup_vs_1     wall_ms(1 thread) / wall_ms(this run); >1 = faster than 1 thread
//   threads_started  software threads that actually began running
//   stuck_aborted    threads that gave up after another thread already decided (wasted)
//   stuck_lock_waits times a thread had to wait for the shared uniquing lock
//   unrav_needed     k-unravelings on the path to the answer (k = 0..K*)
//   unrav_computed   k-unravelings actually built across ALL threads
//   redundancy       unrav_computed / unrav_needed (how much extra work was done)
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

  std::cout << std::string(122, '-') << "\n";
  std::printf("%10s %9s %13s %15s %14s %16s %13s %15s %12s\n",
              "sw_threads", "wall_ms", "speedup_vs_1", "threads_started",
              "stuck_aborted", "stuck_lock_waits", "unrav_needed",
              "unrav_computed", "redundancy");
  std::cout << std::string(122, '-') << "\n";

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
    std::printf("%10zu %9.1f %13.2f %15zu %14zu %16zu %13zu %15zu %11.2fx%s\n",
                T, ms, base_ms > 0 ? base_ms / ms : 0.0,
                c.launched_threads, c.aborted_threads, c.lock_waits,
                c.unravelings_needed, c.unravelings_computed,
                c.redundancy, over);
  }

  std::cout << std::string(122, '-') << "\n";
  std::cout << "speedup_vs_1 = 1-thread time / this run's time (>1 means faster"
               " than one thread).\n"
               "'*' after redundancy = oversubscribed (more software threads than"
               " hardware threads).\n";
  return 0;
}
