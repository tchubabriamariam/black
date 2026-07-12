//
// Benchmark: sequential vs naive-parallel vs shared-encoding-parallel,
// plus the hardware-independent parallelism upper bound.
//
// Columns:
//   Seq        wall time of the sequential solve()
//   NaivePar   solve_parallel()        (per-thread encoder, re-encodes)
//   SharedPar  solve_parallel_shared() (one shared read-only encoding)
//   xNaive     Seq / NaivePar
//   xShared    Seq / SharedPar
//   K*         decisive unrolling depth (from the instrumented run)
//   xIdeal     theoretical upper bound on speedup (unbounded cores)
//
// The final column comes from analyze_parallelism(): it measures the shape of
// the computation (critical path) rather than the machine, so it holds no
// matter how many cores are available.
//

#include <black/solver/solver.hpp>
#include <black/logic/logic.hpp>
#include <black/logic/prettyprint.hpp>

#include <iostream>
#include <chrono>
#include <vector>
#include <string>
#include <iomanip>
#include <limits>
#include <cstdio>

#if defined(__unix__) || defined(__APPLE__)
  #include <unistd.h>
  #include <sys/wait.h>
  #define BENCH_HAVE_FORK 1
#endif

using namespace black;
using namespace std::chrono;

static constexpr size_t k_unbounded = std::numeric_limits<size_t>::max();

struct timed {
  double ms;
  tribool res;
  bool crashed = false;
};

static int encode_res(tribool t) {
  if(t == true)  return 1;
  if(t == false) return 0;
  return -1;
}
static tribool decode_res(int i) {
  if(i == 1) return true;
  if(i == 0) return false;
  return tribool::undef;
}

static timed time_seq(scope& xi, formula f) {
  black::solver slv;
  auto start = steady_clock::now();
  tribool res = slv.solve(xi, f);
  auto end = steady_clock::now();
  return {duration<double, std::milli>(end - start).count(), res, false};
}

// The naive parallel solver has a data race on the shared alphabet (every
// thread's encoder mutates it without a lock) and can crash at higher thread
// counts. Run it in a forked child so a crash is reported rather than taking
// down the whole benchmark.
static timed time_naive(scope& xi, formula f, size_t threads) {
#if defined(BENCH_HAVE_FORK)
  int fds[2];
  if(pipe(fds) != 0) return {0.0, tribool::undef, true};

  pid_t pid = fork();
  if(pid == 0) {
    close(fds[0]);
    black::solver slv;
    auto start = steady_clock::now();
    tribool res =
      slv.solve_parallel(xi, f, false, k_unbounded, {}, false, threads);
    auto end = steady_clock::now();
    double ms = duration<double, std::milli>(end - start).count();
    int ires = encode_res(res);
    ssize_t w1 = write(fds[1], &ms, sizeof(ms));
    ssize_t w2 = write(fds[1], &ires, sizeof(ires));
    close(fds[1]);
    _exit((w1 == sizeof(ms) && w2 == sizeof(ires)) ? 0 : 1);
  }

  close(fds[1]);
  double ms = 0.0; int ires = -1;
  ssize_t r1 = read(fds[0], &ms, sizeof(ms));
  ssize_t r2 = read(fds[0], &ires, sizeof(ires));
  close(fds[0]);
  int status = 0;
  waitpid(pid, &status, 0);

  bool clean = WIFEXITED(status) && WEXITSTATUS(status) == 0
               && r1 == (ssize_t)sizeof(ms) && r2 == (ssize_t)sizeof(ires);
  if(!clean)
    return {0.0, tribool::undef, true};
  return {ms, decode_res(ires), false};
#else
  black::solver slv;
  auto start = steady_clock::now();
  tribool res = slv.solve_parallel(xi, f, false, k_unbounded, {}, false, threads);
  auto end = steady_clock::now();
  return {duration<double, std::milli>(end - start).count(), res, false};
#endif
}

static timed time_shared(scope& xi, formula f, size_t threads, bool pin) {
  black::solver slv;
  auto start = steady_clock::now();
  tribool res =
    slv.solve_parallel_shared(xi, f, false, k_unbounded, {}, false, threads, pin);
  auto end = steady_clock::now();
  return {duration<double, std::milli>(end - start).count(), res};
}

static const char* show(tribool t) {
  if(t == true)  return "SAT";
  if(t == false) return "UNSAT";
  return "?";
}

// n nested tomorrows: X(X(...X(f)))
static formula Xn(formula f, int n) {
  for(int i = 0; i < n; ++i) f = X(f);
  return f;
}

int main() {
  alphabet sigma;
  scope xi{sigma};

  auto p = sigma.proposition("p");
  auto q = sigma.proposition("q");
  auto r = sigma.proposition("r");
  auto s = sigma.proposition("s");

  struct BenchCase { std::string label; formula f; };

  std::vector<BenchCase> cases = {
    // --- Easy formulas (decide at small k): parallelism cannot help, so the
    //     thread/solver startup overhead makes the parallel versions slower.
    {"G(F(p & X(q & X(r))))",             G(F(p && X(q && X(r))))},
    {"G(F(p)) & G(F(q)) & G(F(r))",       G(F(p)) && G(F(q)) && G(F(r))},
    {"G(p->F(q)) & G(q->F(r)) & G(r->F(p)) & F(p)",
       G(implies(p,F(q))) && G(implies(q,F(r))) && G(implies(r,F(p))) && F(p)},
    {"G(!p) & (p U q)",                   G(!p) && U(p,q)},
    // --- Hard family X^n(p) & X^n(!p): UNSAT, forces the decisive bound K*=n,
    //     so the unrolling depth (and the parallelism on offer) grows with n.
    {"X^8(p) & X^8(!p)",    Xn(p,8)  && Xn(!p,8)},
    {"X^12(p) & X^12(!p)",  Xn(p,12) && Xn(!p,12)},
    {"X^16(p) & X^16(!p)",  Xn(p,16) && Xn(!p,16)},
    {"X^20(p) & X^20(!p)",  Xn(p,20) && Xn(!p,20)},
    {"X^24(p) & X^24(!p)",  Xn(p,24) && Xn(!p,24)},
  };
  (void)s;

  size_t pcores  = solver::performance_core_count();
  size_t threads = pcores >= 2 ? pcores : 2;

  std::cout << "Performance cores detected: " << pcores
            << "  (using " << threads << " worker threads, pinned to P-cores)\n\n";

  std::cout << std::left
            << std::setw(46) << "Formula"
            << std::setw(7)  << "Res"
            << std::right
            << std::setw(9)  << "Seq"
            << std::setw(9)  << "Naive"
            << std::setw(9)  << "Shared"
            << std::setw(9)  << "xNaive"
            << std::setw(9)  << "xShared"
            << std::setw(5)  << "K*"
            << std::setw(9)  << "xIdeal"
            << "\n"
            << std::string(115, '-') << "\n";

  for(auto& c : cases) {
    timed seq    = time_seq(xi, c.f);
    timed naive  = time_naive(xi, c.f, threads);
    timed shared = time_shared(xi, c.f, threads, /*pin=*/true);

    black::solver analyzer;
    auto rep = analyzer.analyze_parallelism(xi, c.f);

    // Correctness: every non-crashed variant must agree with sequential.
    bool ok = (shared.res == seq.res) && (rep.result == seq.res)
              && (naive.crashed || naive.res == seq.res);

    std::string label =
      c.label.size() > 44 ? c.label.substr(0,41) + "..." : c.label;

    auto cell = [](const timed& tm) -> std::string {
      if(tm.crashed) return "CRASH";
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.2f", tm.ms);
      return buf;
    };
    auto speedup = [](const timed& base, const timed& tm) -> std::string {
      if(tm.crashed || tm.ms <= 0) return "-";
      char buf[32];
      std::snprintf(buf, sizeof(buf), "%.2f", base.ms / tm.ms);
      return buf;
    };

    std::cout << std::left  << std::setw(46) << label
              << std::setw(7) << show(seq.res)
              << std::right  << std::fixed << std::setprecision(2)
              << std::setw(9) << cell(seq)
              << std::setw(9) << cell(naive)
              << std::setw(9) << cell(shared)
              << std::setw(9) << speedup(seq, naive)
              << std::setw(9) << speedup(seq, shared)
              << std::setw(5) << rep.decisive_k
              << std::setw(9) << rep.speedup_upper_bound
              << (ok ? "" : "   <-- MISMATCH!")
              << "\n";
  }

  std::cout << "\nLegend: times in ms. 'Naive' = per-thread encoder"
               " (racy; may CRASH at high thread counts).\n"
               "xIdeal is the theoretical speedup ceiling (unbounded cores),"
               " from the critical path\nof the instrumented sequential run.\n";

  return 0;
}
