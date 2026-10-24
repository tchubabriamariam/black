//
// Cube-and-conquer study.
//
// Two parts:
//
// 1. Correctness: for a set of SAT and UNSAT formulas, solve_cube_and_conquer()
//    must return the sequential answer, and for SAT formulas the same model
//    length (it explores k in the same order as solve()), at several thread
//    counts and cube depths, with and without the whole-check worker. Every
//    run is repeated to shake out races.
//
// 2. Performance: sequential vs k-racing (solve_parallel_shared) vs pure
//    cube-and-conquer vs concurrent cube-and-conquer (one worker solves each
//    check whole, racing the cubes), over a sweep of thread counts. Columns:
//      threads   worker threads
//      shared_x  sequential time / solve_parallel_shared() time
//      cnc_x     sequential time / pure cube-and-conquer time
//      ccc_x     sequential time / concurrent cube-and-conquer time
//                (>1 = faster than sequential)
//      d         cube depth used (2^d cubes per check)
//      ccc cubes cubes generated / refuted / skipped (check already settled)
//      whole     ccc: checks settled by the whole-check worker / all checks
//
// Usage: cube_and_conquer_study [reps] [formula files...]
// Formula files (BLACK syntax) are added to the performance part.
//

#include <black/solver/solver.hpp>
#include <black/logic/logic.hpp>

#include <iostream>
#include <chrono>
#include <thread>
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <fstream>
#include <sstream>

#include <black/logic/parser.hpp>

using namespace black;
using namespace std::chrono;

static constexpr size_t k_unbounded = std::numeric_limits<size_t>::max();

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

int main(int argc, char** argv) {
  int reps = argc > 1 ? std::atoi(argv[1]) : 5;

  alphabet sigma;
  scope xi{sigma};
  auto p = sigma.proposition("p");
  auto q = sigma.proposition("q");
  auto r = sigma.proposition("r");
  auto s = sigma.proposition("s");

  struct Case { std::string label; formula f; };

  std::vector<Case> correctness = {
    {"p",                              p},
    {"p & !p",                         p && !p},
    {"G(F(p)) & G(F(q)) & G(F(r))",    G(F(p)) && G(F(q)) && G(F(r))},
    {"G(p -> X(!p)) & G(F(p))",        G(implies(p, X(!p))) && G(F(p))},
    {"G(p) & F(!p)",                   G(p) && F(!p)},
    {"p U (q & X(!p)) & G(!q | r)",    U(p, q && X(!p)) && G(!q || r)},
    {"F(G(p)) & G(F(!p))",             F(G(p)) && G(F(!p))},
    {"G(p <-> X(!p)) & X^5(p) & X^6(p)",
       G(iff(p, X(!p))) && Xn(p,5) && Xn(p,6)},
    {"X^8(p) & X^8(!p)",               Xn(p,8) && Xn(!p,8)},
    {"X^6(p & q) & G(q -> X(r)) & F(G(!r))",
       Xn(p && q,6) && G(implies(q, X(r))) && F(G(!r))},
    {"G(F(p & q & r & s)) & G(p -> X(!p))",
       G(F(p && q && r && s)) && G(implies(p, X(!p)))},
  };

  std::cout << "== Correctness (" << reps << " reps per configuration) ==\n";
  size_t failures = 0, runs = 0;
  for(auto &c : correctness) {
    black::solver base;
    tribool ref = base.solve(xi, c.f);
    size_t ref_size = base.model() ? base.model()->size() : 0;

    for(size_t T : std::vector<size_t>{1, 2, 4, 6, 12}) {
      for(size_t d : std::vector<size_t>{0, 1, 3}) {
        for(bool whole : {false, true}) {
          for(int i = 0; i < reps; ++i) {
            black::solver sc;
            tribool res = sc.solve_cube_and_conquer(
              xi, c.f, false, k_unbounded, {}, false, T, d, whole
            );
            size_t size = sc.model() ? sc.model()->size() : 0;
            ++runs;
            if(res != ref || size != ref_size) {
              ++failures;
              std::printf(
                "  MISMATCH %-40s T=%zu d=%zu whole=%d: %s/%zu vs %s/%zu\n",
                c.label.c_str(), T, d, int(whole), show(res), size,
                show(ref), ref_size
              );
            }
          }
        }
      }
    }
    std::printf("  %-44s %-5s model size %zu\n",
                c.label.c_str(), show(ref), ref_size);
  }
  std::printf("  %zu runs, %zu mismatches\n", runs, failures);

  std::vector<Case> perf = {
    {"G(F(p)) & G(F(q)) & G(F(r))",  G(F(p)) && G(F(q)) && G(F(r))},
    {"X^12(p) & X^12(!p)",           Xn(p,12) && Xn(!p,12)},
    {"X^20(p) & X^20(!p)",           Xn(p,20) && Xn(!p,20)},
    {"G(p <-> X(!p)) & X^15(p) & X^16(p)",
       G(iff(p, X(!p))) && Xn(p,15) && Xn(p,16)},
  };

  for(int i = 2; i < argc; ++i) {
    std::ifstream in{argv[i]};
    std::stringstream text;
    text << in.rdbuf();
    auto parsed = black::parse_formula(sigma, text.str(), [&](auto err) {
      std::cerr << argv[i] << ": " << err << "\n";
    });
    if(parsed)
      perf.push_back({argv[i], *parsed});
  }

  size_t hw = std::thread::hardware_concurrency();
  std::cout << "\n== Performance ==   hardware threads: " << hw
            << "   performance cores: " << solver::performance_core_count()
            << "\n";

  for(auto &c : perf) {
    black::solver base;
    tribool seq_res = tribool::undef;
    double seq_ms = ms_of([&]{ seq_res = base.solve(xi, c.f); });

    std::printf("\n%s   (%s, K*=%zu)   sequential = %.1f ms\n",
                c.label.c_str(), show(seq_res), base.last_bound(), seq_ms);
    std::printf("%8s %9s %9s %9s %3s %22s %9s\n",
                "threads", "shared_x", "cnc_x", "ccc_x",
                "d", "ccc cubes gen/unsat/skip", "whole");

    for(size_t T : std::vector<size_t>{2, 4, 6, 8, 12}) {
      black::solver ss;
      tribool rs = tribool::undef;
      double shared_ms = ms_of([&]{
        rs = ss.solve_parallel_shared(
          xi, c.f, false, k_unbounded, {}, false, T, true
        );
      });

      black::solver sc;
      tribool rc = tribool::undef;
      double cnc_ms = ms_of([&]{
        rc = sc.solve_cube_and_conquer(
          xi, c.f, false, k_unbounded, {}, false, T, 0, false, true
        );
      });

      black::solver sh;
      tribool rh = tribool::undef;
      double ccc_ms = ms_of([&]{
        rh = sh.solve_cube_and_conquer(
          xi, c.f, false, k_unbounded, {}, false, T, 0, true, true
        );
      });
      auto cnt = sh.last_cube_counters();

      char cubes[64], whole[32];
      std::snprintf(cubes, sizeof(cubes), "%zu/%zu/%zu",
                    cnt.cubes_generated, cnt.cubes_unsat, cnt.cubes_skipped);
      std::snprintf(whole, sizeof(whole), "%zu/%zu",
                    cnt.full_check_wins, cnt.checks);

      bool mismatch = rs != seq_res || rc != seq_res || rh != seq_res;
      std::printf("%8zu %9.2f %9.2f %9.2f %3zu %22s %9s%s\n",
                  T, seq_ms / shared_ms, seq_ms / cnc_ms, seq_ms / ccc_ms,
                  cnt.cube_depth, cubes, whole,
                  mismatch ? "  MISMATCH" : "");
    }
  }

  return failures == 0 ? 0 : 1;
}
