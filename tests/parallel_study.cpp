//
// Formula-family study: which formulas benefit from k-level parallelism?
//
// For a range of parametric formula families (and a sample of real benchmark
// formulas) this reports, per formula:
//
//   K*        the decisive unrolling depth
//   seq       sequential solve() time
//   shared    solve_parallel_shared() time (P-core count threads)
//   xShared   seq / shared          (actual speedup)
//   xIdeal    hardware-independent speedup ceiling (analyze_parallelism)
//
// and a three-way split of the sequential time (which explains xIdeal):
//
//   par%   SAT checks at depths k < K*  -- the ONLY part that overlaps across
//          threads (the "parallel portion", Amdahl-style)
//   enc%   the encode/assert spine every thread must rebuild   (serial)
//   dec%   the single decisive SAT check at K*                 (serial)
//
// Finding: benefit tracks par% almost exactly. Only "check-spread" families
// (e.g. X^n p & X^n !p) put most of the time into intermediate checks and
// actually beat sequential; "encode-bound" and "decision-bound" families do
// not, and can be much slower in parallel because every thread redundantly
// rebuilds/re-solves. Most real formulas have small K*, so there is little to
// parallelise in practice.
//

#include <black/solver/solver.hpp>
#include <black/logic/logic.hpp>
#include <black/logic/parser.hpp>

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <vector>
#include <functional>
#include <string>
#include <limits>
#include <cstdio>

using namespace black;
using namespace std::chrono;

static constexpr size_t k_unbounded = std::numeric_limits<size_t>::max();

static formula Xn(formula f, int n) { for(int i = 0; i < n; ++i) f = X(f); return f; }

template<class F> static double ms_of(F&& fn) {
  auto a = steady_clock::now(); fn(); auto b = steady_clock::now();
  return duration<double, std::milli>(b - a).count();
}
static const char* R(tribool t) { return t == true ? "SAT" : t == false ? "UNSAT" : "?"; }

int main() {
  alphabet sigma;
  scope xi{sigma};
  auto p = sigma.proposition("p");
  auto q = sigma.proposition("q");
  auto r = sigma.proposition("r");

  auto conj = [&](std::vector<formula> v) {
    formula f = sigma.top();
    for(auto& x : v) f = f && x;
    return f;
  };

  // Ring counter over n states: forced lasso of period n (SAT, K*=n).
  auto ring = [&](int n) {
    size_t N = size_t(n);
    std::vector<proposition> s;
    for(size_t i = 0; i < N; ++i) s.push_back(sigma.proposition("s" + std::to_string(i)));
    std::vector<formula> ps;
    ps.push_back(s[0]);
    for(size_t i = 1; i < N; ++i) ps.push_back(!s[i]);
    for(size_t i = 0; i < N; ++i) ps.push_back(G(implies(s[i], X(s[(i + 1) % N]))));
    for(size_t i = 0; i < N; ++i)
      for(size_t j = i + 1; j < N; ++j) ps.push_back(G(!(s[i] && s[j])));
    return conj(ps);
  };

  size_t T = solver::performance_core_count();
  std::cout << "threads=" << T << " (P-cores)\n\n";
  std::cout << "PARAMETRIC FAMILIES\n";
  std::cout << "family                         n   res    K*      seq     shared"
               "  xShared  xIdeal | par% enc% dec%\n";
  std::cout << std::string(104, '-') << "\n";

  struct Fam { std::string name; std::function<formula(int)> gen; };
  std::vector<Fam> fams = {
    {"UNSAT Xn p & Xn !p  (spread)",  [&](int n){ return Xn(p,n) && Xn(!p,n); }},
    {"UNSAT p&G(p->Xp)&Xn !p (conc)", [&](int n){ return p && G(implies(p,X(p))) && Xn(!p,n); }},
    {"SAT   ring counter    (conc)",  [&](int n){ return ring(n); }},
    {"SAT   G(Fp)&G(Fq)&G(Fr)(easy)", [&](int n){ (void)n; return G(F(p)) && G(F(q)) && G(F(r)); }},
  };

  for(auto& fam : fams) {
    for(int n : {6, 10, 14, 18}) {
      formula f = fam.gen(n);
      tribool r1{tribool::undef}, r3{tribool::undef};

      black::solver s0;
      auto rep = s0.analyze_parallelism(xi, f);
      double seq    = ms_of([&]{ black::solver s; r1 = s.solve(xi, f); });
      double shared = ms_of([&]{
        black::solver s;
        r3 = s.solve_parallel_shared(xi, f, false, k_unbounded, {}, false, T, true);
      });

      double sumc = 0; for(double c : rep.check_ms)  sumc += c;
      double suma = 0; for(double a : rep.assert_ms) suma += a;
      double finalc = rep.check_ms.empty() ? 0 : rep.check_ms.back();
      double total  = suma + sumc;
      double par = total > 0 ? 100.0 * (sumc - finalc) / total : 0;
      double enc = total > 0 ? 100.0 * suma / total : 0;
      double dec = total > 0 ? 100.0 * finalc / total : 0;

      bool ok = (r1 == r3) && (rep.result == r1);
      std::printf("%-30s %2d  %-5s %4zu  %8.1f %8.1f  %6.2f  %6.2f | %4.0f %4.0f %4.0f%s\n",
        fam.name.c_str(), n, R(rep.result), rep.decisive_k, seq, shared,
        shared > 0 ? seq / shared : 0.0, rep.speedup_upper_bound,
        par, enc, dec, ok ? "" : "  MISMATCH");
    }
    std::cout << "\n";
  }

  // Real benchmark formulas, k capped at 25 so intractable ones don't hang.
  std::cout << "REAL BENCHMARK FORMULAS (k capped at 25)\n";
  std::cout << "file                                  res    K*      seq   xIdeal\n";
  std::cout << std::string(70, '-') << "\n";
  std::vector<std::string> files = {
    "benchmarks/formulas/future_only/acacia/example/demo-v15.pltl",
    "benchmarks/formulas/future_only/rozier/counter/counterCarry/counterCarry4.pltl",
    "benchmarks/formulas/future_only/schuppan/O1formula/O1formula20.pltl",
    "benchmarks/formulas/future_only/forobots/forobotsr1f0_GF_d.pltl",
  };
  for(auto& path : files) {
    std::ifstream in(path);
    if(!in) { std::cout << path << " (missing - run from repo root)\n"; continue; }
    std::stringstream ss; ss << in.rdbuf();
    auto pf = parse_formula(sigma, ss.str());
    if(!pf) { std::cout << path << " (parse error)\n"; continue; }

    black::solver s0;
    auto rep = s0.analyze_parallelism(xi, *pf, false, 25);
    double seq = ms_of([&]{ black::solver s; s.solve(xi, *pf, false, 25); });
    std::string b = path.substr(path.find_last_of('/') + 1);
    std::printf("%-36s  %-5s %4zu  %8.1f  %6.2f\n",
      b.c_str(), R(rep.result), rep.decisive_k, seq, rep.speedup_upper_bound);
  }

  return 0;
}
