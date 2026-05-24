//
// Benchmark: sequential vs parallel solve
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

using namespace black;
using namespace std::chrono;

static double time_seq(scope& xi, formula f) {
  black::solver slv;   // fresh solver every call
  auto start = high_resolution_clock::now();
  tribool res = slv.solve(xi, f);
  auto end = high_resolution_clock::now();
  (void)res;
  return duration<double, std::milli>(end - start).count();
}

static double time_par(scope& xi, formula f, size_t threads) {
  black::solver slv;   // fresh solver every call
  auto start = high_resolution_clock::now();
  tribool res = slv.solve_parallel(xi, f, false,
        std::numeric_limits<size_t>::max(), {}, false, threads);
  auto end = high_resolution_clock::now();
  (void)res;
  return duration<double, std::milli>(end - start).count();
}

int main() {
  alphabet sigma;
  scope xi{sigma};

  auto p = sigma.proposition("p");
  auto q = sigma.proposition("q");
  auto r = sigma.proposition("r");
  auto s = sigma.proposition("s");

  struct BenchCase {
    std::string label;
    formula f;
  };

  std::vector<BenchCase> cases = {
    {"G(F(p & X(q & X(r))))",
      G(F(p && X(q && X(r))))},
    {"G(F(p)) & G(F(q)) & G(F(r))",
      G(F(p)) && G(F(q)) && G(F(r))},
    {"G(p->F(q)) & G(q->F(r)) & G(r->F(p)) & F(p)",
      G(implies(p,F(q))) && G(implies(q,F(r))) && G(implies(r,F(p))) && F(p)},
    {"G(F(p)) & G(F(q)) & G(F(r)) & G(F(s))",
      G(F(p)) && G(F(q)) && G(F(r)) && G(F(s))},
    {"G(!p) & (p U q)",
      G(!p) && U(p,q)},
    {"F(p) & G(!p)",
      F(p) && G(!p)},
    {"G(F(p)) & G(!p)",
      G(F(p)) && G(!p)},
  };

  std::cout << std::left
            << std::setw(52) << "Formula"
            << std::setw(12) << "Seq(ms)"
            << std::setw(12) << "Par2(ms)"
            << std::setw(12) << "Par4(ms)"
            << std::setw(10) << "Speedup2"
            << std::setw(10) << "Speedup4"
            << "\n"
            << std::string(108, '-') << "\n";

  for(auto& c : cases) {
    double seq  = time_seq(xi, c.f);
    double par2 = time_par(xi, c.f, 2);
    double par4 = time_par(xi, c.f, 4);

    std::string label = c.label.size() > 49 ? c.label.substr(0,46)+"..." : c.label;

    std::cout << std::left  << std::setw(52) << label
              << std::fixed << std::setprecision(2)
              << std::setw(12) << seq
              << std::setw(12) << par2
              << std::setw(12) << par4
              << std::setw(10) << (par2 > 0 ? seq/par2 : 0.0)
              << std::setw(10) << (par4 > 0 ? seq/par4 : 0.0)
              << "\n";
  }

  return 0;
}

