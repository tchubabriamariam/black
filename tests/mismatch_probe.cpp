//
// Probe: what wrong value does the naive parallel solver return on a MISMATCH?
//
// Runs solve_parallel() many times on a formula whose correct answer we know
// (from the sequential solver) and tallies how often it comes back SAT, UNSAT,
// or UNDEF. This tells us whether a mismatch is a wrong definite answer
// (e.g. UNSAT for a SAT formula) or an "undecided / stopped early" result.
//
#include <black/solver/solver.hpp>
#include <black/logic/logic.hpp>

#include <iostream>
#include <limits>
#include <cstdlib>

using namespace black;

static int code(tribool t) { if(t == true) return 1; if(t == false) return 0; return -1; }
static const char* show(int c) { return c == 1 ? "SAT" : c == 0 ? "UNSAT" : "UNDEF"; }

int main(int argc, char** argv) {
  size_t threads = argc > 1 ? size_t(std::atoi(argv[1])) : 24;
  int    iters   = argc > 2 ? std::atoi(argv[2]) : 300;

  alphabet sigma; scope xi{sigma};
  auto p = sigma.proposition("p");
  auto q = sigma.proposition("q");
  auto r = sigma.proposition("r");
  formula f = G(F(p)) && G(F(q)) && G(F(r)); // this formula is SAT

  int ref;
  { black::solver s; ref = code(s.solve(xi, f)); }
  std::cout << "reference (sequential) = " << show(ref) << "\n";
  std::cout << "naive solve_parallel, " << iters << " runs at "
            << threads << " threads:\n";

  int sat = 0, unsat = 0, undef = 0, mism = 0;
  for(int i = 0; i < iters; ++i) {
    black::solver s;
    int res = code(s.solve_parallel(
      xi, f, false, std::numeric_limits<size_t>::max(), {}, false, threads));
    if(res == 1) ++sat; else if(res == 0) ++unsat; else ++undef;
    if(res != ref) ++mism;
  }
  std::cout << "  SAT=" << sat << "  UNSAT=" << unsat << "  UNDEF=" << undef
            << "   -> mismatches=" << mism << "\n";
  return 0;
}
