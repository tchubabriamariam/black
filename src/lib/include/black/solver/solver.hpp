//
// BLACK - Bounded Ltl sAtisfiability ChecKer
//
// (C) 2019 Luca Geatti
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#ifndef BLACK_SOLVER_HPP
#define BLACK_SOLVER_HPP

#include <black/support/common.hpp>
#include <black/logic/logic.hpp>
#include <black/support/tribool.hpp>

#include <vector>
#include <variant>
#include <utility>
#include <limits>
#include <unordered_set>
#include <string>
#include <numeric>
#include <chrono>
#include <memory>

namespace black_internal::solver {

  using namespace black;

  // main solver class
  class BLACK_EXPORT solver 
  {
    public:
      friend class model;

      static bool
      check_syntax(formula f, std::function<void(std::string)> const& err);

      solver();
      ~solver();

      solver(solver const&) = delete;
      solver &operator=(solver const&) = delete;
      solver(solver &&);
      solver &operator=(solver &&);

      tribool solve(
        scope const& xi,
        formula f,
        bool finite = false,
        size_t k_max = std::numeric_limits<size_t>::max(),
        std::optional<std::chrono::seconds> timeout = {},
        bool semi_decision = false
      );

      // Parallel version of solve(). num_threads controls parallelism.
      //
      // Naive variant: every worker owns its own encoder and re-builds the
      // k-unraveling / prune formulas from scratch. Kept for benchmarking
      // against solve_parallel_shared().
      tribool solve_parallel(
        scope const& xi,
        formula f,
        bool finite = false,
        size_t k_max = std::numeric_limits<size_t>::max(),
        std::optional<std::chrono::seconds> timeout = {},
        bool semi_decision = false,
        size_t num_threads = 4
      );

      // Parallel version with a shared, read-only encoding. A single encoder
      // owns all mutations of the alphabet and builds each k-indexed formula
      // exactly once (under a lock); worker threads only READ those immutable
      // formulas to feed their own independent SAT solvers. This removes both
      // the redundant re-encoding and the data race on the shared alphabet
      // that solve_parallel() has.
      //
      // If pin_performance_cores is true, worker threads request a high QoS
      // class so the OS schedules them on performance ("P") cores rather than
      // efficiency ("E") cores (relevant on Apple silicon / big.LITTLE).
      tribool solve_parallel_shared(
        scope const& xi,
        formula f,
        bool finite = false,
        size_t k_max = std::numeric_limits<size_t>::max(),
        std::optional<std::chrono::seconds> timeout = {},
        bool semi_decision = false,
        size_t num_threads = 4,
        bool pin_performance_cores = false
      );

      // Instrumentation recorded by the most recent solve_parallel() call, for
      // the software-threads study: how the software threads behaved and how
      // much redundant work they performed relative to what was actually needed.
      struct parallel_counters {
        size_t requested_threads    = 0; // threads asked for
        size_t launched_threads     = 0; // threads that actually started running
        size_t aborted_threads      = 0; // "stuck" sense 1: gave up after another decided
        size_t lock_waits           = 0; // "stuck" sense 2: had to wait for the shared lock
        size_t unravelings_needed   = 0; // k-unravelings on the path to the answer (0..K*)
        size_t unravelings_computed = 0; // k-unravelings actually built across all threads
        double redundancy           = 0.0; // computed / needed
      };

      // Counters from the last solve_parallel() call.
      parallel_counters last_parallel_counters() const;

      // Result of analyze_parallelism(): a hardware-independent estimate of how
      // much a naive branch-parallelisation of solve() can help, derived from a
      // single instrumented sequential run (no extra cores required).
      struct parallelism_report {
        tribool result = tribool::undef;
        // k at which the sequential algorithm reached a definitive answer (K*).
        size_t decisive_k = 0;
        // Sum over k=0..K* of (encode+assert time + SAT-check time). This is
        // what the sequential algorithm pays.
        double seq_total_ms = 0.0;
        // Critical path under an idealised speculative scheme with unbounded
        // cores: one solver per k built from scratch, all running at once. The
        // deciding solver must build the whole spine of unravelings 0..K* and
        // then perform its own check at K*; every other check overlaps.
        //   ideal = sum(assert_ms[0..K*]) + check_ms[K*]
        double ideal_parallel_ms = 0.0;
        // seq_total_ms / ideal_parallel_ms: upper bound on achievable speedup.
        double speedup_upper_bound = 0.0;
        // Per-k breakdown (indices 0..K*).
        std::vector<double> assert_ms;
        std::vector<double> check_ms;
      };

      // Runs solve() single-threaded while timing each k, then computes the
      // theoretical parallelism available (see parallelism_report). Hardware
      // independent: it measures the shape of the computation, not the machine.
      parallelism_report analyze_parallelism(
        scope const& xi,
        formula f,
        bool finite = false,
        size_t k_max = std::numeric_limits<size_t>::max(),
        bool semi_decision = false
      );

      // Number of performance ("P") cores available. On Apple silicon this
      // queries hw.perflevel0.logicalcpu; elsewhere it falls back to
      // std::thread::hardware_concurrency().
      static size_t performance_core_count();

      tribool is_valid(
        scope const& xi,
        formula f,
        bool finite = false,
        size_t k_max = std::numeric_limits<size_t>::max(),
        std::optional<std::chrono::seconds> timeout = {},
        bool semi_decision = false
      );

      void interrupt();

      std::optional<class model> model() const;

      size_t last_bound() const;

      void set_sat_backend(std::string name);

      std::string sat_backend() const;

      struct trace_t {
        enum type_t {
          stage,
          nnf,
          unrav,
          empty,
          loop,
          prune
        };

        scope const *xi;
        type_t type;
        std::variant<size_t, logic::formula> data;
      };

      void set_tracer(std::function<void(trace_t)> const&tracer);

    private:
      struct _solver_t;
      std::shared_ptr<_solver_t> _data;

  }; // end class solver

  class BLACK_EXPORT model
  {
    public:
      size_t size() const;
      size_t loop() const;
      tribool value(proposition a, size_t t) const;
      tribool value(atom a, size_t t) const;
      tribool value(equality a, size_t t) const;
      tribool value(comparison a, size_t t) const;
    private:
      friend class solver;
      model(solver const&s) : _solver{s} { }

      solver const&_solver;
  };

} // end namespace black_internal

namespace black {
  using black_internal::solver::solver;
  using black_internal::solver::model;
}

#endif // SOLVER_HPP

