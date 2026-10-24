//
// BLACK - Bounded Ltl sAtisfiability ChecKer
//
// (C) 2019 Luca Geatti
// (C) 2019 Nicola Gigante
// (C) 2020 Gabriele Venturato
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

#include <black/support/config.hpp>
#include <black/support/range.hpp>
#include <black/solver/solver.hpp>
#include <black/solver/encoding.hpp>
#include <black/sat/solver.hpp>

#include <numeric>
#include <atomic>
#include <memory>
#include <future>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <cstdio>

#include <pthread.h>

#if defined(__APPLE__)
  #include <pthread/qos.h>
  #include <sys/sysctl.h>
#endif

namespace black_internal::solver
{
  /*
   * Private implementation of the solver class.
   */
  struct solver::_solver_t : std::enable_shared_from_this<solver::_solver_t>
  {
    bool model = false;
    std::optional<encoder::encoder> enc;
    alphabet *sigma = nullptr;
    size_t model_size = 0;
    size_t last_bound = 0;
    std::unique_ptr<black::sat::solver> sat;
    std::string sat_backend = BLACK_DEFAULT_BACKEND;
    std::atomic<bool> interrupt_flag = false;
    std::function<void(trace_t)> tracer = [](trace_t){};
    solver::parallel_counters last_counters;
    solver::cube_counters last_cube_counters;

    void trace(size_t k);
    void trace(trace_t::type_t, scope const&, logic::formula);

    tribool solve(
      scope const& xi, logic::formula f,
      bool finite, size_t k_max, std::optional<std::chrono::seconds> timeout,
      bool semi_decision
    );

    tribool solve_parallel(
      scope const& xi, logic::formula f,
      bool finite, size_t k_max, std::optional<std::chrono::seconds> timeout,
      bool semi_decision, size_t num_threads
    );

    tribool solve_parallel_shared(
      scope const& xi, logic::formula f,
      bool finite, size_t k_max, std::optional<std::chrono::seconds> timeout,
      bool semi_decision, size_t num_threads, bool pin_performance_cores
    );

    tribool solve_cube_and_conquer(
      scope const& xi, logic::formula f,
      bool finite, size_t k_max, std::optional<std::chrono::seconds> timeout,
      bool semi_decision, size_t num_threads, size_t cube_depth,
      bool full_check_worker, bool pin_performance_cores
    );

    solver::parallelism_report analyze_parallelism(
      scope const& xi, logic::formula f,
      bool finite, size_t k_max, bool semi_decision
    );

    void interrupt();
  };

  solver::solver() : _data{std::make_unique<_solver_t>()} { }
  solver::~solver() = default;
  solver::solver(solver &&) = default;
  solver &solver::operator=(solver &&) = default;

  tribool solver::solve(
    scope const& xi, logic::formula f,
    bool finite, size_t k_max, std::optional<std::chrono::seconds> timeout,
    bool semi_decision
  ) {
    return _data->solve(xi, f, finite, k_max, timeout, semi_decision);
  }

  tribool solver::solve_parallel(
    scope const& xi, logic::formula f,
    bool finite, size_t k_max, std::optional<std::chrono::seconds> timeout,
    bool semi_decision, size_t num_threads
  ) {
    return _data->solve_parallel(
      xi, f, finite, k_max, timeout, semi_decision, num_threads
    );
  }

  tribool solver::solve_parallel_shared(
    scope const& xi, logic::formula f,
    bool finite, size_t k_max, std::optional<std::chrono::seconds> timeout,
    bool semi_decision, size_t num_threads, bool pin_performance_cores
  ) {
    return _data->solve_parallel_shared(
      xi, f, finite, k_max, timeout, semi_decision,
      num_threads, pin_performance_cores
    );
  }

  tribool solver::solve_cube_and_conquer(
    scope const& xi, logic::formula f,
    bool finite, size_t k_max, std::optional<std::chrono::seconds> timeout,
    bool semi_decision, size_t num_threads, size_t cube_depth,
    bool full_check_worker, bool pin_performance_cores
  ) {
    return _data->solve_cube_and_conquer(
      xi, f, finite, k_max, timeout, semi_decision,
      num_threads, cube_depth, full_check_worker, pin_performance_cores
    );
  }

  solver::parallelism_report solver::analyze_parallelism(
    scope const& xi, logic::formula f,
    bool finite, size_t k_max, bool semi_decision
  ) {
    return _data->analyze_parallelism(xi, f, finite, k_max, semi_decision);
  }

  tribool solver::is_valid(
    scope const& xi, logic::formula f,
    bool finite, size_t k_max, std::optional<std::chrono::seconds> timeout,
    bool semi_decision
  ) {
    tribool res = _data->solve(xi, !f, finite, k_max, timeout, semi_decision);
    if(res == true)  return false;
    if(res == false) return true;
    return tribool::undef;
  }

  void solver::interrupt() { _data->interrupt(); }

  std::optional<model> solver::model() const {
    if(!_data->model) return {};
    return {{*this}};
  }

  size_t solver::last_bound() const { return _data->last_bound; }

  solver::parallel_counters solver::last_parallel_counters() const {
    return _data->last_counters;
  }

  solver::cube_counters solver::last_cube_counters() const {
    return _data->last_cube_counters;
  }

  void solver::set_sat_backend(std::string name) {
    _data->sat_backend = std::move(name);
  }

  std::string solver::sat_backend() const { return _data->sat_backend; }

  void solver::set_tracer(std::function<void(trace_t)> const&tracer) {
    _data->tracer = tracer;
  }

  size_t model::size() const { return _solver._data->model_size; }

  size_t model::loop() const {
    using black_internal::encoder::encoder;
    black_assert(size() > 0);
    size_t k = size() - 1;
    for(size_t l = 0; l < k; ++l) {
      proposition loop_prop = encoder::loop_prop(_solver._data->sigma, l, k);
      if(_solver._data->sat->value(loop_prop) == true)
        return l + 1;
    }
    return size();
  }

  tribool model::value(proposition a, size_t t) const {
    using black_internal::encoder::encoder;
    return _solver._data->sat->value(encoder::stepped(a, t));
  }

  tribool model::value(atom a, size_t t) const {
    if(!_solver._data->enc.has_value()) return tribool::undef;
    return _solver._data->sat->value(_solver._data->enc->stepped(a, t));
  }

  tribool model::value(equality e, size_t t) const {
    if(!_solver._data->enc.has_value()) return tribool::undef;
    return _solver._data->sat->value(_solver._data->enc->stepped(e, t));
  }

  tribool model::value(comparison c, size_t t) const {
    if(!_solver._data->enc.has_value()) return tribool::undef;
    return _solver._data->sat->value(_solver._data->enc->stepped(c, t));
  }

  void solver::_solver_t::trace(size_t k) {
    tracer({nullptr, trace_t::stage, {k}});
  }

  void solver::_solver_t::trace(
    trace_t::type_t type, scope const& xi, logic::formula f
  ) {
    tracer({&xi, type, {f}});
  }

  /*
   * Original sequential algorithm.
   */
  tribool solver::_solver_t::solve(
    scope const& s, logic::formula f, bool finite,
    size_t k_max, std::optional<std::chrono::seconds> timeout,
    bool semi_decision
  ) {
    scope xi = chain(s);
    enc = encoder::encoder{f, xi, finite};
    sat = black::sat::solver::get_solver(sat_backend, xi);
    trace(trace_t::nnf, xi, enc->get_formula());
    sigma = f.sigma();
    model = false;
    model_size = 0;
    last_bound = 0;

    if(timeout)
      std::thread([timeout, self = this->shared_from_this()]() {
        std::this_thread::sleep_for(*timeout);
        self->interrupt();
      }).detach();

    for(size_t k = 0; !interrupt_flag && k <= k_max; last_bound = k++) {
      trace(k);

      auto unrav = enc->k_unraveling(k);
      trace(trace_t::unrav, xi, unrav);
      sat->assert_formula(unrav);
      if(tribool res = sat->is_sat(); !res)
        return res;

      auto empty = enc->k_empty(k);
      auto loop  = enc->k_loop(k);
      trace(trace_t::empty, xi, empty);
      trace(trace_t::loop,  xi, loop);
      if(sat->is_sat_with(empty || loop)) {
        model_size = k + 1;
        model = true;
        return true;
      }

      if(!semi_decision) {
        auto prune = enc->prune(k);
        trace(trace_t::prune, xi, prune);
        sat->assert_formula(!prune);
        if(tribool res = sat->is_sat(); !res)
          return res;
      }
    }
    interrupt_flag = false;
    return tribool::undef;
  }

  /*
   * Parallel algorithm.
   *
   * We spawn `num_threads` worker threads. Thread i owns a completely
   * independent encoder and SAT solver instance — zero shared mutable
   * state between workers.
   *
   * Thread i is responsible for checking k = i, i+num_threads,
   * i+2*num_threads, ... but because the encoding is cumulative (each
   * step adds constraints on top of all previous ones) every thread must
   * still build unravelings for ALL k from 0 up to its current target.
   * It just doesn't do the SAT/EMPTY/LOOP/PRUNE checks on intermediate
   * values — those belong to the threads assigned to those k values.
   *
   * A shared atomic `shared_result` carries -1 (no answer yet), 1 (SAT),
   * or 0 (UNSAT). The first thread to reach a definitive answer wins;
   * all others exit their loops on the next iteration check.
   *
   * The winning SAT thread moves its encoder and SAT solver into
   * _solver_t so that model extraction works exactly as in the
   * sequential case.
   */
  tribool solver::_solver_t::solve_parallel(
    scope const& s, logic::formula f, bool finite,
    size_t k_max, std::optional<std::chrono::seconds> timeout,
    bool semi_decision, size_t num_threads
  ) {
    if(num_threads <= 1)
      return solve(s, f, finite, k_max, timeout, semi_decision);

    sigma = f.sigma();
    model = false;
    model_size = 0;
    last_bound = 0;

    // -1 = undecided, 0 = UNSAT, 1 = SAT
    std::atomic<int> shared_result{-1};
    std::atomic<size_t> shared_last_bound{0};

    // --- Instrumentation counters (software-threads study) -------------------
    // cnt_threads_launched   : how many worker (software) threads actually began
    //                          running (vs how many we asked for).
    // cnt_unravelings_computed: total number of k-unravelings computed across ALL
    //                          threads. Compared afterwards against the number
    //                          NEEDED on the path to the answer (k = 0..K*), this
    //                          exposes how much redundant/speculative work the
    //                          naive per-thread encoding does.
    // cnt_threads_aborted    : threads that, while working, noticed another thread
    //                          had already decided and gave up ("got stuck"/wasted).
    std::atomic<size_t> cnt_threads_launched{0};
    std::atomic<size_t> cnt_unravelings_computed{0};
    std::atomic<size_t> cnt_threads_aborted{0};

    std::mutex winner_mutex;
    std::optional<encoder::encoder> winning_enc;
    std::unique_ptr<black::sat::solver> winning_sat;
    size_t winning_model_size = 0;

    if(timeout)
      std::thread([timeout, self = this->shared_from_this()]() {
        std::this_thread::sleep_for(*timeout);
        self->interrupt();
      }).detach();

    // Worker: thread `tid` checks k = tid, tid+N, tid+2N, ...
    // For every k it handles, it first builds all intermediate unravelings
    // from the previous handled k+1 up to the current k-1 (gap fill),
    // then does the full check on its assigned k.

    auto worker = [&](size_t tid) {
      cnt_threads_launched.fetch_add(1, std::memory_order_relaxed);
      scope xi = chain(s);
      encoder::encoder local_enc{f, xi, finite};
      auto local_sat = black::sat::solver::get_solver(sat_backend, xi);

      // Every k-unraveling this thread actually computes is counted here, so we
      // can later compare the total against the number truly needed (0..K*).
      auto unravel = [&](size_t kk) {
        cnt_unravelings_computed.fetch_add(1, std::memory_order_relaxed);
        return local_enc.k_unraveling(kk);
      };

      // Fill in k = 0 .. tid-1 (catch-up, no SAT checks)
      for(size_t k = 0; k < tid; ++k) {
        if(shared_result.load() != -1 || interrupt_flag) {
          cnt_threads_aborted.fetch_add(1, std::memory_order_relaxed);
          return;
        }
        local_sat->assert_formula(unravel(k));
        if(!semi_decision)
          local_sat->assert_formula(!local_enc.prune(k));
      }

      // Main loop over this thread's assigned k values
      bool first = true;
      for(size_t k = tid;
          !interrupt_flag && shared_result.load() == -1 && k <= k_max;
          k += num_threads)
      {
        // Fill gap: assert unravelings for k values between our previous
        // assigned k and the current one (other threads' k values).
        if(!first) {
          size_t gap_start = k - num_threads + 1;
          size_t gap_end   = k; // exclusive — we handle k ourselves below
          for(size_t j = gap_start; j < gap_end; ++j) {
            if(shared_result.load() != -1 || interrupt_flag) {
              cnt_threads_aborted.fetch_add(1, std::memory_order_relaxed);
              return;
            }
            local_sat->assert_formula(unravel(j));
            if(!semi_decision)
              local_sat->assert_formula(!local_enc.prune(j));
          }
        }
        first = false;

        if(shared_result.load() != -1 || interrupt_flag) return;

        // Full check for our assigned k

        // 1. Assert k-unraveling and check satisfiability
        local_sat->assert_formula(unravel(k));

        // Update shared last_bound (best-effort)
        size_t prev = shared_last_bound.load();
        while(k > prev &&
              !shared_last_bound.compare_exchange_weak(prev, k))
          prev = shared_last_bound.load();

        if(tribool res = local_sat->is_sat(); !res) {
          int expected = -1;
          shared_result.compare_exchange_strong(expected, 0);
          return;
        }

        if(shared_result.load() != -1 || interrupt_flag) return;

        // 2. Check EMPTY or LOOP
        auto empty = local_enc.k_empty(k);
        auto loop  = local_enc.k_loop(k);
        if(local_sat->is_sat_with(empty || loop)) {
          int expected = -1;
          if(shared_result.compare_exchange_strong(expected, 1)) {
            std::lock_guard<std::mutex> lock(winner_mutex);
            winning_enc        = std::move(local_enc);
            winning_sat        = std::move(local_sat);
            winning_model_size = k + 1;
          }
          return;
        }

        if(shared_result.load() != -1 || interrupt_flag) return;

        // 3. Assert negation of PRUNE and check
        if(!semi_decision) {
          local_sat->assert_formula(!local_enc.prune(k));
          if(tribool res = local_sat->is_sat(); !res) {
            int expected = -1;
            shared_result.compare_exchange_strong(expected, 0);
            return;
          }
        }
      }
    }; // end worker lambda

    // Snapshot the alphabet's lock-contention count so we can report how many
    // times a worker had to wait for the shared uniquing lock during this run.
    size_t lock_waits_before = sigma->lock_contention_count();

    // Launch all threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(size_t i = 0; i < num_threads; ++i)
      threads.emplace_back(worker, i);

    for(auto& t : threads)
      t.join();

    interrupt_flag = false;
    last_bound = shared_last_bound.load();

    // Record the instrumentation counters. `needed` is the number of
    // k-unravelings on the path to the answer (k = 0..K*); `computed` is the
    // total actually performed across all threads. Their ratio is the redundant
    // work factor the naive per-thread encoding pays.
    {
      size_t needed   = last_bound + 1;
      size_t computed = cnt_unravelings_computed.load();
      size_t lock_waits = sigma->lock_contention_count() - lock_waits_before;
      last_counters = solver::parallel_counters{
        num_threads,
        cnt_threads_launched.load(),
        cnt_threads_aborted.load(),
        lock_waits,
        needed,
        computed,
        needed ? double(computed) / double(needed) : 0.0
      };

      if(std::getenv("BLACK_PARALLEL_COUNTERS"))
        std::fprintf(stderr,
          "[counters] requested_threads=%zu launched=%zu | stuck: aborted=%zu "
          "lock_waits=%zu | unravelings needed(0..K*)=%zu computed=%zu "
          "redundancy=%.2fx | K*~%zu\n",
          num_threads, last_counters.launched_threads,
          last_counters.aborted_threads, last_counters.lock_waits,
          needed, computed, last_counters.redundancy, last_bound);
    }

    int result = shared_result.load();

    if(result == 1) {
      std::lock_guard<std::mutex> lock(winner_mutex);
      enc        = std::move(winning_enc);
      sat        = std::move(winning_sat);
      model_size = winning_model_size;
      model      = true;
      return true;
    }

    if(result == 0)
      return false;

    return tribool::undef;
  }

  namespace {
    //
    // Thread-safe, lazily-populated cache of the k-indexed encoding formulas.
    //
    // A single `encoder` owns every mutation of the shared alphabet. Because
    // the encoding functions (k_unraveling, prune, k_empty, k_loop) all intern
    // nodes into that alphabet — which is NOT thread-safe — every call goes
    // through one mutex. Each formula is built exactly once and then handed out
    // read-only to any number of worker threads. Reading an already-built
    // formula (e.g. feeding it to a SAT backend) touches only the immutable,
    // pointer-stable nodes, never the allocator's hash map, so it needs no
    // lock.
    //
    // This is the concrete realisation of "precompute the encoding once and
    // share it read-only across threads": the redundant re-encoding of the
    // naive version disappears, and so does its data race on the alphabet.
    //
    struct shared_encoding {
      encoder::encoder enc;
      bool semi_decision;

      std::mutex mutex;
      std::vector<logic::formula> _unrav; // _unrav[k] == k_unraveling(k)
      std::vector<logic::formula> _prune; // _prune[k] == !prune(k)

      shared_encoding(logic::formula f, scope &xi, bool finite, bool semi)
        : enc{f, xi, finite}, semi_decision{semi} { }

      // k-unraveling for k, built in order and memoised.
      logic::formula unraveling(size_t k) {
        std::lock_guard<std::mutex> lock(mutex);
        while(_unrav.size() <= k)
          _unrav.push_back(enc.k_unraveling(_unrav.size()));
        return _unrav[k];
      }

      // Negated PRUNE constraint for k, built in order and memoised.
      logic::formula prune_neg(size_t k) {
        std::lock_guard<std::mutex> lock(mutex);
        while(_prune.size() <= k)
          _prune.push_back(!enc.prune(_prune.size()));
        return _prune[k];
      }

      // EMPTY_k || LOOP_k. Only ever queried once per k (by the single thread
      // that owns k), so it is not memoised — but the disjunction itself is
      // built under the lock because it interns a new node.
      logic::formula empty_or_loop(size_t k) {
        std::lock_guard<std::mutex> lock(mutex);
        return enc.k_empty(k) || enc.k_loop(k);
      }
    };

#if defined(__APPLE__)
    void request_performance_core() {
      // On Apple silicon the scheduler places USER_INTERACTIVE-QoS threads on
      // performance ("P") cores and leaves the efficiency ("E") cores for
      // background work. There is no hard affinity API, but QoS is the
      // documented lever.
      pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
    }
#else
    void request_performance_core() { }
#endif
  } // anonymous namespace

  size_t solver::performance_core_count() {
#if defined(__APPLE__)
    int n = 0;
    size_t sz = sizeof(n);
    if(sysctlbyname("hw.perflevel0.logicalcpu", &n, &sz, nullptr, 0) == 0
       && n > 0)
      return size_t(n);
#endif
    unsigned hc = std::thread::hardware_concurrency();
    return hc > 0 ? size_t(hc) : size_t(1);
  }

  /*
   * Parallel algorithm with a shared, read-only encoding.
   *
   * Identical racing strategy to solve_parallel() — thread i checks
   * k = i, i+N, i+2N, ... and the first to a definitive answer wins — but the
   * k-unraveling / prune / empty / loop formulas are built ONCE by a single
   * shared encoder and only read by the workers. Each worker still owns its
   * own SAT solver, into which it asserts the (shared) formulas cumulatively.
   */
  tribool solver::_solver_t::solve_parallel_shared(
    scope const& s, logic::formula f, bool finite,
    size_t k_max, std::optional<std::chrono::seconds> timeout,
    bool semi_decision, size_t num_threads, bool pin_performance_cores
  ) {
    if(num_threads <= 1)
      return solve(s, f, finite, k_max, timeout, semi_decision);

    sigma = f.sigma();
    model = false;
    model_size = 0;
    last_bound = 0;

    // -1 = undecided, 0 = UNSAT, 1 = SAT
    std::atomic<int> shared_result{-1};
    std::atomic<size_t> shared_last_bound{0};

    // The single shared encoder. Its scope must outlive the encoder, so keep
    // it here on the stack of the calling thread.
    scope enc_scope = chain(s);
    shared_encoding shared{f, enc_scope, finite, semi_decision};

    std::mutex winner_mutex;
    std::unique_ptr<black::sat::solver> winning_sat;
    size_t winning_model_size = 0;

    // Build each worker's own scope up front, on this thread, so we never touch
    // the parent scope `s` concurrently from the workers.
    std::vector<scope> worker_scopes;
    worker_scopes.reserve(num_threads);
    for(size_t i = 0; i < num_threads; ++i)
      worker_scopes.push_back(chain(s));

    if(timeout)
      std::thread([timeout, self = this->shared_from_this()]() {
        std::this_thread::sleep_for(*timeout);
        self->interrupt();
      }).detach();

    auto worker = [&](size_t tid) {
      if(pin_performance_cores)
        request_performance_core();

      scope xi = std::move(worker_scopes[tid]);
      auto local_sat = black::sat::solver::get_solver(sat_backend, xi);

      // asserted_upto is exclusive: unravelings (and prunes) for [0, asserted_upto)
      // have already been asserted into this worker's solver.
      size_t asserted_upto = 0;

      for(size_t k = tid;
          !interrupt_flag && shared_result.load() == -1 && k <= k_max;
          k += num_threads)
      {
        // Gap fill: assert the unravelings/prunes for every k we skip, so our
        // solver holds the full cumulative encoding up to k. These are the
        // k values owned by other threads — we do no SAT checks on them.
        for(size_t j = asserted_upto; j < k; ++j) {
          if(shared_result.load() != -1 || interrupt_flag) return;
          local_sat->assert_formula(shared.unraveling(j));
          if(!semi_decision)
            local_sat->assert_formula(shared.prune_neg(j));
        }
        asserted_upto = k;

        if(shared_result.load() != -1 || interrupt_flag) return;

        // Full check for our assigned k, mirroring the sequential order.

        // 1. Assert k-unraveling and check satisfiability.
        local_sat->assert_formula(shared.unraveling(k));

        size_t prev = shared_last_bound.load();
        while(k > prev && !shared_last_bound.compare_exchange_weak(prev, k))
          prev = shared_last_bound.load();

        if(tribool res = local_sat->is_sat(); !res) {
          int expected = -1;
          shared_result.compare_exchange_strong(expected, 0);
          return;
        }

        if(shared_result.load() != -1 || interrupt_flag) return;

        // 2. Check EMPTY or LOOP.
        if(local_sat->is_sat_with(shared.empty_or_loop(k))) {
          int expected = -1;
          if(shared_result.compare_exchange_strong(expected, 1)) {
            std::lock_guard<std::mutex> lock(winner_mutex);
            winning_sat        = std::move(local_sat);
            winning_model_size = k + 1;
          }
          return;
        }

        if(shared_result.load() != -1 || interrupt_flag) return;

        // 3. Assert negation of PRUNE and check.
        if(!semi_decision) {
          local_sat->assert_formula(shared.prune_neg(k));
          if(tribool res = local_sat->is_sat(); !res) {
            int expected = -1;
            shared_result.compare_exchange_strong(expected, 0);
            return;
          }
        }
        asserted_upto = k + 1;
      }
    }; // end worker lambda

    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(size_t i = 0; i < num_threads; ++i)
      threads.emplace_back(worker, i);

    for(auto &t : threads)
      t.join();

    interrupt_flag = false;
    last_bound = shared_last_bound.load();

    int result = shared_result.load();

    if(result == 1) {
      std::lock_guard<std::mutex> lock(winner_mutex);
      enc        = std::move(shared.enc);
      sat        = std::move(winning_sat);
      model_size = winning_model_size;
      model      = true;
      return true;
    }

    if(result == 0)
      return false;

    return tribool::undef;
  }

  namespace {
    //
    // Fixed pool of worker threads that all execute the same job together.
    // run() hands the job to every worker and blocks until all of them have
    // finished it, so consecutive jobs are separated by a barrier. This keeps
    // the threads (and the SAT solvers they own) alive across all the checks
    // of a cube-and-conquer run instead of respawning them per check.
    //
    struct worker_pool {
      // Secondary threads get a small default stack (512KB on macOS), and both
      // the formula-to-backend translation and the backends themselves (e.g.
      // Z3's model construction) recurse on deep formulas, so the workers are
      // created with the stack size of a main thread.
      static constexpr size_t stack_size = size_t(64) << 20;

      std::vector<pthread_t> threads;
      std::mutex mutex;
      std::condition_variable job_cv;
      std::condition_variable done_cv;
      std::function<void(size_t)> job;
      size_t generation = 0;
      size_t pending = 0;
      bool stop = false;

      struct start_t {
        worker_pool *pool;
        size_t tid;
      };

      explicit worker_pool(size_t n) {
        pthread_attr_t attr;
        pthread_attr_init(&attr);
        pthread_attr_setstacksize(&attr, stack_size);
        threads.resize(n);
        for(size_t tid = 0; tid < n; ++tid) {
          [[maybe_unused]] int err = pthread_create(&threads[tid], &attr, [](void *arg) -> void* {
            std::unique_ptr<start_t> start{static_cast<start_t *>(arg)};
            start->pool->loop(start->tid);
            return nullptr;
          }, new start_t{this, tid});
          black_assert(err == 0);
        }
        pthread_attr_destroy(&attr);
      }

      ~worker_pool() {
        {
          std::lock_guard<std::mutex> lock(mutex);
          stop = true;
        }
        job_cv.notify_all();
        for(auto t : threads)
          pthread_join(t, nullptr);
      }

      void run(std::function<void(size_t)> j) {
        std::unique_lock<std::mutex> lock(mutex);
        job = std::move(j);
        pending = threads.size();
        ++generation;
        job_cv.notify_all();
        done_cv.wait(lock, [&] { return pending == 0; });
      }

      void loop(size_t tid) {
        size_t seen = 0;
        while(true) {
          std::function<void(size_t)> j;
          {
            std::unique_lock<std::mutex> lock(mutex);
            job_cv.wait(lock, [&] { return stop || generation != seen; });
            if(stop)
              return;
            seen = generation;
            j = job;
          }
          j(tid);
          {
            std::lock_guard<std::mutex> lock(mutex);
            if(--pending == 0)
              done_cv.notify_one();
          }
        }
      }
    };

    //
    // Chooses the split variables for cube-and-conquer.
    //
    // Real cube-and-conquer solvers pick split variables with a look-ahead
    // solver. Running a look-ahead through the SAT backend would cost two SAT
    // calls per candidate, so we use the cheap static proxy of the same idea:
    // the variables that occur in the most places of the encoding asserted so
    // far are the ones whose assignment simplifies the most constraints, so
    // splitting on them gives the most constrained (and most balanced)
    // sub-problems. Each newly asserted formula is scanned once; shared
    // sub-formulas are visited only once. Variables asserted as top-level
    // units are skipped, since one of their two cubes would be trivially UNSAT.
    //
    struct split_heuristic {
      std::unordered_map<proposition, size_t> occurrences;
      std::unordered_set<proposition> units;
      std::unordered_set<logic::formula> visited;

      void add(logic::formula f) {
        mark_units(f);

        std::vector<logic::formula> stack{f};
        while(!stack.empty()) {
          logic::formula g = stack.back();
          stack.pop_back();
          if(!visited.insert(g).second)
            continue;

          auto child = [&](logic::formula c) {
            if(auto p = c.to<proposition>(); p)
              ++occurrences[*p];
            stack.push_back(c);
          };

          g.match(
            [&](negation, auto arg) { child(arg); },
            [&](conjunction c) {
              for(logic::formula op : operands(c))
                child(op);
            },
            [&](disjunction c) {
              for(logic::formula op : operands(c))
                child(op);
            },
            [&](implication, auto left, auto right) {
              child(left);
              child(right);
            },
            [&](iff, auto left, auto right) {
              child(left);
              child(right);
            },
            [](otherwise) { }
          );
        }
      }

      void mark_units(logic::formula f) {
        f.match(
          [&](proposition p) { units.insert(p); },
          [&](negation, auto arg) {
            if(auto p = arg.template to<proposition>(); p)
              units.insert(*p);
          },
          [&](conjunction c) {
            for(logic::formula op : operands(c))
              mark_units(op);
          },
          [](otherwise) { }
        );
      }

      // The (at most) d most frequently occurring non-unit variables.
      std::vector<proposition> pick(size_t d) const {
        std::vector<std::pair<size_t, proposition>> ranked;
        ranked.reserve(occurrences.size());
        for(auto [p, n] : occurrences)
          if(!units.contains(p))
            ranked.push_back({n, p});

        d = std::min(d, ranked.size());
        std::partial_sort(
          ranked.begin(), ranked.begin() + long(d), ranked.end(),
          [](auto const& a, auto const& b) {
            if(a.first != b.first)
              return a.first > b.first;
            return a.second.unique_id() < b.second.unique_id();
          }
        );

        std::vector<proposition> result;
        for(size_t i = 0; i < d; ++i)
          result.push_back(ranked[i].second);
        return result;
      }
    };

    // All 2^d cubes over the given split variables: cube i sets variable j to
    // true iff bit j of i is set. Together they partition the search space.
    // With no split variables there is a single, empty cube (no assumption).
    std::vector<std::optional<logic::formula>>
    make_cubes(std::vector<proposition> const& vars) {
      std::vector<std::optional<logic::formula>> cubes;
      size_t n = size_t(1) << vars.size();
      cubes.reserve(n);
      for(size_t i = 0; i < n; ++i) {
        std::optional<logic::formula> cube;
        for(size_t j = 0; j < vars.size(); ++j) {
          logic::formula lit = vars[j];
          if(!(i & (size_t(1) << j)))
            lit = !vars[j];
          cube = cube ? logic::formula{*cube && lit} : lit;
        }
        cubes.push_back(cube);
      }
      return cubes;
    }
  } // anonymous namespace

  /*
   * Cube-and-conquer algorithm.
   *
   * The k-loop is the sequential one: unravel, check, check EMPTY/LOOP, prune,
   * check. What changes is how each of those SAT checks is answered.
   *
   * Cube phase (coordinator thread): it alone owns the encoder and so is the
   * only thread that creates formulas. Before every check it picks d split
   * variables (split_heuristic) and builds the 2^d cubes over them.
   *
   * Conquer phase (worker pool): each worker owns an independent SAT solver
   * that holds the same cumulative encoding (the coordinator hands every new
   * formula to all workers, which assert it into their own solver). The
   * workers then pull cube indices from a shared atomic counter, so a worker
   * that finishes an easy cube immediately takes the next one, and solve
   * `is_sat_with(cube [&& EMPTY||LOOP])`. The check is SAT as soon as one cube
   * is SAT, and UNSAT once every cube has been refuted; once it is settled no
   * further cubes are started. Optionally worker 0 races the cubes by
   * solving the whole, unsplit check (see conquer below).
   *
   * When the EMPTY/LOOP check is SAT, the solver of the worker that found the
   * model holds a model of the whole encoding, so it becomes this solver's
   * `sat` and model extraction works exactly as after solve().
   */
  tribool solver::_solver_t::solve_cube_and_conquer(
    scope const& s, logic::formula f, bool finite,
    size_t k_max, std::optional<std::chrono::seconds> timeout,
    bool semi_decision, size_t num_threads, size_t cube_depth,
    bool full_check_worker, bool pin_performance_cores
  ) {
    using clock = std::chrono::steady_clock;
    auto ms = [](clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };

    if(num_threads == 0)
      num_threads = 1;
    if(cube_depth == 0) {
      // smallest d with 2^d >= 4 * num_threads
      while((size_t(1) << cube_depth) < 4 * num_threads)
        ++cube_depth;
    }

    last_cube_counters = {};
    last_cube_counters.threads = num_threads;
    last_cube_counters.cube_depth = cube_depth;
    auto &counters = last_cube_counters;

    scope xi = chain(s);
    enc = encoder::encoder{f, xi, finite};
    trace(trace_t::nnf, xi, enc->get_formula());
    sigma = f.sigma();
    sat.reset();
    model = false;
    model_size = 0;
    last_bound = 0;

    // Each worker's scope is built here, on the coordinator, so the parent
    // scope is never touched concurrently.
    std::vector<scope> worker_scopes;
    worker_scopes.reserve(num_threads);
    for(size_t i = 0; i < num_threads; ++i)
      worker_scopes.push_back(chain(s));

    std::vector<std::unique_ptr<black::sat::solver>> sats(num_threads);
    split_heuristic heuristic;

    if(timeout)
      std::thread([timeout, self = this->shared_from_this()]() {
        std::this_thread::sleep_for(*timeout);
        self->interrupt();
      }).detach();

    worker_pool pool{num_threads};
    pool.run([&](size_t tid) {
      if(pin_performance_cores)
        request_performance_core();
      sats[tid] =
        black::sat::solver::get_solver(sat_backend, worker_scopes[tid]);
    });

    // Formulas built since the last check; every worker asserts them into its
    // own solver at the start of the next conquer phase.
    std::vector<logic::formula> to_assert;
    auto assert_formula = [&](logic::formula g) {
      heuristic.add(g);
      to_assert.push_back(g);
    };

    // One cubed SAT check. `extra`, if given, is conjoined to every cube (it
    // plays the role of the assumption of is_sat_with()). Returns true if the
    // check is SAT (the worker holding the model in `winner`), false if it is
    // UNSAT, and undef if that could not be established (interrupt/unknown).
    //
    // With full_check_worker (and more than one worker), worker 0 does not
    // take cubes: it solves the whole check, as solve() would, racing the
    // other workers that split it. This is the "concurrent cube-and-conquer"
    // remedy for checks too easy to be worth splitting: the whole-check worker
    // settles them about as fast as the sequential solver, while on hard
    // checks the cubes win.
    size_t winner = 0;
    auto conquer = [&](std::optional<logic::formula> extra) -> tribool {
      auto t0 = clock::now();
      std::vector<std::optional<logic::formula>> cubes =
        make_cubes(heuristic.pick(cube_depth));
      if(extra)
        for(auto &c : cubes)
          c = c ? logic::formula{*c && *extra} : *extra;
      counters.encode_ms += ms(clock::now() - t0);
      ++counters.checks;
      counters.cubes_generated += cubes.size();

      std::vector<logic::formula> asserting = std::move(to_assert);
      to_assert.clear();

      bool racing = full_check_worker && num_threads > 1;

      // -1 = undecided, 0 = UNSAT, 1 = SAT
      std::atomic<int> decided{-1};
      std::atomic<size_t> next{0};
      std::atomic<size_t> unsat{0};
      std::atomic<size_t> solved{0};

      // The first worker to settle the check records the answer; the others
      // take no new cubes and finish only the one they are solving. They are
      // deliberately NOT stopped with the backend's interrupt(): a Z3 solver
      // interrupted mid-check and then reused incrementally was observed to
      // give wrong answers on later checks.
      auto decide = [&](size_t tid, int result) {
        int expected = -1;
        if(decided.compare_exchange_strong(expected, result))
          winner = tid;
      };

      t0 = clock::now();

      pool.run([&](size_t tid) {
        auto &local = sats[tid];
        for(auto g : asserting)
          local->assert_formula(g);

        if(racing && tid == 0) {
          if(decided != -1 || interrupt_flag)
            return;
          tribool res = extra ? local->is_sat_with(*extra) : local->is_sat();
          if(res == true)
            decide(tid, 1);
          else if(res == false)
            decide(tid, 0);
          return;
        }

        while(decided == -1 && !interrupt_flag) {
          size_t i = next.fetch_add(1);
          if(i >= cubes.size())
            break;

          tribool res =
            cubes[i] ? local->is_sat_with(*cubes[i]) : local->is_sat();
          if(res == true) {
            ++solved;
            decide(tid, 1);
            break;
          }
          if(res == false) {
            ++solved;
            if(++unsat == cubes.size())
              decide(tid, 0);
          }
        }
      });
      counters.conquer_ms += ms(clock::now() - t0);

      counters.cubes_unsat += unsat;
      counters.cubes_skipped += cubes.size() - solved;
      if(decided == -1)
        return tribool::undef;
      if(racing && winner == 0)
        ++counters.full_check_wins;
      else if(decided == 1)
        ++counters.cubes_sat;
      return decided == 1;
    };

    auto timed = [&](auto build) {
      auto t0 = clock::now();
      auto g = build();
      counters.encode_ms += ms(clock::now() - t0);
      return g;
    };

    for(size_t k = 0; !interrupt_flag && k <= k_max; last_bound = k++) {
      trace(k);

      auto unrav = timed([&] { return enc->k_unraveling(k); });
      trace(trace_t::unrav, xi, unrav);
      assert_formula(unrav);
      if(tribool res = conquer({}); !res)
        return res;

      auto empty_or_loop = timed([&] {
        auto empty = enc->k_empty(k);
        auto loop  = enc->k_loop(k);
        trace(trace_t::empty, xi, empty);
        trace(trace_t::loop,  xi, loop);
        return logic::formula{empty || loop};
      });
      if(conquer(empty_or_loop) == true) {
        sat = std::move(sats[winner]);
        model_size = k + 1;
        model = true;
        return true;
      }

      if(!semi_decision) {
        auto prune = timed([&] { return enc->prune(k); });
        trace(trace_t::prune, xi, prune);
        assert_formula(!prune);
        if(tribool res = conquer({}); !res)
          return res;
      }
    }
    interrupt_flag = false;
    return tribool::undef;
  }

  /*
   * Instrumentation: run the sequential algorithm once, timing each k, and
   * report how much parallelism is theoretically available. This is hardware
   * independent — it characterises the shape of the computation (its critical
   * path), not the machine it happens to run on, so it bounds the speedup a
   * naive branch-parallelisation could ever reach even with unbounded cores.
   */
  solver::parallelism_report solver::_solver_t::analyze_parallelism(
    scope const& s, logic::formula f, bool finite,
    size_t k_max, bool semi_decision
  ) {
    using clock = std::chrono::steady_clock;
    auto ms = [](clock::duration d) {
      return std::chrono::duration<double, std::milli>(d).count();
    };

    solver::parallelism_report report;

    scope xi = chain(s);
    encoder::encoder local_enc{f, xi, finite};
    auto local_sat = black::sat::solver::get_solver(sat_backend, xi);

    for(size_t k = 0; k <= k_max; ++k) {
      double assert_ms = 0.0;
      double check_ms  = 0.0;

      // encode + assert the k-unraveling
      auto t0 = clock::now();
      auto unrav = local_enc.k_unraveling(k);
      local_sat->assert_formula(unrav);
      assert_ms += ms(clock::now() - t0);

      // SAT check after unraveling
      t0 = clock::now();
      tribool sat_res = local_sat->is_sat();
      check_ms += ms(clock::now() - t0);

      if(!sat_res) {
        report.result = false;
        report.decisive_k = k;
        report.assert_ms.push_back(assert_ms);
        report.check_ms.push_back(check_ms);
        break;
      }

      // EMPTY / LOOP check
      t0 = clock::now();
      auto el = local_enc.k_empty(k) || local_enc.k_loop(k);
      bool sat_with = bool(local_sat->is_sat_with(el));
      check_ms += ms(clock::now() - t0);

      if(sat_with) {
        report.result = true;
        report.decisive_k = k;
        report.assert_ms.push_back(assert_ms);
        report.check_ms.push_back(check_ms);
        break;
      }

      // encode + assert !PRUNE, then check
      if(!semi_decision) {
        t0 = clock::now();
        auto prune = !local_enc.prune(k);
        local_sat->assert_formula(prune);
        assert_ms += ms(clock::now() - t0);

        t0 = clock::now();
        tribool res = local_sat->is_sat();
        check_ms += ms(clock::now() - t0);

        if(!res) {
          report.result = false;
          report.decisive_k = k;
          report.assert_ms.push_back(assert_ms);
          report.check_ms.push_back(check_ms);
          break;
        }
      }

      report.assert_ms.push_back(assert_ms);
      report.check_ms.push_back(check_ms);

      if(k == k_max) {
        report.result = tribool::undef;
        report.decisive_k = k;
      }
    }

    // Sequential cost: every phase of every k up to the decisive one.
    double seq_total = 0.0;
    double spine_assert = 0.0;
    for(size_t k = 0; k < report.assert_ms.size(); ++k) {
      seq_total    += report.assert_ms[k] + report.check_ms[k];
      spine_assert += report.assert_ms[k];
    }

    // Ideal parallel (critical path): build the spine of unravelings 0..K*,
    // then perform the single check at K*. Every earlier check overlaps.
    double last_check =
      report.check_ms.empty() ? 0.0 : report.check_ms.back();
    double ideal_parallel = spine_assert + last_check;

    report.seq_total_ms        = seq_total;
    report.ideal_parallel_ms   = ideal_parallel;
    report.speedup_upper_bound =
      ideal_parallel > 0.0 ? seq_total / ideal_parallel : 1.0;

    return report;
  }

  void solver::_solver_t::interrupt() {
    interrupt_flag = true;
    if(sat) sat->interrupt();
  }

  template<hierarchy H, typename F>
  void _check_syntax(H h, bool quantified, F err) {
    if(h.template is<quantifier>())
      quantified = true;

    for_each_child(h, [&](auto child){
      child.match(
        [&](unary_term t, term arg){
          t.match(
            [](negative) { },
            [](to_real) { },
            [](to_integer) { },
            [&](otherwise) {
              if(!arg.is<variable>())
                err(
                  "next()/wnext()/prev()/wprev() terms can only be applied "
                  "directly to variables"
                );
            }
          );
        },
        [](tomorrow) { },
        [](w_tomorrow) { },
        [](yesterday) { },
        [](w_yesterday) { },
        [&](otherwise) {
          if(!quantified) return;
          bool temporal = child.match(
            [](always)      { return true; },
            [](eventually)  { return true; },
            [](once)        { return true; },
            [](historically){ return true; },
            [](until)       { return true; },
            [](release)     { return true; },
            [](w_until)     { return true; },
            [](s_release)   { return true; },
            [](since)       { return true; },
            [](triggered)   { return true; },
            [](otherwise)   { return false; }
          );
          if(temporal)
            err(
              "Temporal operators (excepting X/wX/Y/Z) cannot appear "
              "inside quantifiers"
            );
        }
      );
      _check_syntax(child, quantified, err);
    });
  }

  bool
  solver::check_syntax(formula f, std::function<void(std::string)> const&err)
  {
    bool ok = true;
    _check_syntax(f, false, [&](auto msg) {
      ok = false;
      err(msg);
    });
    return ok;
  }

} // end namespace black_internal::solver


