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
#include <vector>

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
      scope xi = chain(s);
      encoder::encoder local_enc{f, xi, finite};
      auto local_sat = black::sat::solver::get_solver(sat_backend, xi);

      // Fill in k = 0 .. tid-1 (catch-up, no SAT checks)
      for(size_t k = 0; k < tid; ++k) {
        if(shared_result.load() != -1 || interrupt_flag) return;
        local_sat->assert_formula(local_enc.k_unraveling(k));
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
            if(shared_result.load() != -1 || interrupt_flag) return;
            local_sat->assert_formula(local_enc.k_unraveling(j));
            if(!semi_decision)
              local_sat->assert_formula(!local_enc.prune(j));
          }
        }
        first = false;

        if(shared_result.load() != -1 || interrupt_flag) return;

        // Full check for our assigned k

        // 1. Assert k-unraveling and check satisfiability
        local_sat->assert_formula(local_enc.k_unraveling(k));

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

    // Launch all threads
    std::vector<std::thread> threads;
    threads.reserve(num_threads);
    for(size_t i = 0; i < num_threads; ++i)
      threads.emplace_back(worker, i);

    for(auto& t : threads)
      t.join();

    interrupt_flag = false;
    last_bound = shared_last_bound.load();

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


