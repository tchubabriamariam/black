# Can Z3's incremental mode be shared across threads?

Investigation into whether the k-level parallel solver can avoid making each
worker thread rebuild its Z3 solver state from scratch — i.e. whether Z3's
*incremental* solving (learned-clause reuse) can be **shared across threads**.

Context: the [formula-family study](../parallel_study.cpp) showed that k-level
branch-parallelism only helps "check-spread" formulas and does nothing for
"decision-bound" ones (where a single deep SAT check dominates). This asks
whether sharing solver state could close the gap.

Tested with Z3 4.15.4 on an Apple M3 Pro (6 performance cores).

## How Z3 constrains the design

From Z3's own API documentation (`z3_api.h`):

> Thread safety: operations on a context are not thread safe. To use Z3 from
> different threads create separate context objects. The `Z3_translate`,
> `Z3_solver_translate` … methods are exposed to allow copying state from one
> context to another.

So a solver cannot be *touched* from two threads. The only sanctioned way to
move built-up solver state to another thread is `Z3_solver_translate`
(a deep copy into that thread's own context). `push`/`pop` exist but only work
within a single thread.

## Experiment 1 — is incremental reuse worth sharing, and can translate share it?

`incremental_vs_translate.cpp`. A growing hard SAT instance (random 3-SAT near
the phase transition) asserted in batches ("depths"), under three strategies:

| strategy | what it does | mirrors |
|----------|--------------|---------|
| `INC`    | one solver, assert batch + check each depth (reuses learning) | the sequential solver |
| `COLD`   | fresh solver rebuilt from scratch at each depth | what a parallel worker does |
| `XLATE`  | build the spine once, `Z3_solver_translate` a copy per depth to check | the proposed sharing |

Representative results:

```
V=90  depths=22 batch=18  final=UNSAT
  INC   (incremental, 1 solver)       6.7 ms
  COLD  (rebuild each depth)         26.6 ms   (3.97x INC)
  XLATE (translate copy each depth) 2924.4 ms   (436x INC)  translate-only=1887 ms
V=130 depths=30 batch=19  final=UNSAT
  INC   (incremental, 1 solver)      17.2 ms
  COLD  (rebuild each depth)         45.7 ms   (2.67x INC)
  XLATE (translate copy each depth) 3948.4 ms   (230x INC)  translate-only=2534 ms
```

Two conclusions:

1. **Incremental reuse is a real prize.** `INC` beats `COLD` by **2.7–4.5×** on
   hard instances — the sequential solver's learned-clause reuse genuinely
   matters, and parallel workers throw it away.
2. **`Z3_solver_translate` cannot deliver that prize.** It is **230–570×**
   slower, ~75–90 ms *per call*, and scales with solver-state size. It costs far
   more than the solving it is meant to accelerate. (Re-*asserting* the spine is
   cheap; the expensive thing to move is the *learning*, and translate is too
   heavy to move it.)

**Verdict: sharing incremental state across threads is blocked by Z3's
architecture.** A legitimate negative result.

## Experiment 2 — Z3's *own* internal parallelism

`parallel_check.cpp`. Instead of sharing state across our threads, let Z3
parallelise a single hard `check()` internally via `sat.threads`:

```
V=170 CL=720 (single hard check)
  sat.threads=1 (seq)   152.2 ms   UNSAT
  sat.threads=4          41.1 ms   UNSAT   (~3.7x)
  sat.threads=8          46.1 ms   UNSAT   (past the sweet spot)
```

**~3–3.7× on a single hard check**, and it is a one-line change to enable.

## Takeaway

The two forms of parallelism are **complementary**, and together they cover the
gap the formula-family study identified:

| formula type | bottleneck | what helps |
|--------------|-----------|------------|
| check-spread (high `par%`) | many intermediate checks | k-level branch-parallelism (`solve_parallel_shared`) |
| decision-bound (high `dec%`) | one deep hard check | Z3 internal parallelism (`sat.threads`) |

Naive cross-thread *state sharing* is a dead end; k-parallelism plus Z3's
internal `sat.threads` is the practical combination worth pursuing.

## Building / reproducing

These are standalone programs that link Z3 directly (not through BLACK), so they
are not part of the CMake build. Build them with the Z3 from Homebrew:

```sh
c++ -std=c++20 -O2 incremental_vs_translate.cpp \
    -I/opt/homebrew/include -L/opt/homebrew/lib -lz3 \
    -Wl,-rpath,/opt/homebrew/lib -o incremental_vs_translate
./incremental_vs_translate            # or: ./incremental_vs_translate <V> <depths> <batch>

c++ -std=c++20 -O2 parallel_check.cpp \
    -I/opt/homebrew/include -L/opt/homebrew/lib -lz3 \
    -Wl,-rpath,/opt/homebrew/lib -o parallel_check
./parallel_check 170 720              # <vars> <clauses>
```
