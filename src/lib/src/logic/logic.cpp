//
// BLACK - Bounded Ltl sAtisfiability ChecKer
//
// (C) 2022 Nicola Gigante
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

#include <black/logic/logic.hpp>

#include <boost/unordered/concurrent_flat_map.hpp>

#include <atomic>
#include <mutex>
#include <variant>
#include <vector>

//
// This file contains the implementation of some components declared in
// `logic.hpp` and subfiles. In particular, here we declare some components of
// the `alphabet` class. BLACK's logic API is for the most part a header library
// being 99% templates, but this part is implemented in a source file mainly in
// order to keep `boost::concurrent_flat_map` as a private dependency. To
// understand what follows, be sure to read the explanations in `core.hpp` and
// `generation.hpp`.
//

namespace black_internal::logic {

  } namespace std {
    template<typename T>
    struct hash<std::vector<T>>
    {
      size_t operator()(std::vector<T> const&v) const {
        hash<T> h;
        size_t result = 0;
        for(size_t i = 0; i < v.size(); ++i)
          result = ::black_internal::hash_combine(result, h(v[i]));

        return result;
      }
    };
  } namespace black_internal::logic {

  //
  // The `alphabet` class keeps an hash table from nodes to pointer to nodes.
  // When we insert a node, if it already exists, we get the existing copy of it
  // from the hash table. If it does not, we insert it in the hash table. This
  // mechanism is implemented in the following class, which will be indirectly
  // inherited by the pimpl class `alphabet_impl`.
  //
  // Stage 2 of parallelizing BLACK: `_map` is now a `boost::concurrent_flat_map`,
  // which handles its own internal locking (many small locks instead of one big
  // one) and is safe for several threads to read and write at once. That removes
  // the need for `alphabet_impl`'s old coarse `std::mutex` around every lookup:
  // the common case (the node already exists) now needs no lock from us at all.
  //
  // `_store` is still a plain `std::deque`, which is NOT safe for concurrent
  // `emplace_back()`. So the only remaining critical section is the rarer path
  // where a node is genuinely new: `_store_mutex` protects just that insertion,
  // not the lookup. Two threads racing to insert the same new node is handled
  // by `_map`'s own `try_emplace_or_visit`: only one insert wins atomically, and
  // the losing thread is hand back the winner's pointer instead, so uniqueness
  // still holds. The loser's `_store` slot is simply never referenced again
  // (a small, harmless amount of wasted memory, not a correctness problem,
  // since nothing else in the alphabet keeps `_store` slots reachable by index).
  //
  template<storage_type Storage>
  struct storage_allocator {
    std::deque<storage_node<Storage>> _store;
    std::mutex _store_mutex;
    boost::concurrent_flat_map<
      storage_node<Storage>, storage_node<Storage> *,
      std::hash<storage_node<Storage>>
    > _map;

    // Counts how many times a thread found `_store_mutex` already held and had
    // to wait for it. Only the "genuinely new node" path takes this lock at
    // all, so this is a much narrower measure of contention than the old
    // alphabet-wide mutex counter: uncontended lookups never touch it.
    std::atomic<size_t> _store_contended{0};

    storage_node<Storage> *allocate(storage_node<Storage> const& node) {
      // Fast path: lock-free (from our side) lookup. Most calls land here.
      storage_node<Storage> *found = nullptr;
      _map.visit(node, [&](auto const& kv) { found = kv.second; });
      if(found)
        return found;

      // Slow path: build the node under our own lock (protects `_store`,
      // which the concurrent map doesn't own or know about).
      storage_node<Storage> *obj;
      {
        if(!_store_mutex.try_lock()) {
          _store_contended.fetch_add(1, std::memory_order_relaxed);
          _store_mutex.lock();
        }
        std::lock_guard<std::mutex> lock(_store_mutex, std::adopt_lock);
        obj = &_store.emplace_back(node);
      }

      // Atomically insert if still absent, or fetch whichever pointer won the
      // race if another thread inserted the same node in the meantime.
      storage_node<Storage> *winner = obj;
      _map.try_emplace_or_visit(
        node, obj,
        [&](auto const& kv) { winner = kv.second; }
      );

      return winner;
    }
  };

  //
  // We specialize the case of a single boolean field (i.e. the `boolean`
  // storage kind). Other optimized specializations could be possible in the
  // future.
  //
  template<storage_type Storage>
    requires (std::is_same_v<
      typename storage_data_t<Storage>::tuple_type, std::tuple<bool>
    >)
  struct storage_allocator<Storage>
  {
    storage_node<Storage> _true{element_of_storage_v<Storage>, true};
    storage_node<Storage> _false{element_of_storage_v<Storage>, false};

    // Never touched: returning one of two fixed constants needs no locking
    // and no map lookup at all. Present only so `lock_contention_count()`'s
    // generated loop over every storage kind compiles uniformly.
    std::atomic<size_t> _store_contended{0};

    storage_node<Storage> *allocate(storage_node<Storage> node) {
      if(std::get<0>(node.data.values))
        return &_true;
      return &_false;
    }
  };

  struct alphabet_base::alphabet_impl : std::monostate
  #define declare_storage_kind(Base, Storage) \
    , storage_allocator<storage_type::Storage>
  #include <black/internal/logic/hierarchy.hpp>
  {
    #define declare_storage_kind(Base, Storage) \
      using storage_allocator<storage_type::Storage>::allocate;
    #include <black/internal/logic/hierarchy.hpp>

    //
    // Stage 2 of parallelizing BLACK: the old alphabet-wide `std::mutex` that
    // serialised every lookup across every storage kind is gone. Concurrency
    // is now handled per storage kind, inside each `storage_allocator`
    // (`boost::concurrent_flat_map` for lookups, a small `_store_mutex` only
    // for genuinely new nodes). Sequential (single-threaded) use pays for
    // exactly the same work as before; concurrent use no longer serialises on
    // one lock for reads that don't need it.
    //

    // Total lock contention across every storage kind's `_store_mutex`, i.e.
    // how many times a thread had to wait to insert a genuinely new node.
    // This is a coarser signal than the old per-lookup counter (it only
    // fires on the slow/new-node path now, since the fast/lookup path no
    // longer takes a lock at all), but it is comparable across runs for the
    // software-threads study.
    size_t lock_contention_count() const {
      size_t total = 0;
      #define declare_storage_kind(Base, Storage) \
        total += storage_allocator<storage_type::Storage>::_store_contended \
          .load(std::memory_order_relaxed);
      #include <black/internal/logic/hierarchy.hpp>
      return total;
    }
  };

  //
  // Out-of-line definitions of constructors and assignments of `alphabet_base`,
  // declared in `generation.hpp`
  //
  // The pimpl is created eagerly (rather than lazily in `impl()`) so that the
  // `if(!_impl)` check below cannot race: with several threads interning nodes
  // into a shared alphabet, a lazy first-touch would otherwise let two threads
  // both construct the impl. Construction is cheap (empty stores/maps), so
  // doing it up front costs nothing meaningful for sequential use.
  alphabet_base::alphabet_base() : _impl{std::make_unique<alphabet_impl>()} { }
  alphabet_base::alphabet_base(alphabet_base &&) = default;
  alphabet_base &alphabet_base::operator=(alphabet_base &&) = default;
  alphabet_base::~alphabet_base() = default;

  alphabet_base::alphabet_impl *alphabet_base::impl() {
    if(!_impl)
      _impl = std::make_unique<alphabet_impl>();

    return _impl.get();
  }

  //
  // out-of-line definitions of `alphabet_base` member functions, which will be
  // inherited by `alphabet` and used by the constructors of storage and element
  // classes.
  //
  #define declare_storage_kind(Base, Storage) \
    storage_node<storage_type::Storage> * \
    alphabet_base::unique( \
      storage_node<storage_type::Storage> node \
    ) { \
      /* No outer lock here anymore: allocate() below does its own, narrower */ \
      /* locking (concurrent map for lookups, a small per-kind mutex only */ \
      /* for genuinely new nodes), instead of one lock shared by everything. */ \
      return impl()->allocate(std::move(node)); \
    }

  #include <black/internal/logic/hierarchy.hpp>

  // Total number of times a thread had to wait to insert a genuinely new node,
  // summed across every storage kind, over this alphabet's lifetime.
  // solve_parallel() snapshots this before/after a run to report the
  // per-solve lock contention. Returns 0 if the impl is not built.
  size_t alphabet_base::lock_contention_count() const {
    return _impl ? _impl->lock_contention_count() : 0;
  }

}

