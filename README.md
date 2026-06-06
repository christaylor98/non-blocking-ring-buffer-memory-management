# non-blocking-ring-buffer-memory-management

A small, single-file Rust crate of lock-free shared-state primitives built
under a strict discipline: **only `AtomicU64::load` and `AtomicU64::store`.**
No `compare_exchange`, no `fetch_add`, no `fetch_or` — no atomic
read-modify-write instructions of any kind on the hot path. This rules out
the bus-locked operations that drive cache-coherence storms and the
scalability collapses you see in `Mutex`/`RwLock` under contention.

The trade is real: without CAS you can't build a true multi-writer
counter, and you can't build MPMC queues. What you *can* build covers
~80% of shared-state patterns in real systems code — and the benchmarks in
this repo show those patterns running 1.4×–55× faster than the obvious
`std::sync` alternative for the same workload.

## The primitives

| Primitive            | Best for                                                     | Cost per op                                  |
|----------------------|--------------------------------------------------------------|----------------------------------------------|
| `Cell<T>` inline (T ≤ 4 B) | Latest-value publish, tiny T (flags, counters, IDs)          | 1 Release store / 1 Acquire load, zero alloc |
| `SeqCell<T>`         | Large T, `Copy`, no allocation allowed (HFT, embedded, RT)   | 3 atomic ops, zero alloc, brief reader spin  |
| `BridgedCell<T>`     | Large T, readers hold zero-copy views across writes (RCU)    | 1 alloc per write, lock-free reads           |
| `Cell::append` chain | Append-only history (MVCC, Raft log, KV cache)               | 1 alloc per append, never reclaimed*         |
| `Ring<T>`            | N-snapshot retention (multi-reader staggered views)          | 1 alloc per slot per cycle                   |
| `SpscQueue<T, N>`    | Bounded single-producer single-consumer FIFO                 | 1 Acquire + 1 Release per side, zero alloc   |
| `DoubleBuffer<T>`    | Frame-boundary publish (game/graphics/audio)                 | 1 Release store per publish, zero alloc      |

\* compaction is out of scope; use `write` (mutable replace) if you want
reclamation.

For everything else — multi-writer atomic counters, MPMC queues, compound
critical sections — reach for `std::sync::atomic` or `std::sync::Mutex`
directly. This crate is opinionated about what it covers and what it
deliberately doesn't.

## Quick examples

### `Cell<u32>` — tiny atomic publish

```rust
let mut cell: Cell<u32> = Cell::new();
unsafe { cell.write(42u32); }                    // 1 Release store, no alloc
let r = cell.read(0);                            // 1 Acquire load
assert_eq!(r.value(), Some(42));
```

### `SeqCell<T>` — zero-allocation large-T publish

```rust
let cell = SeqCell::<[u64; 16]>::new();          // 128-byte payload, inline
unsafe { cell.write([0xAA; 16]); }
let r = cell.read(0);                            // reader spins only if
                                                 // it catches a write in flight
```

### `BridgedCell<T>` — RCU with automatic reclamation

```rust
let registry = ReaderRegistry::new();
let handle   = registry.acquire();                // 1 per reader thread, for life
let mut cell: BridgedCell<Vec<u64>> = BridgedCell::new();

unsafe { cell.write(vec![1, 2, 3]); }
let r = cell.read_ref(&handle, 0).unwrap();      // zero-copy pointer into heap

// ... reader iterates r while writer keeps writing ...
unsafe { cell.write(vec![4, 5, 6]); }            // old block goes to retired list
// r is still valid — its block is pinned by the handle's floor

drop(r);                                          // Drop releases the floor
cell.reclaim(&registry);                          // frees the old block
```

The Drop on `ReadRef` releases the reader's epoch floor. The Drop on
`ReaderHandle` releases the registry slot — so a panicking reader thread
stops pinning memory automatically, no heartbeat required.

### `write_lazy` — futex-style fast path

```rust
let registry = ReaderRegistry::new();
let mut cell: BridgedCell<Vec<u64>> = BridgedCell::new();

// While no reader has ever registered, write_lazy frees the old block
// IMMEDIATELY instead of retiring it. Single Acquire load to check.
unsafe { cell.write_lazy(vec![1, 2, 3], &registry); }
unsafe { cell.write_lazy(vec![4, 5, 6], &registry); }  // old block freed inline
```

### `SpscQueue<T, N>` — bounded SPSC FIFO

```rust
let q: SpscQueue<u64, 1024> = SpscQueue::new();
unsafe { q.push(42).unwrap(); }                  // single producer thread
let v = unsafe { q.pop() };                      // single consumer thread
assert_eq!(v, Some(42));
```

### `DoubleBuffer<T>` — frame-boundary swap

```rust
let db: DoubleBuffer<Vec<u64>> = DoubleBuffer::new(vec![]);

// Writer fills the back slot over the course of a "frame":
unsafe {
    let back = db.back_mut();
    back.clear();
    back.push(1); back.push(2); back.push(3);
    db.publish();                                // 1 Release store — readers
                                                 // now see the new vec
}

let v = db.read();                               // &Vec<u64>, zero-copy
```

Caller is responsible for not holding a `read()` across a `publish()`
(frame-sync discipline — the explicit tradeoff vs `BridgedCell`: cheaper
but requires the caller to manage the swap point).

## Picking the right primitive

```
size_of::<T>() ≤ 4 ────────────────────────────────────► Cell<T>      (inline)
T: Copy, writes ≥ reads, brief reader spin OK ─────────► SeqCell<T>
readers hold pointers across writes (zero-copy) ───────► BridgedCell<T>
producer / consumer FIFO ──────────────────────────────► SpscQueue<T, N>
frame-boundary publish (caller-managed sync) ──────────► DoubleBuffer<T>
append-only history retention ─────────────────────────► Cell::append chain
multi-writer counter / CAS handoff ────────────────────► std::sync::atomic
compound critical section over several fields ─────────► std::sync::Mutex
```

## Benchmark headlines

From `./bench` on a 24-core x86-64 machine, comparing each primitive
against the natural `std::sync` alternative for the same workload:

| Scenario                                           | This crate        | std::sync         | Speedup |
|----------------------------------------------------|-------------------|-------------------|---------|
| `BridgedCell<u64>` read, 8 concurrent readers      | 106 M reads/s     | 7 M (`RwLock`)    | **15×** |
| `BridgedCell<Vec<u64;128>>` read, 8 readers, ~1 KB | 128 M reads/s     | 8 M (`RwLock`)    | **16×** |
| `SpscQueue<u64, 1024>` push / pop                  | 264 M ops/s       | 10 M (`Mutex<VecDeque>`) | **26×** |
| `DoubleBuffer<Vec<u64;128>>` read+iterate          | 1.21 G iters/s    | 22 M (`RwLock`)   | **55×** |
| `write_lazy` Level B (no reader ever)              | 35.7 M writes/s   | 26.3 M (eager)    | **1.36×** |

`BridgedCell` write rate is *lower* than `Mutex<T>` write rate on small T
because each Block-path write does one heap allocation (the `Block`
struct around the value). That's the explicit cost the design pays so
readers can hold pointers across writes without blocking the writer. For
T ≤ 4 bytes the inline path of `Cell<T>` does zero allocation and runs
at ~98% of plain `u64` increment speed.

Full numbers, including reader/writer scaling tables for 1–8 readers,
live in `bench.rs` — sections 1–11.

## Safety

- **No CAS in the core path.** `grep -nE 'compare_exchange|fetch_add|fetch_sub|fetch_or' non-blocking-memory.rs`
  finds matches only in `#[cfg(test)]` telemetry counters and pre-existing
  `SeqCell` test-spin counters.
- **No POSIX signals, no TSC reads, no `Arc`/`Rc` in the core path.**
- **Adversarial concurrent stress test:** 100,000 writes, 4 reader
  threads, 1 reclaimer thread through a `Mutex<BridgedCell>` harness;
  byte-exact drop-counter reconciliation at the end proves zero
  double-free and zero leak of reclaimable memory.
- **AddressSanitizer clean:** `RUSTFLAGS="-Z sanitizer=address" rustc +nightly --test`
  builds and runs all 60 tests with no ASan diagnostics.
- **Panicking-reader test:** a thread that panics while holding a
  `ReadRef` releases its registry slot during unwind (RAII Drop runs on
  panic paths too), and reclamation proceeds normally afterwards.

## Build

This is a single-file crate — no `Cargo.toml`. Build directly with
`rustc`:

```sh
# Run the test suite
rustc --test non-blocking-memory.rs -o non-blocking-memory-test
./non-blocking-memory-test

# Run the benchmarks
rustc -O bench.rs -o bench
./bench

# AddressSanitizer pass (requires nightly)
RUSTFLAGS="-Z sanitizer=address" rustc +nightly --test \
    --target x86_64-unknown-linux-gnu \
    non-blocking-memory.rs -o non-blocking-memory-asan
./non-blocking-memory-asan
```

The benchmark binary prints 11 sections (single-thread baselines, sparse
writes, heavy writes, full stress, SeqCell size scaling, BridgedCell
single-thread + concurrent small + concurrent large, write_lazy fast
paths, SpscQueue, DoubleBuffer) — each measurement runs for 2 seconds.
Full bench takes ~2 minutes.

## Design notes

For the full design — the no-CAS discipline, the epoch-floor reclamation
protocol, the materialise-out guarantee on `read()`, the conservative-
fallback overflow policy for nested ReadRefs, and the workload survey
that mapped DBs / OS kernels / web servers / AI / real-time / embedded /
HFT / games / distributed / streaming to the right primitive — see
[`chronicle.summary.md`](chronicle.summary.md).

The doc comments at the top of each type in `non-blocking-memory.rs` are
also intended to be read directly; they include the memory-ordering
invariants and the safety arguments.

## License

See `LICENSE`.
