# non-blocking-ring-buffer-memory-management — chronicle summary

**Session: 2026-06-06**
**Repo:** non-blocking-ring-buffer-memory-management
**Branch:** main
**Starting state:** v0.9 single-atomic Cell<T> + SeqCell<T> + Ring<T> + ReadRef<T> (zero-copy read handle), all 37 baseline tests passing, Blocks leaked at TODO.

---

## Summary — bridge-side reclamation, full primitive set, empirical benchmarks

This session closed the long-standing reclamation TODO and then built out the full
shared-state primitive set that the axOS / axAporia memory model was hinting at.
Three threads, in order:

1. **Bridge-side automatic reclamation** (INTENT_SPEC.exec.v2)
2. **Toolbox survey** — which primitive optimally fits which programming workload class
3. **The "start naive, wrap on contention" question** → concrete implementation as
   `write_lazy` (Level A + B), plus the missing primitives identified by the
   workload survey (SpscQueue, DoubleBuffer)

By session end: **60 tests passing** (37 → 60), zero use-after-free / double-free
under AddressSanitizer + concurrent stress, clean greps for the no-CAS discipline,
and 6 benchmark sections comparing against `std::sync::{Mutex, RwLock}` head-to-head.

---

## Thread 1 — Bridge-side automatic reclamation

Folded the v2 spec in fully. The contact-points design held: **two RAII Drops**
(`ReadRef::drop`, `ReaderHandle::drop`) are the only reclamation surface the Axis
programmer indirectly touches — and even then only via Rust's automatic drop
insertion at scope end. The axis-layer test code reads and uses values; it never
mentions epochs, floors, slots, or reclaim.

Key design moves the spec forced into shape:

- **ReaderRegistry** — fixed `[AtomicU64; 64]` slots, `u64::MAX = idle`. Single
  writer per slot (the owning ReaderHandle), sweeper Acquires all, writes none.
  Slot allocation via `Mutex<usize>` confined to `acquire()` — the cold path; the
  hot read/write/reclaim paths see only `AtomicU64::load`/`store`. No CAS anywhere
  in the core.
- **ReaderHandle::drop clears the slot** to `u64::MAX`. This is the dead-reader
  fix: a panicking thread releases its pin during unwind, automatically, no
  heartbeat, no timeout. Verified by an adversarial test that spawns a thread,
  acquires a pin, then panics — slot returns to MAX, reclamation proceeds.
- **HoldStack with conservative-fallback overflow.** Per-handle stack of live
  ReadRef epochs (depth 8). On overflow: `overflow_count` tracks extra holds and
  a separate `overflow_floor` keeps the floor conservatively low. The floor is
  released to MAX only when both `count == 0 AND overflow_count == 0`. Pins
  longer than strictly necessary, never frees too early. Telemetry counter
  records every overflow so we can see whether depth-8 is ever genuinely
  exceeded (stress test confirmed: 0 hits in the realistic concurrent workload;
  the explicit-overflow test fires exactly the predicted 4).
- **Materialise-out as a load-bearing API distinction.** `read()` returns a
  cloned-out owned value and never receives a `ReaderHandle` — the API itself
  makes pinning impossible. `read_ref()` requires a handle and pins the floor
  for Block-path values. Inline values still pin nothing (the value is decoded
  to a register-sized copy; there is no Block to pin). The adversarial test
  proves the floor stays at `u64::MAX` across `read()` even with a handle in
  scope, and the source Block is reclaimed by the next sweep while the owned
  clone survives untouched.
- **Floor pin protocol (no CAS).** On `read_ref`:
  1. `slot.store(1, Release)` — conservative pre-publish
  2. `head.load(Acquire)` — read the head
  3. push epoch onto handle's HoldStack
  4. `slot.store(holds.floor(), Release)` — tighten to live min
  A sweeper running between (1) and (4) sees floor = 1; since epochs start at 1,
  nothing can be freed (retired_epoch < 1 has no solutions). After (4) the floor
  is exactly the read epoch.

**Boxed HoldStack** — a real bug caught the first time the adversarial test
`floor_min_reflects_slowest_reader` ran. Pushing handles into a `Vec` moved them
in memory; `ReadRef`'s raw `*mut HoldStack` pointer into the original location
became dangling. Fix: `holds: Box<UnsafeCell<HoldStack>>` so the heap allocation
stays put even when the `ReaderHandle` struct moves. Documented as the invariant
that lets ReadRef carry raw pointers safely.

**Stress reconciliation** as the primary safety evidence: 100,001 writes,
~420,000 transient reads, 4 reader threads + 1 writer + 1 reclaimer through a
`Mutex<BridgedCell>` harness. Drop-counter equality: `drops_after == writes_total`
proves zero double-free and zero leak of reclaimable memory. Repeated under
`RUSTFLAGS="-Z sanitizer=address" rustc +nightly --test` — also clean.

---

## Thread 2 — Toolbox survey

Asked across major programming workload classes (DB, OS, web server, AI
inference + training, real-time control, embedded, HFT, game engine, distributed
consensus, stream processing, browser engines) what the optimal memory model is.

The honest reduction: **across all those workloads, only ~5 shared-state
patterns recur**:

| Pattern                                                       | Best primitive               |
|---------------------------------------------------------------|------------------------------|
| Latest-value publish, tiny T (≤ 4 B)                          | `Cell<T>` inline             |
| Latest-value publish, large T, Copy, brief read-spin OK       | `SeqCell<T>`                 |
| Latest-value publish, large T, readers escape the read scope  | `BridgedCell<T>`             |
| Append-only history                                           | `Cell::append` chain         |
| Bounded SPSC FIFO                                             | (gap)                        |
| Multi-writer atomic counter / CAS handoff                     | `std::sync::atomic` (gap)    |
| Frame-boundary publish                                        | (gap)                        |
| Compound critical section                                     | `Mutex<T>`                   |

Three honest observations fell out:

1. **`BridgedCell` is the universal answer for RCU-shaped workloads.** Databases
   (buffer pool, MVCC, B-tree internals), OS kernels (dentry cache, page tables
   inner — Linux RCU literally), web routing tables, browser DOMs, AI KV caches,
   search-engine inverted indexes mid-merge. The largest single class of
   shared-state in systems software.
2. **`SeqCell` owns "publish latest, no alloc allowed."** HFT, embedded, real-time
   control, automotive, aerospace. Any workload that excludes `malloc` on the
   hot path.
3. **Three honest gaps in the toolbox:** (a) bounded SPSC FIFO, (b) double-buffer
   frame-boundary publish, (c) multi-writer atomic counter. (c) deliberately
   stays outside the no-CAS discipline; we document the surrender. (a) and (b)
   are small, well-scoped additions.

The "smart memory manager" question got the answer:

- **Level 1 — decision table at compile time** is the highest-value artefact.
  `size_of::<T>()` + `T: Copy` + "does reader escape the scope" disambiguates
  ~80% of cases.
- **Level 2 — uniform facade trait** is worth it if many call sites are written
  by non-experts; one vtable indirection per op (~1 ns), invisible against any
  non-inline backing.
- **Level 3 — adaptive/migrating storage** is where projects go to fail (Java
  biased locking deprecated in JDK 15, removed in JDK 18). Don't build.

---

## Thread 3 — "Why assume contention?" → fast paths and the missing primitives

The load-bearing question: **99% of "shared" data is not actually under
contention even in multiprocessor code.** Why pay the synchronization cost
unconditionally? This is the futex pattern (Linux fast user-space mutex):
cheap fast path that handles the no-contention case, slow path only on actual
contention.

For our toolbox, three concrete fast-path opportunities:

- **Level A** — BridgedCell::write does a `floor_min()` scan after publishing
  the new head; if `floor_min > old_epoch`, free the old block immediately
  instead of pushing to the retired list. Safe by construction (the publish
  Release-store and the slot Release-stores synchronise correctly with the
  Acquire scan). Pays 64 cache-warm Acquire loads per write.
- **Level B** — ReaderRegistry holds a monotonic `any_handle_ever: AtomicBool`.
  Flipped Release-true on first `acquire()`, never cleared. BridgedCell::write
  checks it first (one Acquire load); if false, no reader has ever existed →
  free immediately. Bypasses the retired list and reclaim cycle entirely.
- **Level C** — promote-on-contention `Mem<T>` is the trap. Skipped.

Built as `BridgedCell::write_lazy(value, registry)` — opt-in alongside the
existing `write()`. Same wire semantics, with the fast-path branches.

Empirical results (section 9 of the bench):

| Variant                                  | rate     | vs `write` |
|------------------------------------------|----------|------------|
| `write` baseline (always retire)         | 26.3 M/s | 1.00 ×     |
| `write_lazy` Level B (no reader ever)    | 35.7 M/s | **1.36 ×** |
| `write_lazy` Level A (reader idle, scan) | 19.3 M/s | 0.73 ×     |
| `write_lazy` (pinning reader)            |  1.3 M/s | 0.05 ×     |

**The hypothesis confirmed for the no-reader case**: Level B (single Acquire
load) buys a 36 % uplift by skipping the alloc/retire/reclaim cycle entirely.
Level A's 64-slot scan costs more than it saves unless the reader is genuinely
idle for long stretches — useful as a knob, not a default. Pinning case shows
the scan as pure tax. **Recommendation: Level B becomes the default; Level A
opt-in for profiled "reader exists but mostly idle" workloads.**

### The missing primitives, built

- **`SpscQueue<T, N>`** — Lamport ring buffer, head/tail `AtomicUsize`, no CAS.
  Each side does one Acquire load (other side's index) + one Release store. No
  lock, no kernel, no allocation per op. Drop walks the unread range so T::drop
  fires once per item.

  vs `Mutex<VecDeque<u64>>`: **~26 × faster** (264 M/s vs 10 M/s). Largest single
  gap in the whole bench. Every render command stream, audio feed, log shipper,
  IPC channel currently behind `Mutex<VecDeque>` is leaving an order of magnitude
  on the floor.

- **`DoubleBuffer<T>`** — two `UnsafeCell<T>` slots, one `AtomicUsize` front
  index. Writer fills back, single Release-store flips front. Reader does one
  Acquire load and then has a stable `&T` until the next publish. No lock, no
  refcount, no allocation. Caller is responsible for not holding `read()`
  across a `publish()` (frame-sync discipline — the explicit tradeoff vs
  BridgedCell: cheaper, but requires the caller to manage the swap point).

  vs `RwLock<Vec<u64>>`: **~55 × faster on reads** (1.21 G/s vs 22 M/s), 2 × on
  writes. The 55× gap is the price of the caller-managed frame contract.

---

## Through-line

The axOS memory model started from a no-CAS discipline (only `AtomicU64::load` and
`store`) and a small set of primitives (Cell, SeqCell, Ring). The bridge layer
added the missing piece — automatic Block reclamation — without breaking the
discipline: epoch floors, registry slots, ReadRef::drop and ReaderHandle::drop
all use only load/store. The workload survey showed that **most of what real
systems software needs reduces to ~5 patterns**, and the existing primitives
already covered three; this session closed the remaining two (`SpscQueue`,
`DoubleBuffer`) and added the futex-style fast path to BridgedCell that the
"99 % of data isn't actually contended" insight demands.

The toolbox now in hand:

| Primitive            | When                                                            |
|----------------------|-----------------------------------------------------------------|
| `Cell<T>` inline     | T ≤ 4 B, atomic publish-latest                                  |
| `SeqCell<T>`         | T > 4 B, Copy, no alloc allowed, brief reader spin OK           |
| `BridgedCell<T>`     | T > 4 B, readers hold zero-copy views across writes (RCU)       |
| `Cell::append` chain | Append-only history / MVCC / Raft log / KV cache                |
| `Ring<T>`            | N-snapshot retention (axAporia ring 1/2/3 pattern)              |
| `SpscQueue<T, N>`    | Bounded single-producer single-consumer FIFO                    |
| `DoubleBuffer<T>`    | Frame-boundary publish (game/graphics/audio swap pattern)       |
| `std::sync::atomic`  | Multi-writer counter / CAS handoff (the deliberate gap)         |
| `Mutex<T>`           | Compound critical section over several fields (last resort)     |

Decision rule for picking among them is captured as the table above and the
narrative in this chronicle; no runtime facade was built (Level 1 from the
smart-manager discussion). The plan is to deploy these across real workloads
and let observation drive any future Level 2 / facade decision.

Outstanding deliberate gap: multi-writer atomic counters (metrics, transaction
IDs, Raft terms). These genuinely require CAS and stay outside this crate's
discipline; users reach for `std::sync::atomic::AtomicU64` directly. The
chronicle records this as a non-goal, not a debt.

---

## Artefacts

- `non-blocking-memory.rs` — single-file crate, ~3200 lines, all primitives + tests
- `bench.rs` — 11 benchmark sections (sections 6–11 added this session)
- Test count: 60 passing (37 baseline + 9 bridge adversarial + 4 write_lazy +
  5 SpscQueue + 4 DoubleBuffer + 1 holdstack-overflow)
- ASAN: clean under `rustc +nightly -Z sanitizer=address --test`
- Greps: no `compare_exchange|fetch_add|fetch_sub|fetch_or` in the core path
  (all hits are in `#[cfg(test)]` counters or pre-existing SeqCell test
  telemetry); no `signal|SIGUSR|rdtsc|Arc<|Rc<` in the core path either.
