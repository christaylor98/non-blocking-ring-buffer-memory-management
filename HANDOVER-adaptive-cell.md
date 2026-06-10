# Handover: AdaptiveCell prototype → bridge-layer adoption

Audience: the session owning `non-blocking-memory.rs` (the bridge layer).
Everything below was built and measured against the crate as of this
session; the prototype lives in `adaptive-cell-prototype.rs` in this
repo and includes the main file via `#[path]`, so it compiles against
the real primitives, not copies.

```sh
# tests (5 adaptive_* + your original 60, all green)
rustc --edition 2021 --test adaptive-cell-prototype.rs -o adaptive-test && ./adaptive-test
# single-thread bench           multithread bench
rustc --edition 2021 -O adaptive-cell-prototype.rs -o adaptive-bench
./adaptive-bench                ./adaptive-bench mt
```

---

## 1. What the prototype is

`AdaptiveCell<T: Copy>` — one cell, two engines, picked per write by the
single writer:

- COLD = SeqCell-style seqlock on an inline slot. Zero allocation.
- HOT  = BridgedCell block path. One alloc/write; readers hold
  floor-pinned zero-copy `ReadRef`s across writes.

Readers signal "I want pinned views" by storing 1 into a per-handle
DEMAND slot (`AdaptiveRegistry` = `ReaderRegistry` + a parallel
`[AtomicU64; MAX_READERS]`, same single-writer-per-slot rule). The
writer checks demand before each write (gated on `has_any_reader()`, so
never-shared cells pay one Acquire load) and escalates/de-escalates
with a `COOL_DOWN = 64`-write hysteresis. No CAS anywhere; demand is
advisory — a stale scan costs one write in the wrong mode, never
safety.

### The ctrl word (the one new atomic)

Mode and seqlock counter must be read atomically together, so:

```text
bit 0      MODE         0 = cold, 1 = hot
bit 1      IN-PROGRESS  seqlock odd bit
bits 63:2  EPOCH        completed-write count
```

- Cold write: `ctrl→odd, write_volatile slot, ctrl→even`
- Hot write:  `ctrl→odd|MODE, write_volatile slot, Block publish
  (Release, via BridgedCell::write), ctrl→even|MODE`
- Escalation = first hot write (odd store sets MODE);
  de-escalation = first cold write after cool-down (odd store clears
  MODE; straddling readers fail validation and retry).

### The load-bearing safety rule (learned the hard way)

**The inline slot is kept current in BOTH modes** (hot writes pay one
extra 128 B volatile copy). Therefore owned `read()` is a pure seqlock
in every mode and never dereferences a Block; only floor-pinned
`read_pinned()` views touch Blocks, through the unmodified floor
protocol. V1 of the prototype instead cloned out of the head Block for
hot owned reads — owned reads pin nothing, so a concurrent watermark
sweep could `Box::from_raw` the Block mid-clone. The stress test caught
it within one run as a torn read containing recycled-heap garbage.
Rule for adoption: **no unpinned Block dereference may coexist with any
free site**, ever.

De-escalation never frees the hot head Block — it just stops publishing
through it. The Block stays as BridgedCell's head until a later hot
write retires it, and retirement is only freed by `reclaim`, which
honours floors. So the wrapper adds zero new free sites.

### Transition validation (readers, load/store only)

- Owned read: exact ctrl match around the volatile copy (any write or
  transition forces retry).
- Pinned read: serve from head only when ctrl is even-with-MODE (the
  escalating write publishes its Block *before* storing even|MODE);
  after `read_ref` returns, re-check MODE is still set — if a
  de-escalation raced, drop the ref (releases its floor) and retry on
  the cold/copy path. ABA (de-esc + re-esc between c0 and validation)
  is benign: the value was current at c0; pinned memory stays protected
  by floors regardless.

---

## 2. Findings, ordered by adoption value

### F1 — Shrink `ReadRef` (do this first; it's independent of everything else)
`ReadRefInner::Inline(T)` reserves `size_of::<T>()` inside every
`ReadRef<T>`, though Inline is only used for T ≤ 4 B.
`ReadRef<[u64;16]>` is **184 bytes**. `BridgedCell::read_ref` hides it
via RVO (4.2 ns), but moving a ReadRef through any wrapping enum/struct
defeats RVO and double-copies ~200 B: measured 4.2 → 18.7 ns for the
wrap alone (single-thread bench, section 4 attribution probes). Store
the inline payload as 4 encoded bytes (`u32`/`[u8;4]`, decode on
deref) → ReadRef ≈ 48 B → composition becomes free, and the
prototype's pinned-read gap (and part of its MT reader gap) closes.

### F2 — `BridgedCell::read` is not reclaim-concurrent (doc/API hazard)
Owned `read()` clones from the head Block while pinning nothing. It is
sound only because `&mut self` (write/reclaim) can't overlap `&self`
(read) in compiler-checked code. Any harness that shares a BridgedCell
across threads with a free-running writer (Mutex released between ops,
UnsafeCell, future API changes) makes owned reads UAF-racy. Either
document this loudly on `read()`, or adopt the prototype's
slot-mirroring trick so owned reads never touch Blocks. This is also
why the MT bench has no "BridgedCell owned reads under write pressure"
row.

### F3 — Block-path writer throughput under pinning readers is
coherence-bound, not policy-bound
With 4–8 readers churning `read_ref`, the BridgedCell writer collapses
to ~1.2 M writes/s. Falsified hypotheses, in order: retired-list
growth (disproved — `retired_len` stays ~150 under the watermark
policy); sweep policy (disproved — watermark vs write-count-amortised
sweeps measured identical 1.2 M/s). Surviving explanation: every
reader pass shares the head + Block cache lines, so each writer store
pays a cross-core RFO. Supporting probe: throttling a single reader
recovered the writer 5.3 → 27.5 M/s in one run (high run-to-run
variance on the shared VM — treat as well-supported, not proven).
The AdaptiveCell hot writer sustains 3.2–8.4 M/s in the same scenario,
plausibly because its ctrl seqlock bracket makes readers yield while a
write is in flight — an implicit writer-priority backoff that
BridgedCell readers have no signal for. If you want this without full
AdaptiveCell adoption: give BridgedCell readers a cheap
write-in-progress signal to yield on.

### F4 — Floor-slot false sharing
`ReaderRegistry.slots` packs 8 `AtomicU64` per cache line; each pinned
read does 3 stores to its slot (conservative 1, tightened floor, drop
release). Neighbouring reader threads false-share those lines.
Pad each slot to 64 B (`#[repr(align(64))]` wrapper). Cheap, likely
improves `read_ref` scaling beyond what we measured. Same applies to
the prototype's demand array if adopted (less critical — demand slots
are written once per phase, not per read).

### F5 — `reclaim_if_watermark` has an every-write regime
Once `retired_len ≥ WATERMARK` and floors are starved (some reader is
almost always inside its conservative `store(1)` window, so
`floor_min == 1`), the sweep runs on EVERY write: 64 contended floor
loads + O(retired_len) scan, freeing nothing. Per F3 this is not the
dominant writer cost, but it is wasted work. The prototype sweeps every
`SWEEP_PERIOD = 8192` hot writes instead. Trade observed at 8 readers:
watermark holds `retired_len` ≈ 150 (CPU for memory); amortised lets
the backlog ride to ~10–60 k entries ≈ MBs between successful drains
(memory for CPU). A reasonable crate policy: watermark-triggered but
with exponential back-off after a failed (zero-freed) sweep.

### F6 — Minor: sticky-demand conditional store
`read_pinned` re-storing 1 into its demand slot every call keeps the
line in M state against the writer's scan; load-then-conditional-store
is effectively free. Already in the prototype.

---

## 3. Numbers (this hardware: 32-core shared VM, rustc 1.96 -O, T = [u64;16], 128 B)

### Single-thread (ns/op)
| Scenario | Adaptive | SeqCell | BridgedCell |
|---|---|---|---|
| Write, no demand | 8.6 | 5.9 | 12.1 (write_lazy) |
| Write, demand present | 27.6 | — | 24.1 |
| Owned read | 5.1 | 3.1 | 11.0 |
| Pinned read acquire+drop | 19.5 (F1 explains the gap) | n/a | 4.2 |
| Mixed 20×200k phases | **19.2** | 5.9 (can't pin) | 30.0 |

### Multithread, 1 writer flat-out (writer M/s → readers M/s)
| | 1 reader | 4 readers | 8 readers |
|---|---|---|---|
| A owned — Adaptive cold | 46.2 → 16.2 | 26.8 → 21.8 | 18.0 → 28.3 |
| A owned — SeqCell | 65.3 → 18.6 | 45.0 → 32.3 | 20.3 → 36.1 |
| A owned — RwLock | 16.0 → 24.6 | 13.9 → 10.2 | 6.9 → 5.9 |
| B pinned — Adaptive hot | 8.4 → 11.5 | 5.1 → 8.1 | 3.2 → 17.2 |
| B pinned — BridgedCell | 1.6 → 16.9 | 1.2 → 18.5 | 1.2 → 34.2 |
| B pinned — RwLock | 7.2 → 14.9 | 11.4 → 8.8 | 6.6 → 8.1 |
| C mixed — Adaptive | — | 12.8 → 7.7 | 8.5 → 11.8 |
| C mixed — BridgedCell | — | 1.2 → 17.8 | 1.2 → 32.4 |

Read of the data: seqlock-family readers SCALE with reader count under
a live writer, RwLock's collapse; scenario B is a frontier (BridgedCell
maxes readers, Adaptive trades reader peak for 3–5× writer); scenario C
(demand comes and goes) is the hybrid's headline — 7–10× writer at a
2–3× reader cost, and the reader cost is substantially F1.

---

## 4. Suggested integration path

1. **F1 ReadRef shrink** in `non-blocking-memory.rs`. Re-run the
   prototype benches; expect the pinned-read gap to mostly close.
2. **F4 slot padding** in `ReaderRegistry`. Re-run MT scenario B.
3. Move `AdaptiveCell` in as a sibling of `BridgedCell`, constructing
   its pinned return **in place** (inside the crate you don't need the
   `Pinned` wrapper at all — return the shrunken `ReadRef` plus a
   cell-level epoch, or extend ReadRef with the cell epoch).
4. Decide epoch semantics: prototype reports the ctrl epoch
   (all writes, both modes); the inner Cell/Block epochs count only hot
   writes and stay internal. If you want one counter, plumb
   `write_with_epoch` through `BridgedCell::write` so Block epochs can
   be driven by the ctrl count. The floor protocol only needs Block
   epochs to be monotone — feeding it ctrl epochs is safe (gaps are
   fine, reordering is not).
5. **F2**: at minimum a doc warning on `BridgedCell::read`; ideally the
   slot-mirroring rule.
6. README: AdaptiveCell row in the primitives table ("demand-driven
   SeqCell/BridgedCell hybrid; pinned views on demand, seqlock speed
   otherwise") and a picking-guide line: `pin demand intermittent or
   unknown → AdaptiveCell`.

## 5. Contracts, limits, open questions

- Same writer contract as everything else: ONE owning writer
  (`&mut self`), any number of `&self` readers. The bench/test
  `Shared`/`SharedCell` UnsafeCell harnesses uphold it structurally.
- `T: Copy` (seqlock slot). Non-Copy T (e.g. `Vec`) cannot use the
  slot-mirroring rule — an adaptive cell for non-Copy T would need
  owned reads to clone under pin, a different design.
- Demand is sticky per handle until `release_demand()`/Drop — a reader
  that pinned once keeps the cell hot (biased, intentional; hysteresis
  prevents flapping). Releasing demand while holding live pins is safe
  (test: `adaptive_release_demand_with_live_pin_is_safe`).
- `ReaderRegistry` slots are never recycled (monotonic `next`), so
  handle churn exhausts 64 acquires per registry lifetime. Fine for
  thread-per-handle; worth noting for adopting sessions writing tests.
- Cold-mode `read_pinned` fallback returns a copy (boxed) and relies on
  the writer writing again to escalate. A reader on a write-idle cold
  cell never gets zero-copy. Acceptable? (If not: escalation would need
  a reader-side publish, which breaks single-writer — don't.)
- Epochs in `Pinned` are c0-approximate (value may be up to one ctrl
  bump newer). Fine for missed-count heuristics, not for exactness.
- Not yet done for the prototype: ASan pass, 31-bit-style epoch
  overflow audit on ctrl (63:2 = plenty), loom/shuttle model checking
  of the transition protocol. The concurrent stress
  (`adaptive_concurrent_stress_mode_flapping`: 150 k writes, 3 readers
  toggling demand, lane-equality + epoch-monotonicity + final
  drop-reconciliation) passed 10/10 runs and caught the V1 UAF
  immediately, but it is not a proof.

## 6. File inventory

- `adaptive-cell-prototype.rs` — everything: module docs with the full
  safety argument, `AdaptiveRegistry`/`AdaptiveHandle`/`AdaptiveCell`/
  `Pinned`, 5 tests (`adaptive_*`), single-thread bench (`main`), MT
  bench (`main mt`), attribution probes left in deliberately.
- `non-blocking-memory.rs` — untouched.
