//! # AdaptiveCell<T> — escalation-hybrid prototype
//!
//! PROTOTYPE — not yet part of the main crate. Combines the two large-T
//! strategies behind one type, picking per-write:
//!
//!   COLD mode  = SeqCell-style seqlock. Inline slot, zero allocation,
//!                readers copy out under an odd/even counter bracket.
//!   HOT mode   = BridgedCell block path. One alloc per write; readers
//!                hold zero-copy `ReadRef`s pinned by the epoch-floor
//!                protocol, valid across any number of subsequent writes.
//!
//! The cell starts COLD and stays there while nobody needs a pinned view —
//! so the writer runs at seqlock speed with zero allocation. The first
//! time a reader calls `read_pinned`, it stores 1 into its per-handle
//! DEMAND slot (plain Release store — single writer per slot, no CAS).
//! The single writer scans demand slots before each write (gated behind
//! the registry's `has_any_reader()` flag, so cells never shared with a
//! reader pay one Acquire load) and ESCALATES: subsequent writes go
//! through the block path, and `read_pinned` returns zero-copy refs.
//! When demand disappears (handles released it or dropped) the writer
//! waits out a hysteresis window of `COOL_DOWN` writes, then
//! DE-ESCALATES back to the seqlock path and drains the retired list.
//!
//! ## The one new word: `ctrl`
//!
//! Readers cannot consult two atomics ("mode" + "seq") without a race, so
//! both live in one AtomicU64:
//!
//! ```text
//!   bit  0      MODE        0 = cold (slot), 1 = hot (block head)
//!   bit  1      IN-PROGRESS seqlock odd bit (cold writes only)
//!   bits 63:2   EPOCH       completed-write count
//! ```
//!
//! Cold write:  ctrl -> odd, write_volatile slot, ctrl -> even.
//! Hot write:   ctrl -> odd|MODE, write_volatile slot, Block publish via
//!              BridgedCell (Release), ctrl -> even|MODE.
//! Escalation   = first hot write (odd store sets MODE).
//! De-escalation = a cold write while MODE was 1 (odd store clears MODE;
//!              readers mid-flight fail validation and retry).
//!
//! THE SLOT IS CURRENT IN BOTH MODES. That is the load-bearing decision:
//! owned reads (`read`) are a pure seqlock copy from the slot in every
//! mode and NEVER dereference a Block. Blocks exist solely to serve
//! pinned views, and every Block dereference goes through the unmodified
//! floor protocol. (The first prototype cloned out of the head Block for
//! hot owned reads; since owned reads pin nothing, a concurrent
//! watermark sweep could free the Block mid-clone — caught immediately
//! by the stress test as a torn read of recycled heap. The double-write
//! costs one extra 128-byte volatile copy per hot write and removes the
//! entire class of unpinned Block dereferences.)
//!
//! ## Why the mode transition is safe (load/store only)
//!
//! * Owned readers validate with an exact ctrl match (classic seqlock) —
//!   any write or transition that lands mid-copy forces a retry.
//! * Pinned readers serve from the Block head only when ctrl is
//!   even-with-MODE (head is guaranteed fresh: the escalating write
//!   publishes its Block before storing even|MODE). After taking the
//!   ref they re-check that MODE is still set; a de-escalation that
//!   raced them drops the ref (releasing its floor) and retries cold.
//! * De-escalation NEVER frees the hot head block — it simply stops
//!   publishing through it. The block stays alive as the BridgedCell's
//!   head until a later hot write retires it, and retirement is freed
//!   only by `reclaim`, which honours reader floors. So every pointer a
//!   pinned reader can hold is protected by the *unmodified* BridgedCell
//!   floor protocol — this wrapper adds no new free sites.
//! * Demand is advisory, not safety-critical: a stale demand scan only
//!   means one more write in the "wrong" mode. `read_pinned` on a cold
//!   cell falls back to a seqlock copy (correct, just not zero-copy) and
//!   the writer escalates on its next write.
//!
//! Discipline preserved: `AtomicU64` load/store only on every hot path —
//! no CAS, no fetch_add. Demand slots are single-writer-per-slot like
//! the floor slots.
//!
//! ## Findings for crate integration (from bench attribution)
//!
//! 1. `ReadRefInner::Inline(T)` reserves `size_of::<T>()` inside every
//!    `ReadRef<T>` even though the Inline variant is only ever used for
//!    T ≤ 4 B — `ReadRef<[u64;16]>` is 184 B. `BridgedCell::read_ref`
//!    hides this via RVO (one in-place construction), but any wrapper
//!    that moves a ReadRef into another struct/enum pays a second
//!    ~200 B memcpy: measured 4.2 → 18.7 ns/op for the wrap alone.
//!    Storing the inline payload as 4 encoded bytes instead of T would
//!    shrink ReadRef to ~48 B and make composition free. Until then,
//!    an integrated AdaptiveCell should return its pinned view by
//!    constructing in place (i.e. live inside the crate, not wrap).
//! 2. The hot-write demand scan (≤64 Acquire loads) plus the extra
//!    slot copy cost ~3 ns/write vs raw BridgedCell::write — the price
//!    of being able to leave hot mode. Scanning every Nth write would
//!    amortise it at the cost of de-escalation latency.
//! 3. Multithreaded (see `mt` bench): under continuous pinning readers,
//!    block-path WRITER throughput is governed by coherence traffic from
//!    reader line-sharing, not by sweep policy (watermark vs amortised
//!    measured identical; throttling one reader recovered the writer
//!    5.3 -> 27.5 M/s). AdaptiveCell's hot writer sustains 3-5x the
//!    BridgedCell writer at 4-8 readers, apparently because the ctrl
//!    seqlock bracket makes readers yield while a write is in flight —
//!    an implicit writer-priority backoff BridgedCell readers have no
//!    signal for — at the cost of lower peak reader throughput (also
//!    depressed by finding 1's wrapper copies).
//! 4. All 64 ReaderRegistry floor slots pack into 8 cache lines (8
//!    slots/line) — neighbouring reader threads false-share their floor
//!    stores 3x per pinned read. Padding slots to one line each is a
//!    cheap likely win for read_ref scaling in the main crate.
//!
//! Build:
//!
//! ```sh
//! rustc --test adaptive-cell-prototype.rs -o adaptive-test && ./adaptive-test adaptive_
//! rustc -O adaptive-cell-prototype.rs -o adaptive-bench && ./adaptive-bench
//! ```

#![allow(dead_code)]

#[path = "non-blocking-memory.rs"]
mod nbm;

use std::cell::UnsafeCell;
use std::mem::MaybeUninit;
use std::ptr;
use std::sync::atomic::{AtomicU64, Ordering};

use nbm::{BridgedCell, ReadRef, ReadResult, ReaderHandle, ReaderRegistry, MAX_READERS};

/// ctrl bit 0: 0 = cold (seqlock slot), 1 = hot (block head).
const MODE_HOT: u64 = 1;
/// ctrl bit 1: seqlock in-progress (cold writes only).
const IN_PROGRESS: u64 = 2;
/// Hot writes with zero demand before the writer de-escalates.
/// Hysteresis so a reader briefly between two read_pinned calls doesn't
/// make the cell flap modes on every write.
const COOL_DOWN: u64 = 64;

#[inline]
fn ctrl_epoch(c: u64) -> u64 { c >> 2 }

#[inline]
fn missed_since(epoch: u64, last_epoch: u64) -> u64 {
    if last_epoch == 0 { 0 } else { epoch.saturating_sub(last_epoch + 1) }
}

// ---------------------------------------------------------------------------
// AdaptiveRegistry — ReaderRegistry plus per-reader DEMAND slots
// ---------------------------------------------------------------------------
//
// demand[i] is written ONLY by the handle owning slot i (store 0/1) and
// read by the single writer's scan — same single-writer-per-slot rule as
// the floor slots. No CAS anywhere.

pub struct AdaptiveRegistry {
    pub floors: ReaderRegistry,
    demand: [AtomicU64; MAX_READERS],
}

impl AdaptiveRegistry {
    pub fn new() -> Self {
        AdaptiveRegistry {
            floors: ReaderRegistry::new(),
            demand: std::array::from_fn(|_| AtomicU64::new(0)),
        }
    }

    /// One per reader thread for the thread's lifetime (slots are not
    /// recycled — same contract as ReaderRegistry::acquire).
    pub fn acquire(&self) -> AdaptiveHandle<'_> {
        let floors = self.floors.acquire();
        let idx = floors.slot_index();
        AdaptiveHandle { floors, demand: &self.demand[idx] }
    }

    /// Writer-side scan: does any live handle currently want pinned views?
    /// 64 cache-warm Acquire loads worst case, early exit on first hit.
    fn any_demand(&self) -> bool {
        self.demand.iter().any(|s| s.load(Ordering::Acquire) != 0)
    }
}

impl Default for AdaptiveRegistry {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// AdaptiveHandle — ReaderHandle plus this reader's demand slot
// ---------------------------------------------------------------------------

pub struct AdaptiveHandle<'r> {
    floors: ReaderHandle<'r>,
    demand: &'r AtomicU64,
}

impl<'r> AdaptiveHandle<'r> {
    /// Withdraw this reader's pin demand. Optional — Drop does it too.
    /// Allowed while still holding `Pinned` refs: those stay valid via
    /// the floor protocol; the cell may merely de-escalate around them.
    pub fn release_demand(&self) {
        self.demand.store(0, Ordering::Release);
    }
}

impl<'r> Drop for AdaptiveHandle<'r> {
    fn drop(&mut self) {
        // Demand slot clears first; the inner ReaderHandle's Drop then
        // clears the floor slot. Runs on panic unwind too.
        self.demand.store(0, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// Pinned<T> — what read_pinned returns
// ---------------------------------------------------------------------------

enum PinnedKind<T> {
    /// Zero-copy ref into a Block, floor-pinned (hot mode).
    Ref(ReadRef<T>),
    /// Seqlock copy fallback (cold mode; demand now signalled, the
    /// writer escalates on its next write). Boxed so `Pinned<T>` stays
    /// pointer-sized regardless of T — an inline `Copied(T)` made every
    /// hot-path return a sizeof(T) move (measured 4x slowdown at 128 B).
    /// The fallback is transitional (a handful of calls before the
    /// writer escalates), so its one allocation amortises to zero.
    Copied(Box<T>),
}

pub struct Pinned<T> {
    kind: PinnedKind<T>,
    /// Cell-level epoch (ctrl) at time of read.
    pub epoch: u64,
    pub missed: u64,
}

impl<T> Pinned<T> {
    /// True when this is a floor-pinned zero-copy view.
    pub fn is_zero_copy(&self) -> bool {
        matches!(self.kind, PinnedKind::Ref(_))
    }
}

impl<T> std::ops::Deref for Pinned<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match &self.kind {
            PinnedKind::Ref(r) => r,
            PinnedKind::Copied(v) => v,
        }
    }
}

// ---------------------------------------------------------------------------
// AdaptiveCell<T>
// ---------------------------------------------------------------------------

pub struct AdaptiveCell<T: Copy> {
    /// See module docs for bit layout. Monotone increasing.
    ctrl: AtomicU64,
    /// Cold-mode storage (seqlock slot). Valid whenever a cold write has
    /// completed and MODE is 0.
    slot: UnsafeCell<MaybeUninit<T>>,
    /// Hot-mode storage + retired list. Its head block stays allocated
    /// across cold periods (never freed by de-escalation).
    hot: BridgedCell<T>,
    /// Writer-private hysteresis counter (no atomicity needed).
    cool_streak: u64,
    /// Writer-private hot-write counter for amortised sweeping.
    hot_writes: u64,
}

/// Hot writes between reclaim attempts. A fixed write-count period
/// bounds sweep cost per write; under continuous pin churn the
/// length-watermark policy instead re-runs failing sweeps every write
/// (floor_min is pinned at 1 whenever any reader sits in its
/// conservative pre-publish window).
///
/// MEASURED HONESTLY: sweep policy is NOT what limits the writer under
/// concurrent pinning readers — watermark and amortised variants both
/// sit at ~1.2 M writes/s with 4-8 readers. The writer collapse is
/// cache-coherence coupling: every reader pass shares the head + Block
/// lines, so each writer store pays a cross-core RFO (probe: throttling
/// a single reader recovered the writer 5.3 -> 27.5 M/s in one run,
/// though that probe is high-variance on a shared VM). The policies
/// differ in what they spend instead: watermark burns sweep CPU to keep
/// retired_len ~100; amortised lets the backlog ride to ~10-60 k
/// entries (~MBs) between successful drains. Pick per deployment.
const SWEEP_PERIOD: u64 = 8192;

unsafe impl<T: Copy + Send> Send for AdaptiveCell<T> {}
unsafe impl<T: Copy + Send> Sync for AdaptiveCell<T> {}

impl<T: Copy> AdaptiveCell<T> {
    pub fn new() -> Self {
        AdaptiveCell {
            ctrl: AtomicU64::new(0),
            slot: UnsafeCell::new(MaybeUninit::uninit()),
            hot: BridgedCell::new(),
            cool_streak: 0,
            hot_writes: 0,
        }
    }

    /// Current mode — introspection for tests/telemetry.
    pub fn is_hot(&self) -> bool {
        self.ctrl.load(Ordering::Acquire) & MODE_HOT != 0
    }

    /// Completed writes.
    pub fn epoch(&self) -> u64 {
        ctrl_epoch(self.ctrl.load(Ordering::Acquire))
    }

    /// Retired-but-unfreed block count (test introspection).
    pub fn retired_len(&self) -> usize {
        self.hot.retired_len()
    }

    /// Manual sweep passthrough (writes already sweep opportunistically).
    pub fn reclaim(&mut self, reg: &AdaptiveRegistry) -> usize {
        self.hot.reclaim(&reg.floors)
    }

    /// BENCH-ONLY: raw inner read_ref, bypassing the adaptive wrapper.
    /// Used to attribute pinned-read overhead (wrapper vs floor protocol).
    #[doc(hidden)]
    pub fn bench_read_ref_raw(&self, handle: &AdaptiveHandle<'_>) -> Option<ReadRef<T>> {
        self.hot.read_ref(&handle.floors, 0)
    }

    // =========================================================================
    // WRITE — the adaptive decision point
    //
    // SAFETY: single owning writer only (same contract as Cell/SeqCell/
    // BridgedCell). Returns the new epoch.
    //
    // Decision: demand → hot. No demand → cold, except a hot cell waits
    // out COOL_DOWN writes (hysteresis) before de-escalating.
    //
    // Demand scan cost: cells never shared with any reader pay exactly
    // one Acquire load (`has_any_reader`, the write_lazy Level-B signal);
    // shared cells pay a ≤64-load early-exit scan.
    // =========================================================================
    pub unsafe fn write(&mut self, value: T, reg: &AdaptiveRegistry) -> u64 {
        // Single writer owns ctrl — Relaxed self-load is sufficient.
        let c = self.ctrl.load(Ordering::Relaxed);
        let cnt = c >> 1;
        debug_assert!(cnt & 1 == 0, "write() while a cold write is in progress");
        let hot_now = c & MODE_HOT != 0;

        let demand = reg.floors.has_any_reader() && reg.any_demand();
        let go_hot = if demand {
            self.cool_streak = 0;
            true
        } else if hot_now {
            self.cool_streak += 1;
            self.cool_streak < COOL_DOWN // hysteresis window
        } else {
            false
        };

        if go_hot {
            // HOT WRITE: seqlock bracket around BOTH stores — the slot
            // (kept current for owned readers in every mode) and the
            // Block publish (for pinned readers). Block publish happens
            // before the even|MODE store, so any reader that observes
            // even|MODE finds a fresh head.
            self.ctrl.store(((cnt + 1) << 1) | MODE_HOT, Ordering::Release);
            unsafe { ptr::write_volatile((*self.slot.get()).as_mut_ptr(), value) };
            unsafe { self.hot.write(value) };
            self.ctrl.store(((cnt + 2) << 1) | MODE_HOT, Ordering::Release);
            self.hot_writes += 1;
            if self.hot_writes % SWEEP_PERIOD == 0 {
                self.hot.reclaim(&reg.floors); // amortised — see SWEEP_PERIOD
            }
        } else {
            // COLD WRITE (stay cold / de-escalate).
            // The odd store also clears MODE in the same word — a pinned
            // reader mid-flight sees MODE=0 on validation and retries.
            self.ctrl.store((cnt + 1) << 1, Ordering::Release);
            unsafe { ptr::write_volatile((*self.slot.get()).as_mut_ptr(), value) };
            self.ctrl.store((cnt + 2) << 1, Ordering::Release);
            if hot_now {
                // De-escalation. The hot head block is NOT freed — it
                // remains BridgedCell's head (pinned readers keep full
                // floor protection). Drain whatever the floors allow.
                self.cool_streak = 0;
                self.hot.reclaim(&reg.floors);
            }
        }
        (cnt + 2) >> 1
    }

    // =========================================================================
    // READ — owned copy, any thread, pins nothing (materialise-out)
    //
    // Pure seqlock copy from the slot in EVERY mode (the slot is kept
    // current by hot writes too). Never dereferences a Block, so it can
    // never race a reclaim sweep — that is the safety story, not just an
    // optimisation. Exact-match validation handles writes and mode
    // transitions identically: anything that bumps ctrl forces a retry.
    // =========================================================================
    pub fn read(&self, last_epoch: u64) -> ReadResult<T> {
        loop {
            let c0 = self.ctrl.load(Ordering::Acquire);
            if c0 >> 1 == 0 {
                return ReadResult::Empty;
            }
            if c0 & IN_PROGRESS != 0 {
                std::thread::yield_now();
                continue; // write in flight (either mode)
            }
            let value = unsafe { ptr::read_volatile((*self.slot.get()).as_ptr()) };
            let c1 = self.ctrl.load(Ordering::Acquire);
            if c1 != c0 {
                std::thread::yield_now();
                continue; // write or transition landed mid-copy
            }
            let epoch = ctrl_epoch(c0);
            return ReadResult::Value { value, epoch, missed: missed_since(epoch, last_epoch) };
        }
    }

    // =========================================================================
    // READ_PINNED — zero-copy view when hot, copy fallback when cold
    //
    // Always signals demand FIRST (sticky on the handle until
    // release_demand / Drop) so the writer escalates even if this call
    // had to fall back to a copy. Service class degrades for at most the
    // calls issued before the writer's next write; safety never does.
    //
    // Hot path: delegate to BridgedCell::read_ref (unmodified floor
    // protocol), then confirm MODE is still hot. If a de-escalation
    // raced us the ref is dropped (releasing its floor) and we retry on
    // the cold path.
    // =========================================================================
    #[inline]
    pub fn read_pinned(&self, handle: &AdaptiveHandle<'_>, last_epoch: u64) -> Option<Pinned<T>> {
        // Sticky demand — single writer (this handle) for this slot.
        // Conditional: re-storing 1 every call keeps the line in M state
        // and stalls the writer's scan; a load of our own slot is ~free.
        if handle.demand.load(Ordering::Relaxed) == 0 {
            handle.demand.store(1, Ordering::Release);
        }

        loop {
            let c0 = self.ctrl.load(Ordering::Acquire);
            if c0 >> 1 == 0 {
                return None;
            }
            let epoch = ctrl_epoch(c0);
            let missed = missed_since(epoch, last_epoch);

            if c0 & MODE_HOT != 0 {
                if c0 & IN_PROGRESS != 0 {
                    // Hot write in flight. Don't serve from head during
                    // the ESCALATING write — head isn't published yet
                    // (or is stale from a previous hot period). Cheap to
                    // wait out the bracket uniformly.
                    std::thread::yield_now();
                    continue;
                }
                match self.hot.read_ref(&handle.floors, 0) {
                    Some(r) => {
                        let c1 = self.ctrl.load(Ordering::Acquire);
                        if c1 & MODE_HOT == 0 {
                            drop(r); // releases its floor entry
                            std::thread::yield_now();
                            continue;
                        }
                        return Some(Pinned { kind: PinnedKind::Ref(r), epoch, missed });
                    }
                    None => { std::thread::yield_now(); continue; } // transient
                }
            } else {
                if c0 & IN_PROGRESS != 0 {
                    std::thread::yield_now();
                    continue;
                }
                let value = unsafe { ptr::read_volatile((*self.slot.get()).as_ptr()) };
                let c1 = self.ctrl.load(Ordering::Acquire);
                if c1 != c0 {
                    std::thread::yield_now();
                    continue;
                }
                return Some(Pinned { kind: PinnedKind::Copied(Box::new(value)), epoch, missed });
            }
        }
    }
}

impl<T: Copy> Default for AdaptiveCell<T> {
    fn default() -> Self { Self::new() }
}

// ===========================================================================
// TESTS — run with: ./adaptive-test adaptive_
// ===========================================================================

#[cfg(test)]
mod adaptive_tests {
    use super::*;

    /// Test-only shared-mutability harness: one writer thread calls
    /// `writer_mut`, everyone else uses `&`. Same discipline the crate's
    /// own concurrent tests enforce via Mutex; here the single-writer
    /// rule is upheld by test structure instead so readers never block.
    struct Shared<T>(UnsafeCell<T>);
    unsafe impl<T> Send for Shared<T> {}
    unsafe impl<T> Sync for Shared<T> {}
    impl<T> Shared<T> {
        fn new(v: T) -> Self { Shared(UnsafeCell::new(v)) }
        fn get(&self) -> &T { unsafe { &*self.0.get() } }
        #[allow(clippy::mut_from_ref)]
        unsafe fn writer_mut(&self) -> &mut T { unsafe { &mut *self.0.get() } }
    }

    #[test]
    fn adaptive_cold_stays_cold_and_correct() {
        let reg = AdaptiveRegistry::new();
        let mut cell: AdaptiveCell<u64> = AdaptiveCell::new();

        assert_eq!(cell.read(0), ReadResult::Empty);

        let e = unsafe { cell.write(7, &reg) };
        assert_eq!(e, 1);
        assert!(!cell.is_hot());
        match cell.read(0) {
            ReadResult::Value { value, epoch, missed } => {
                assert_eq!(value, 7);
                assert_eq!(epoch, 1);
                assert_eq!(missed, 0);
            }
            _ => panic!("expected value"),
        }

        for i in 0..100u64 {
            unsafe { cell.write(i, &reg) };
        }
        assert!(!cell.is_hot(), "no demand → must never escalate");
        assert_eq!(cell.epoch(), 101);
        assert_eq!(cell.retired_len(), 0, "cold writes must not allocate blocks");
        assert_eq!(cell.read(0).value(), Some(99));
    }

    #[test]
    fn adaptive_escalates_pins_and_deescalates() {
        let reg = AdaptiveRegistry::new();
        let mut cell: AdaptiveCell<[u64; 8]> = AdaptiveCell::new();

        unsafe { cell.write([1; 8], &reg) };

        {
            let handle = reg.acquire();

            // Cold cell: fallback copy + demand signalled.
            let p1 = cell.read_pinned(&handle, 0).unwrap();
            assert!(!p1.is_zero_copy());
            assert_eq!(*p1, [1; 8]);
            drop(p1);

            // Writer sees demand on its next write → escalates.
            unsafe { cell.write([2; 8], &reg) };
            assert!(cell.is_hot());

            // Now zero-copy.
            let p2 = cell.read_pinned(&handle, 0).unwrap();
            assert!(p2.is_zero_copy());
            assert_eq!(*p2, [2; 8]);

            // Pinned view stays byte-identical across 50 overwrites.
            for i in 3..53u64 {
                unsafe { cell.write([i; 8], &reg) };
            }
            assert_eq!(*p2, [2; 8], "pinned view must not move under the reader");
            assert_eq!(cell.read(0).value(), Some([52; 8]), "owned read sees latest");
            assert!(cell.retired_len() > 0, "overwrites of a pinned block must retire");

            // Release the pin → everything reclaims.
            drop(p2);
            cell.reclaim(&reg);
            assert_eq!(cell.retired_len(), 0);

            // Handle (and its demand) drops here.
        }

        // No demand → writer cools down over COOL_DOWN writes, then flips cold.
        for i in 0..(COOL_DOWN + 2) {
            unsafe { cell.write([100 + i; 8], &reg) };
        }
        assert!(!cell.is_hot(), "demand gone → must de-escalate");
        assert_eq!(cell.retired_len(), 0, "de-escalation sweep must drain retired list");
        assert_eq!(cell.read(0).value(), Some([100 + COOL_DOWN + 1; 8]));

        // And it can escalate again later (no one-way ratchet).
        let handle2 = reg.acquire();
        let _ = cell.read_pinned(&handle2, 0).unwrap();
        unsafe { cell.write([777; 8], &reg) };
        assert!(cell.is_hot());
        let p = cell.read_pinned(&handle2, 0).unwrap();
        assert!(p.is_zero_copy());
        assert_eq!(*p, [777; 8]);
    }

    #[test]
    fn adaptive_empty_read_pinned_is_none() {
        let reg = AdaptiveRegistry::new();
        let cell: AdaptiveCell<u64> = AdaptiveCell::new();
        let handle = reg.acquire();
        assert!(cell.read_pinned(&handle, 0).is_none());
        // Demand was still signalled — first write escalates immediately.
        let mut cell = cell;
        unsafe { cell.write(5, &reg) };
        assert!(cell.is_hot());
    }

    #[test]
    fn adaptive_release_demand_with_live_pin_is_safe() {
        let reg = AdaptiveRegistry::new();
        let mut cell: AdaptiveCell<[u64; 8]> = AdaptiveCell::new();
        let handle = reg.acquire();

        unsafe { cell.write([1; 8], &reg) };
        let _ = cell.read_pinned(&handle, 0); // demand on
        unsafe { cell.write([2; 8], &reg) }; // hot
        let p = cell.read_pinned(&handle, 0).unwrap();
        assert!(p.is_zero_copy());

        // Withdraw demand while the pin is live → cell de-escalates, but
        // the pinned block must survive (floor protocol, not demand,
        // guards memory).
        handle.release_demand();
        for i in 0..(COOL_DOWN + 2) {
            unsafe { cell.write([10 + i; 8], &reg) };
        }
        assert!(!cell.is_hot());
        assert_eq!(*p, [2; 8], "pin must outlive de-escalation");

        drop(p);
        cell.reclaim(&reg);
        assert_eq!(cell.retired_len(), 0);
    }

    /// Adversarial: 1 writer flat out, 3 readers mixing owned reads and
    /// pinned reads while toggling demand → the cell flaps between modes
    /// under load. Lanes-equal asserts catch torn reads; epoch
    /// monotonicity catches time travel; the final sweep catches leaks
    /// of reclaimable blocks.
    #[test]
    fn adaptive_concurrent_stress_mode_flapping() {
        const WRITES: u64 = 150_000;
        let reg = AdaptiveRegistry::new();
        let shared = Shared::new(AdaptiveCell::<[u64; 4]>::new());
        let done = AtomicU64::new(0);

        std::thread::scope(|s| {
            // Writer
            s.spawn(|| {
                let cell = unsafe { shared.writer_mut() };
                for i in 1..=WRITES {
                    unsafe { cell.write([i; 4], &reg) };
                }
                done.store(1, Ordering::Release);
            });

            // Readers
            for r in 0..3 {
                let regr = &reg;
                let sharedr = &shared;
                let doner = &done;
                s.spawn(move || {
                    let handle = regr.acquire();
                    let mut last_epoch = 0u64;
                    let mut iter = 0u64;
                    while doner.load(Ordering::Acquire) == 0 {
                        iter += 1;
                        if iter % 64 == r {
                            // pinned read, held across a few spins
                            if let Some(p) = sharedr.get().read_pinned(&handle, last_epoch) {
                                let v = *p;
                                assert!(v[0] == v[1] && v[1] == v[2] && v[2] == v[3],
                                    "torn pinned read: {v:?}");
                                std::hint::spin_loop();
                                assert_eq!(*p, v, "pinned view moved");
                                assert!(p.epoch >= last_epoch, "epoch went backwards");
                                last_epoch = p.epoch;
                            }
                            if iter % 1024 == 0 {
                                handle.release_demand(); // force mode flapping
                            }
                        } else if let ReadResult::Value { value: v, epoch, .. } =
                            sharedr.get().read(last_epoch)
                        {
                            assert!(v[0] == v[1] && v[1] == v[2] && v[2] == v[3],
                                "torn owned read: {v:?}");
                            assert!(epoch >= last_epoch, "epoch went backwards");
                            last_epoch = epoch;
                        }
                    }
                });
            }
        });

        // All readers (and their demand + floors) are gone.
        let cell = unsafe { shared.writer_mut() };
        assert_eq!(cell.read(0).value(), Some([WRITES; 4]));
        assert_eq!(cell.epoch(), WRITES);
        cell.reclaim(&reg);
        assert_eq!(cell.retired_len(), 0, "leak: retired blocks survived final sweep");
    }
}

// ===========================================================================
// MINI-BENCH — build -O. No args = single-thread suite. `mt` = multithread.
// ===========================================================================

/// Bench-only shared-mutability harness (same discipline as the test one):
/// exactly one thread calls writer_mut, everyone else uses get().
#[cfg(not(test))]
struct SharedCell<T>(UnsafeCell<T>);
#[cfg(not(test))]
unsafe impl<T> Send for SharedCell<T> {}
#[cfg(not(test))]
unsafe impl<T> Sync for SharedCell<T> {}
#[cfg(not(test))]
impl<T> SharedCell<T> {
    fn new(v: T) -> Self { SharedCell(UnsafeCell::new(v)) }
    fn get(&self) -> &T { unsafe { &*self.0.get() } }
    #[allow(clippy::mut_from_ref)]
    unsafe fn writer_mut(&self) -> &mut T { unsafe { &mut *self.0.get() } }
}

#[cfg(not(test))]
mod mt_bench {
    use super::*;
    use std::hint::black_box;
    use std::time::{Duration, Instant};

    pub type T16 = [u64; 16];
    const DUR: Duration = Duration::from_millis(300);

    pub struct Rates {
        pub writes_m: f64,
        pub reads_m: f64,
    }

    fn row(name: &str, readers: usize, r: Rates) {
        println!("  {name:<44} {readers:>2} readers   writer {:>7.1} M/s   readers {:>8.1} M/s",
            r.writes_m, r.reads_m);
    }

    /// Generic harness: spawns one writer closure + N reader closures,
    /// runs for DUR, returns summed op counts as M ops/s.
    fn run<W, R>(readers: usize, writer: W, reader: R) -> Rates
    where
        W: FnOnce(&AtomicU64) -> u64 + Send,
        R: Fn(&AtomicU64) -> u64 + Send + Sync,
    {
        let stop = AtomicU64::new(0);
        let t = Instant::now();
        let (w, r, dur) = std::thread::scope(|s| {
            let wh = s.spawn(|| writer(&stop));
            let rhs: Vec<_> = (0..readers).map(|_| s.spawn(|| reader(&stop))).collect();
            std::thread::sleep(DUR);
            stop.store(1, Ordering::Release);
            let w = wh.join().unwrap();
            let r: u64 = rhs.into_iter().map(|h| h.join().unwrap()).sum();
            (w, r, t.elapsed().as_secs_f64())
        });
        Rates { writes_m: w as f64 / dur / 1e6, reads_m: r as f64 / dur / 1e6 }
    }

    // ── Scenario A: OWNED reads under full write pressure ──────────────────
    //
    // BridgedCell is deliberately absent: its owned read clones from the
    // head Block without pinning, which is only sound when reads cannot
    // overlap reclaim — under a free-running writer that contract cannot
    // be met (the exact hazard the AdaptiveCell stress test caught).

    pub fn owned_adaptive(readers: usize) -> Rates {
        let reg = AdaptiveRegistry::new();
        let cell = SharedCell::new(AdaptiveCell::<T16>::new());
        unsafe { cell.writer_mut().write([1; 16], &reg) };
        run(readers,
            |stop| {
                let c = unsafe { cell.writer_mut() };
                let (mut i, mut n) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    i += 1;
                    unsafe { c.write(black_box([i; 16]), &reg) };
                    n += 1;
                }
                n
            },
            |stop| {
                let (mut n, mut acc) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    if let Some(v) = cell.get().read(0).value() { acc ^= v[0]; }
                    n += 1;
                }
                black_box(acc);
                n
            })
    }

    pub fn owned_seqcell(readers: usize) -> Rates {
        let cell = nbm::SeqCell::<T16>::new();
        unsafe { cell.write([1; 16]) };
        run(readers,
            |stop| {
                let (mut i, mut n) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    i += 1;
                    unsafe { cell.write(black_box([i; 16])) };
                    n += 1;
                }
                n
            },
            |stop| {
                let (mut n, mut acc) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    if let Some(v) = cell.read(0).value() { acc ^= v[0]; }
                    n += 1;
                }
                black_box(acc);
                n
            })
    }

    pub fn owned_rwlock(readers: usize) -> Rates {
        let cell = std::sync::RwLock::new([1u64; 16]);
        run(readers,
            |stop| {
                let (mut i, mut n) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    i += 1;
                    *cell.write().unwrap() = black_box([i; 16]);
                    n += 1;
                }
                n
            },
            |stop| {
                let (mut n, mut acc) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    acc ^= cell.read().unwrap()[0];
                    n += 1;
                }
                black_box(acc);
                n
            })
    }

    // ── Scenario B: PINNED (zero-copy) reads under full write pressure ─────
    // The RCU workload: every read takes a stable view, iterates it, drops.

    pub fn pinned_adaptive(readers: usize) -> Rates {
        let reg = AdaptiveRegistry::new();
        let cell = SharedCell::new(AdaptiveCell::<T16>::new());
        unsafe { cell.writer_mut().write([1; 16], &reg) };
        let rates = run(readers,
            |stop| {
                let c = unsafe { cell.writer_mut() };
                let (mut i, mut n) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    i += 1;
                    unsafe { c.write(black_box([i; 16]), &reg) };
                    n += 1;
                }
                n
            },
            |stop| {
                let handle = reg.acquire();
                let (mut n, mut acc) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    if let Some(p) = cell.get().read_pinned(&handle, 0) {
                        acc ^= p[0] ^ p[15]; // touch both ends of the view
                    }
                    n += 1;
                }
                black_box(acc);
                n
            });
        println!("      (final retired_len = {})", cell.get().retired_len());
        rates
    }

    pub fn pinned_bridged(readers: usize) -> Rates {
        let reg = ReaderRegistry::new();
        let cell = SharedCell::new(nbm::BridgedCell::<T16>::new());
        unsafe { cell.writer_mut().write([1; 16]) };
        let rates = run(readers,
            |stop| {
                let c = unsafe { cell.writer_mut() };
                let (mut i, mut n) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    i += 1;
                    unsafe { c.write(black_box([i; 16])) };
                    c.reclaim_if_watermark(&reg);
                    n += 1;
                }
                n
            },
            |stop| {
                let handle = reg.acquire();
                let (mut n, mut acc) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    if let Some(r) = cell.get().read_ref(&handle, 0) {
                        acc ^= r[0] ^ r[15];
                    }
                    n += 1;
                }
                black_box(acc);
                n
            });
        println!("      (final retired_len = {})", cell.get().retired_len());
        rates
    }

    /// Attribution probe: identical to pinned_bridged except the writer
    /// sweeps every SWEEP_PERIOD writes instead of on the length
    /// watermark. If the BridgedCell writer collapse is the watermark
    /// policy (failing sweeps re-running every write over contended
    /// floor slots), this variant recovers the writer rate.
    pub fn pinned_bridged_amortised(readers: usize) -> Rates {
        let reg = ReaderRegistry::new();
        let cell = SharedCell::new(nbm::BridgedCell::<T16>::new());
        unsafe { cell.writer_mut().write([1; 16]) };
        let rates = run(readers,
            |stop| {
                let c = unsafe { cell.writer_mut() };
                let (mut i, mut n) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    i += 1;
                    unsafe { c.write(black_box([i; 16])) };
                    n += 1;
                    if n % SWEEP_PERIOD == 0 { c.reclaim(&reg); }
                }
                n
            },
            |stop| {
                let handle = reg.acquire();
                let (mut n, mut acc) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    if let Some(r) = cell.get().read_ref(&handle, 0) {
                        acc ^= r[0] ^ r[15];
                    }
                    n += 1;
                }
                black_box(acc);
                n
            });
        println!("      (final retired_len = {})", cell.get().retired_len());
        rates
    }

    /// Attribution probe 2: same as pinned_bridged but readers pause
    /// ~32 spin-loop iterations between reads. If the writer collapse is
    /// coherence traffic from reader line-sharing (not sweep policy),
    /// throttled readers recover the writer rate.
    pub fn pinned_bridged_throttled(readers: usize) -> Rates {
        let reg = ReaderRegistry::new();
        let cell = SharedCell::new(nbm::BridgedCell::<T16>::new());
        unsafe { cell.writer_mut().write([1; 16]) };
        run(readers,
            |stop| {
                let c = unsafe { cell.writer_mut() };
                let (mut i, mut n) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    i += 1;
                    unsafe { c.write(black_box([i; 16])) };
                    n += 1;
                    if n % SWEEP_PERIOD == 0 { c.reclaim(&reg); }
                }
                n
            },
            |stop| {
                let handle = reg.acquire();
                let (mut n, mut acc) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    if let Some(r) = cell.get().read_ref(&handle, 0) {
                        acc ^= r[0] ^ r[15];
                    }
                    for _ in 0..32 { std::hint::spin_loop(); }
                    n += 1;
                }
                black_box(acc);
                n
            })
    }

    pub fn pinned_rwlock(readers: usize) -> Rates {
        let cell = std::sync::RwLock::new([1u64; 16]);
        run(readers,
            |stop| {
                let (mut i, mut n) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    i += 1;
                    *cell.write().unwrap() = black_box([i; 16]);
                    n += 1;
                }
                n
            },
            |stop| {
                let (mut n, mut acc) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    let g = cell.read().unwrap(); // guard = the "pin"
                    acc ^= g[0] ^ g[15];
                    drop(g);
                    n += 1;
                }
                black_box(acc);
                n
            })
    }

    // ── Scenario C: MIXED — pin demand comes and goes, 50 ms phases ────────
    //
    // Readers alternate every 50 ms between an RCU phase (pinned reads)
    // and a sampling phase (owned reads, demand released). AdaptiveCell
    // flaps hot/cold with the phases; BridgedCell has only the block
    // path, so its readers use read_ref in both phases (its only
    // reclaim-safe concurrent read).

    pub fn mixed_adaptive(readers: usize) -> Rates {
        let reg = AdaptiveRegistry::new();
        let cell = SharedCell::new(AdaptiveCell::<T16>::new());
        unsafe { cell.writer_mut().write([1; 16], &reg) };
        run(readers,
            |stop| {
                let c = unsafe { cell.writer_mut() };
                let (mut i, mut n) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    i += 1;
                    unsafe { c.write(black_box([i; 16]), &reg) };
                    n += 1;
                }
                n
            },
            |stop| {
                let handle = reg.acquire();
                let t = Instant::now();
                let (mut n, mut acc) = (0u64, 0u64);
                let mut was_pinning = true;
                while stop.load(Ordering::Acquire) == 0 {
                    let pin_phase = (t.elapsed().as_millis() / 50) % 2 == 0;
                    if pin_phase {
                        if let Some(p) = cell.get().read_pinned(&handle, 0) {
                            acc ^= p[0] ^ p[15];
                        }
                        was_pinning = true;
                    } else {
                        if was_pinning {
                            handle.release_demand();
                            was_pinning = false;
                        }
                        if let Some(v) = cell.get().read(0).value() { acc ^= v[0]; }
                    }
                    n += 1;
                }
                black_box(acc);
                n
            })
    }

    pub fn mixed_bridged(readers: usize) -> Rates {
        let reg = ReaderRegistry::new();
        let cell = SharedCell::new(nbm::BridgedCell::<T16>::new());
        unsafe { cell.writer_mut().write([1; 16]) };
        run(readers,
            |stop| {
                let c = unsafe { cell.writer_mut() };
                let (mut i, mut n) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    i += 1;
                    unsafe { c.write(black_box([i; 16])) };
                    c.reclaim_if_watermark(&reg);
                    n += 1;
                }
                n
            },
            |stop| {
                let handle = reg.acquire();
                let (mut n, mut acc) = (0u64, 0u64);
                while stop.load(Ordering::Acquire) == 0 {
                    if let Some(r) = cell.get().read_ref(&handle, 0) {
                        acc ^= r[0] ^ r[15];
                    }
                    n += 1;
                }
                black_box(acc);
                n
            })
    }

    pub fn main_mt() {
        println!("\nAdaptiveCell MT bench — T = [u64; 16] (128 B), 1 writer flat-out,");
        println!("{} ms per measurement, {} cores\n",
            DUR.as_millis(), std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));

        println!("A. OWNED reads under write pressure (copy-out, pins nothing):");
        for &r in &[1usize, 4, 8] {
            row("AdaptiveCell::read (cold seqlock)", r, owned_adaptive(r));
            row("SeqCell::read", r, owned_seqcell(r));
            row("RwLock<T> (std baseline)", r, owned_rwlock(r));
            println!();
        }

        println!("B. PINNED zero-copy reads under write pressure (RCU workload):");
        for &r in &[1usize, 4, 8] {
            row("AdaptiveCell::read_pinned (hot)", r, pinned_adaptive(r));
            row("BridgedCell::read_ref (watermark sweep)", r, pinned_bridged(r));
            row("BridgedCell::read_ref (amortised sweep)", r, pinned_bridged_amortised(r));
            row("BridgedCell::read_ref (throttled readers)", r, pinned_bridged_throttled(r));
            row("RwLock<T> guard held (std baseline)", r, pinned_rwlock(r));
            println!();
        }

        println!("C. MIXED — readers alternate 50 ms RCU / 50 ms sampling phases:");
        for &r in &[4usize, 8] {
            row("AdaptiveCell (flaps hot/cold with phases)", r, mixed_adaptive(r));
            row("BridgedCell (block path in both phases)", r, mixed_bridged(r));
            println!();
        }
    }
}

#[cfg(not(test))]
fn main() {
    if std::env::args().any(|a| a == "mt") {
        mt_bench::main_mt();
        return;
    }
    use std::hint::black_box;
    use std::time::Instant;

    type T16 = [u64; 16]; // 128-byte payload

    fn bench<F: FnMut()>(name: &str, iters: u64, mut f: F) -> f64 {
        // tiny warmup
        for _ in 0..(iters / 20).max(1) { f(); }
        let t = Instant::now();
        for _ in 0..iters { f(); }
        let ns = t.elapsed().as_nanos() as f64 / iters as f64;
        println!("  {name:<58} {ns:>8.1} ns/op  ({:>7.1} M ops/s)", 1000.0 / ns);
        ns
    }

    println!("\nAdaptiveCell prototype bench — T = [u64; 16] (128 B)\n");

    // ── 1. Write throughput, no reader has ever existed ────────────────────
    println!("1. WRITE, no pin demand (cold home turf):");
    {
        let seq = nbm::SeqCell::<T16>::new();
        let mut i = 0u64;
        bench("SeqCell::write (pure seqlock)", 3_000_000, || {
            i += 1;
            unsafe { seq.write(black_box([i; 16])) };
        });
    }
    {
        let reg = AdaptiveRegistry::new();
        let mut cell = AdaptiveCell::<T16>::new();
        let mut i = 0u64;
        bench("AdaptiveCell::write (cold)", 3_000_000, || {
            i += 1;
            unsafe { cell.write(black_box([i; 16]), &reg) };
        });
        assert!(!cell.is_hot());
    }
    {
        let reg = ReaderRegistry::new();
        let mut cell = nbm::BridgedCell::<T16>::new();
        let mut i = 0u64;
        bench("BridgedCell::write_lazy (alloc+free every write)", 1_500_000, || {
            i += 1;
            unsafe { cell.write_lazy(black_box([i; 16]), &reg) };
        });
    }

    // ── 2. Write throughput, pin demand present ─────────────────────────────
    println!("\n2. WRITE, pin demand present (hot home turf):");
    {
        let reg = AdaptiveRegistry::new();
        let mut cell = AdaptiveCell::<T16>::new();
        let handle = reg.acquire();
        unsafe { cell.write([1; 16], &reg) };
        let _ = cell.read_pinned(&handle, 0); // demand on
        let mut i = 0u64;
        bench("AdaptiveCell::write (hot, retire+watermark sweep)", 1_500_000, || {
            i += 1;
            unsafe { cell.write(black_box([i; 16]), &reg) };
        });
        assert!(cell.is_hot());
    }
    {
        let reg = ReaderRegistry::new();
        let _handle = reg.acquire(); // a reader exists, floors idle
        let mut cell = nbm::BridgedCell::<T16>::new();
        let mut i = 0u64;
        bench("BridgedCell::write + watermark sweep", 1_500_000, || {
            i += 1;
            unsafe { cell.write(black_box([i; 16])) };
            cell.reclaim_if_watermark(&reg);
        });
    }

    // ── 3. Owned reads ──────────────────────────────────────────────────────
    println!("\n3. OWNED READ (no concurrent writer):");
    {
        let seq = nbm::SeqCell::<T16>::new();
        unsafe { seq.write([9; 16]) };
        bench("SeqCell::read", 5_000_000, || {
            black_box(seq.read(0).value());
        });
    }
    {
        let reg = AdaptiveRegistry::new();
        let mut cell = AdaptiveCell::<T16>::new();
        unsafe { cell.write([9; 16], &reg) };
        bench("AdaptiveCell::read (cold)", 5_000_000, || {
            black_box(cell.read(0).value());
        });
        let handle = reg.acquire();
        let _ = cell.read_pinned(&handle, 0);
        unsafe { cell.write([9; 16], &reg) };
        assert!(cell.is_hot());
        bench("AdaptiveCell::read (hot)", 5_000_000, || {
            black_box(cell.read(0).value());
        });
    }
    {
        let mut cell = nbm::BridgedCell::<T16>::new();
        unsafe { cell.write([9; 16]) };
        bench("BridgedCell::read", 5_000_000, || {
            black_box(cell.read(0).value());
        });
    }

    // ── 4. Pinned-read churn ────────────────────────────────────────────────
    println!("\n4. PINNED READ acquire+drop (SeqCell cannot do this at all):");
    {
        let reg = AdaptiveRegistry::new();
        let mut cell = AdaptiveCell::<T16>::new();
        let handle = reg.acquire();
        unsafe { cell.write([9; 16], &reg) };
        let _ = cell.read_pinned(&handle, 0);
        unsafe { cell.write([9; 16], &reg) };
        bench("AdaptiveCell::read_pinned (hot, zero-copy)", 5_000_000, || {
            black_box(cell.read_pinned(&handle, 0));
        });
        bench("AdaptiveCell raw inner read_ref (attribution probe)", 5_000_000, || {
            black_box(cell.bench_read_ref_raw(&handle));
        });
        bench("raw read_ref + Pinned wrap (attribution probe B)", 5_000_000, || {
            black_box(cell.bench_read_ref_raw(&handle).map(|r| Pinned {
                kind: PinnedKind::Ref(r),
                epoch: 1,
                missed: 0,
            }));
        });
        println!("    sizeof ReadRef<T16> = {} B, Pinned<T16> = {} B  — the gap above is",
            std::mem::size_of::<ReadRef<T16>>(), std::mem::size_of::<Pinned<T16>>());
        println!("    pure move cost: ReadRefInner::Inline(T) reserves sizeof(T) inline,");
        println!("    so re-wrapping a ReadRef defeats RVO and double-copies ~200 B.");
    }
    {
        let reg = ReaderRegistry::new();
        let handle = reg.acquire();
        let mut cell = nbm::BridgedCell::<T16>::new();
        unsafe { cell.write([9; 16]) };
        bench("BridgedCell::read_ref (zero-copy)", 5_000_000, || {
            black_box(cell.read_ref(&handle, 0));
        });
    }

    // ── 5. THE HEADLINE: mixed workload, demand comes and goes ─────────────
    //
    // 20 phases × 200k writes. Even phases: no pin demand. Odd phases: a
    // reader wants pinned views. The adaptive cell runs each phase in the
    // right mode; the pure strategies are stuck with one. (SeqCell is the
    // absolute write-speed floor but CANNOT serve a pinned view at all —
    // included as the reference line, not as a competitor.)
    println!("\n5. MIXED WORKLOAD writer throughput (20 phases x 200k writes,");
    println!("   pin demand present in odd phases only):");
    const PHASES: u64 = 20;
    const PER_PHASE: u64 = 200_000;
    {
        let reg = AdaptiveRegistry::new();
        let mut cell = AdaptiveCell::<T16>::new();
        let handle = reg.acquire();
        unsafe { cell.write([0; 16], &reg) };
        let t = Instant::now();
        let mut i = 0u64;
        for phase in 0..PHASES {
            if phase % 2 == 1 {
                let _ = cell.read_pinned(&handle, 0); // demand on
            } else {
                handle.release_demand(); // demand off
            }
            for _ in 0..PER_PHASE {
                i += 1;
                unsafe { cell.write(black_box([i; 16]), &reg) };
            }
        }
        let ns = t.elapsed().as_nanos() as f64 / (PHASES * PER_PHASE) as f64;
        println!("  {:<58} {ns:>8.1} ns/op  ({:>7.1} M ops/s)",
            "AdaptiveCell (escalates/de-escalates per phase)", 1000.0 / ns);
    }
    {
        let reg = ReaderRegistry::new();
        let _handle = reg.acquire();
        let mut cell = nbm::BridgedCell::<T16>::new();
        let t = Instant::now();
        let mut i = 0u64;
        for _ in 0..PHASES * PER_PHASE {
            i += 1;
            unsafe { cell.write_lazy(black_box([i; 16]), &reg) };
            cell.reclaim_if_watermark(&reg);
        }
        let ns = t.elapsed().as_nanos() as f64 / (PHASES * PER_PHASE) as f64;
        println!("  {:<58} {ns:>8.1} ns/op  ({:>7.1} M ops/s)",
            "BridgedCell (always block path — allocs in every phase)", 1000.0 / ns);
    }
    {
        let seq = nbm::SeqCell::<T16>::new();
        let t = Instant::now();
        let mut i = 0u64;
        for _ in 0..PHASES * PER_PHASE {
            i += 1;
            unsafe { seq.write(black_box([i; 16])) };
        }
        let ns = t.elapsed().as_nanos() as f64 / (PHASES * PER_PHASE) as f64;
        println!("  {:<58} {ns:>8.1} ns/op  ({:>7.1} M ops/s)",
            "SeqCell (reference floor — CANNOT serve pinned views)", 1000.0 / ns);
    }
    println!();
}
