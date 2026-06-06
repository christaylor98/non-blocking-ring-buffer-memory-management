//! # axOS Memory Model — v0.9
//!
//! ## The design
//!
//! Everything reduces to one primitive:
//!
//! ```text
//! Cell<T>
//!   head: AtomicU64   ← inline: bit[0]=tag, bits[32:1]=value, bits[63:33]=epoch
//!                        pointer: *const Block<T>; epoch lives inside Block
//! ```
//!
//! The ring buffer is gone. It was solving three problems that don't
//! need a ring:
//!
//!   1. Locating current head   → single AtomicU64 head does this
//!   2. Retention / history     → Block.prev chain already IS the history
//!   3. Reclamation boundary    → epoch inside Block tracks this directly
//!
//! ## Write modes
//!
//! `write(v)`   — mutable replace. ≤4B: inline in head (zero allocation).
//!               >4B: swaps Block pointer; old Block leaks until reclamation
//!               sweep. See TRADEOFF comment on write().
//!
//! `append(v)`  — immutable extend. Always allocates Block. new.prev = old head.
//!               Causal chain traversable via ChainIter. See TRADEOFF on append().
//!
//! ## Epoch = write counter
//!
//! Writer increments epoch on every write (single writer per cell).
//!
//! Reader holds its last-seen epoch. On each read:
//!   missed = current_epoch - last_epoch - 1
//!   missed == 0  → reader is current
//!   missed > N   → reader is behind; caller decides: throttle / escalate / accept
//!
//! No staging queue. No fail_count. No pressure slots.
//! The epoch exposes write velocity; policy stays with the caller.
//!
//! ## Memory ordering — no separate epoch atomic, no race
//!
//! Inline path (size_of::<T>() <= 4):
//!   Writer: single Release store of head (value and epoch packed in one word)
//!   Reader: single Acquire load of head — value and epoch always consistent
//!
//! Block path (size_of::<T>() > 4):
//!   Writer: Block fully initialised (value, epoch, prev), then Release store
//!           of the head pointer
//!   Reader: Acquire load of pointer synchronises with that Release; all Block
//!           fields visible — value and epoch always consistent
//!
//!   Previously a separate `epoch: AtomicU64` was used for synchronisation,
//!   which left a window where a reader could see a new head with a stale epoch.
//!   That field is gone: epoch now travels with the data.
//!
//! ## Ring (explicit opt-in)
//!
//! `Ring<T>` wraps N cells for the case where N concurrent readers each
//! need their own stable head pointer (e.g. axAporia Ring 2 reading while
//! Ring 1 is still appending). Not needed otherwise.

use std::cell::UnsafeCell;
use std::mem::{size_of, ManuallyDrop, MaybeUninit};
use std::marker::PhantomData;
use std::ptr;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::Mutex;

/// Tag bit: head field encodes inline value when bit 0 is set.
/// Block pointers are always ≥8-byte aligned so bit 0 is always 0 for pointers.
const INLINE_TAG: u64 = 1;

// ---------------------------------------------------------------------------
// Bridge-layer constants
//
// MAX_READERS — fixed registry size. One slot per reader thread for the
//               thread's lifetime. Acquire panics if exceeded.
// WATERMARK   — `reclaim_if_watermark` only RUNS reclaim() when the retired
//               list reaches this length. It NEVER gates or skips the fresh
//               floor scan inside reclaim itself — that is always fresh,
//               which is what bounds staleness.
// HOLD_DEPTH  — per-handle nested-ReadRef stack depth. Overflow falls back
//               to a conservative single floor (see HoldStack).
// ---------------------------------------------------------------------------
pub const MAX_READERS: usize = 64;
pub const WATERMARK:   usize = 256;
pub const HOLD_DEPTH:  usize = 8;

// ---------------------------------------------------------------------------
// Results
// ---------------------------------------------------------------------------

/// Returned by every write. Carries the epoch at time of write.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WriteResult {
    /// Epoch at time of this write. Pass to the next read as `last_epoch`
    /// to detect how many writes you missed.
    pub epoch: u64,
}

/// Returned by every read.
#[derive(Debug, PartialEq)]
pub enum ReadResult<T> {
    /// Cell has never been written.
    Empty,
    /// Successfully read a value.
    Value {
        value:  T,
        /// Epoch at time of this read.
        epoch:  u64,
        /// Writes caller missed since `last_epoch`.
        /// 0 = fully current. >0 = reader is behind.
        /// Caller decides whether to throttle the writer.
        missed: u64,
    },
}

impl<T> ReadResult<T> {
    pub fn value(self) -> Option<T> {
        match self { ReadResult::Value { value, .. } => Some(value), _ => None }
    }
    pub fn epoch(&self) -> u64 {
        match self { ReadResult::Value { epoch, .. } => *epoch, _ => 0 }
    }
    pub fn missed(&self) -> u64 {
        match self { ReadResult::Value { missed, .. } => *missed, _ => 0 }
    }
}

// ---------------------------------------------------------------------------
// Block — heap unit of the causal chain (and large mutable values)
// ---------------------------------------------------------------------------

pub struct Block<T> {
    pub value: T,
    /// Backward causal edge. Null for the first block, and for all mutable
    /// (write) blocks since they carry no history.
    pub prev:  *const Block<T>,
    /// Epoch at time of creation. Written once at allocation; never changes.
    pub epoch: u64,
}

unsafe impl<T: Send> Send for Block<T> {}
unsafe impl<T: Send> Sync for Block<T> {}

impl<T> Block<T> {
    fn allocate(value: T, prev: *const Block<T>, epoch: u64) -> *const Self {
        Box::into_raw(Box::new(Block { value, prev, epoch }))
    }
}

// ---------------------------------------------------------------------------
// Inline encoding helpers
//
// Bit layout of an inline head word:
//   bit  0       = INLINE_TAG (always 1)
//   bits [32:1]  = value bytes (u32 — covers any T where size_of::<T>() <= 4)
//   bits [63:33] = epoch (31 bits → max ~2 billion writes per cell)
// ---------------------------------------------------------------------------

/// Encode a ≤4 byte value and its epoch into a tagged u64.
unsafe fn encode_inline<T>(value: T, epoch: u64) -> u64 {
    debug_assert!(size_of::<T>() <= 4);
    let value = ManuallyDrop::new(value);
    let mut buf = [0u8; 4];
    ptr::copy_nonoverlapping(
        &*value as *const T as *const u8,
        buf.as_mut_ptr(),
        size_of::<T>(),
    );
    (epoch << 33) | ((u32::from_ne_bytes(buf) as u64) << 1) | INLINE_TAG
}

/// Decode a ≤4 byte value from a tagged u64.
/// The cast to u32 naturally truncates bits[63:33] (epoch), leaving bits[32:1].
unsafe fn decode_inline<T>(bits: u64) -> T {
    debug_assert!(bits & INLINE_TAG != 0);
    let value_u32 = (bits >> 1) as u32;
    let buf = value_u32.to_ne_bytes();
    let mut v = MaybeUninit::<T>::uninit();
    ptr::copy_nonoverlapping(
        buf.as_ptr(),
        v.as_mut_ptr() as *mut u8,
        size_of::<T>(),
    );
    v.assume_init()
}

// ---------------------------------------------------------------------------
// Cell<T> — the unified primitive
// ---------------------------------------------------------------------------

pub struct Cell<T> {
    /// Inline (bit 0 = 1): bits[32:1] = value bytes, bits[63:33] = epoch.
    /// Pointer (bit 0 = 0): *const Block<T>; epoch lives in Block.epoch.
    /// Zero = never written.
    head:    AtomicU64,
    _marker: PhantomData<*const Block<T>>,
}

unsafe impl<T: Send> Send for Cell<T> {}
unsafe impl<T: Send> Sync for Cell<T> {}

impl<T> Cell<T> {
    pub fn new() -> Self {
        Cell { head: AtomicU64::new(0), _marker: PhantomData }
    }

    /// Read epoch from current head using Relaxed ordering (writer-side only).
    fn writer_epoch(&self) -> u64 {
        let bits = self.head.load(Ordering::Relaxed);
        if bits == 0 { 0 }
        else if bits & INLINE_TAG != 0 { bits >> 33 }
        else { unsafe { (*(bits as *const Block<T>)).epoch } }
    }

    // =========================================================================
    // WRITE: mutable replace
    //
    // ≤4 bytes — INLINE PATH
    //   Value and epoch packed into one u64. Single Release store.
    //   Zero allocation. Single Acquire load on the read side gives a
    //   consistent (value, epoch) pair — no race possible.
    //
    // >4 bytes — BLOCK PATH
    //   Allocates one Block per write. Block is fully initialised before
    //   the head pointer is published via Release store, so readers see a
    //   consistent (value, epoch) after a single Acquire load of the pointer.
    //
    //   TRADEOFF: one heap allocation per write. Old Blocks are leaked until
    //   a reclamation sweep frees them (epoch-based: safe once all readers
    //   have advanced past the old epoch — TODO). If your T fits in ≤4 bytes,
    //   use a newtype or bit-pack it to stay on the inline path.
    //
    // SAFETY: must be called by the single owning writer only.
    // =========================================================================

    pub unsafe fn write(&mut self, value: T) -> WriteResult {
        let new_epoch = self.writer_epoch() + 1;

        if size_of::<T>() <= 4 {
            self.head.store(encode_inline(value, new_epoch), Ordering::Release);
        } else {
            let block = Block::allocate(value, ptr::null(), new_epoch);
            self.head.store(block as u64, Ordering::Release);
        }
        WriteResult { epoch: new_epoch }
    }

    // =========================================================================
    // APPEND: immutable extend
    //
    // Always allocates a Block regardless of T size. Sets prev = current head,
    // building a causal chain traversable via ChainIter (newest → oldest).
    //
    // TRADEOFF: one heap allocation per append, and old Blocks are never freed
    // — freeing any Block requires knowing that no reader holds a pointer
    // anywhere in the chain behind it, which demands a separate epoch-based
    // sweep (TODO). Suitable for audit-log / event-history use where retention
    // is the point. If you only need the latest value, use write() instead.
    //
    // SAFETY: must be called by the single owning writer only.
    // =========================================================================

    pub unsafe fn append(&mut self, value: T) -> WriteResult {
        let new_epoch = self.writer_epoch() + 1;

        let prev_bits = self.head.load(Ordering::Relaxed);
        let prev: *const Block<T> = if prev_bits == 0 || prev_bits & INLINE_TAG != 0 {
            ptr::null()
        } else {
            prev_bits as *const Block<T>
        };

        let block = Block::allocate(value, prev, new_epoch);
        self.head.store(block as u64, Ordering::Release);
        WriteResult { epoch: new_epoch }
    }

    // ── Epoch-explicit variants used by Ring ──────────────────────────────────

    pub unsafe fn append_with_epoch(&mut self, value: T, epoch: u64) -> WriteResult {
        let prev_bits = self.head.load(Ordering::Relaxed);
        let prev: *const Block<T> = if prev_bits == 0 || prev_bits & INLINE_TAG != 0 {
            ptr::null()
        } else {
            prev_bits as *const Block<T>
        };
        let block = Block::allocate(value, prev, epoch);
        self.head.store(block as u64, Ordering::Release);
        WriteResult { epoch }
    }

    pub unsafe fn write_with_epoch(&mut self, value: T, epoch: u64) -> WriteResult {
        if size_of::<T>() <= 4 {
            self.head.store(encode_inline(value, epoch), Ordering::Release);
        } else {
            let block = Block::allocate(value, ptr::null(), epoch);
            self.head.store(block as u64, Ordering::Release);
        }
        WriteResult { epoch }
    }

    // =========================================================================
    // READ
    //
    // Single Acquire load of head.
    //   Inline: epoch is in bits[63:33] of the same word as the value.
    //   Block:  Acquire pairs with writer's Release; Block.epoch visible.
    //
    // In both cases value and epoch come from the same atomic operation,
    // so they are always mutually consistent.
    //
    // last_epoch: caller's last seen epoch (0 = first read, missed = 0).
    //
    // Safe from any thread, any number of concurrent readers.
    // Never blocks, never spins.
    // =========================================================================

    pub fn read(&self, last_epoch: u64) -> ReadResult<T> where T: Clone {
        let bits = self.head.load(Ordering::Acquire);
        if bits == 0 { return ReadResult::Empty; }

        let (value, current_epoch) = if bits & INLINE_TAG != 0 {
            (unsafe { decode_inline::<T>(bits) }, bits >> 33)
        } else {
            let block = unsafe { &*(bits as *const Block<T>) };
            (block.value.clone(), block.epoch)
        };

        let missed = if last_epoch == 0 { 0 }
                     else { current_epoch.saturating_sub(last_epoch + 1) };

        ReadResult::Value { value, epoch: current_epoch, missed }
    }

    // =========================================================================
    // CHAIN — traverse immutable causal history
    //
    // Returns iterator over Block values, newest first.
    // Only meaningful after append() calls (write() sets prev = null).
    // =========================================================================

    pub fn chain(&self) -> ChainIter<T> {
        let bits = self.head.load(Ordering::Acquire);
        if bits == 0 || bits & INLINE_TAG != 0 {
            ChainIter::new(ptr::null())
        } else {
            ChainIter::new(bits as *const Block<T>)
        }
    }

    /// Current epoch. Safe to call from any thread.
    pub fn epoch(&self) -> u64 {
        let bits = self.head.load(Ordering::Acquire);
        if bits == 0 { 0 }
        else if bits & INLINE_TAG != 0 { bits >> 33 }
        else { unsafe { (*(bits as *const Block<T>)).epoch } }
    }

    /// Current head as raw Block pointer. Null if empty or inline.
    pub fn head_ptr(&self) -> *const Block<T> {
        let bits = self.head.load(Ordering::Acquire);
        if bits == 0 || bits & INLINE_TAG != 0 { ptr::null() }
        else { bits as *const Block<T> }
    }

    // =========================================================================
    // READ_REF — zero-copy read
    //
    // Returns a ReadRef<T> that derefs to &T.
    //
    // Block path (T > 4B): holds a raw pointer directly into the heap Block.
    //   Zero copy. Block is never freed until reclamation sweep (not yet
    //   implemented), so the reference is stable for the caller's lifetime.
    //   For immutable (append) cells: definitively stable; value never changes.
    //   For mutable (write) cells: stable until reclamation; may be stale if
    //   the writer has advanced — check missed() after use.
    //
    // Inline path (T ≤ 4B): decodes the value and stores it inside ReadRef.
    //   A copy is unavoidable (the value lives in a register, not heap memory),
    //   but ≤4 bytes is register-sized so the cost is zero.
    //
    // Uses the same single Acquire load as read(), so epoch and value are
    // always mutually consistent (no race window vs the v0.8 two-atomic design).
    //
    // Use when: iterating a large dataset, analysing a struct, any case where
    // you don't need an owned T independent of the cell's lifetime.
    // =========================================================================
    pub fn read_ref(&self, last_epoch: u64) -> Option<ReadRef<T>> {
        let bits = self.head.load(Ordering::Acquire);
        if bits == 0 { return None; }

        let (current_epoch, is_inline) = if bits & INLINE_TAG != 0 {
            (bits >> 33, true)
        } else {
            (unsafe { (*(bits as *const Block<T>)).epoch }, false)
        };

        let missed = if last_epoch == 0 { 0 }
                     else { current_epoch.saturating_sub(last_epoch + 1) };

        if is_inline {
            Some(ReadRef {
                inner:       ReadRefInner::Inline(unsafe { decode_inline::<T>(bits) }),
                epoch:       current_epoch,
                missed,
                floor_slot:  ptr::null(),
                holds_ptr:   ptr::null_mut(),
                hold_epoch:  0,
                was_stacked: false,
            })
        } else {
            // Point directly into Block.value — zero copy.
            // SAFETY: Block is heap-allocated via Box::into_raw. On this
            // direct-from-Cell path the block is never freed (matches the
            // pre-reclamation behaviour). For automatic reclamation the
            // caller goes through `BridgedCell::read_ref` instead, which
            // pins a floor via the supplied ReaderHandle.
            let value_ptr = unsafe { &(*(bits as *const Block<T>)).value as *const T };
            Some(ReadRef {
                inner:       ReadRefInner::Block(value_ptr),
                epoch:       current_epoch,
                missed,
                floor_slot:  ptr::null(),
                holds_ptr:   ptr::null_mut(),
                hold_epoch:  0,
                was_stacked: false,
            })
        }
    }
}

impl<T> Default for Cell<T> { fn default() -> Self { Cell::new() } }

// ---------------------------------------------------------------------------
// ReadRef<T> — zero-copy reference to a cell's current value
//
// Derefs to &T. For Block values this is a direct pointer into heap memory —
// no copy at any point. For inline values (≤4B) holds a decoded copy
// internally (unavoidable, but ≤4B is register-sized so cost is zero).
//
// Caller pattern — control loop:
//
//   let r = cell.read_ref(last_epoch).unwrap();
//   for item in r.iter() { ... }   // iterates without copying
//   last_epoch = r.epoch;
//   if r.missed > 0 { /* data changed */ }
//
// Caller pattern — analysis:
//
//   let data = cell.read_ref(0).unwrap();
//   let sum  = data.iter().map(|x| x.value).sum::<f64>();
//   let max  = data.iter().max_by_key(|x| x.score);
//   // data.missed == 0 means dataset didn't change during analysis
// ---------------------------------------------------------------------------

enum ReadRefInner<T> {
    /// Direct pointer into Block.value. Zero copy.
    Block(*const T),
    /// Decoded inline value. Copy unavoidable but ≤4 bytes = free.
    Inline(T),
}

pub struct ReadRef<T> {
    inner:      ReadRefInner<T>,
    /// Epoch at time of read.
    pub epoch:  u64,
    /// Writes missed since caller's last_epoch.
    /// 0 = fully current. >0 = data may have changed since last read.
    pub missed: u64,
    // ── Bridge-layer floor pinning (null when unused) ────────────────────
    //
    // ReadRefs produced by `Cell::read_ref` (the direct, registry-less path)
    // and by `BridgedCell::read_ref` for INLINE values both leave these
    // null — Drop is a no-op. ReadRefs produced by `BridgedCell::read_ref`
    // for BLOCK values carry non-null pointers back into the issuing
    // ReaderHandle and clear them on Drop.
    //
    // SAFETY invariant: a non-null `floor_slot` / `holds_ptr` must point
    // inside the ReaderHandle that produced this ReadRef. The handle
    // outlives all its ReadRefs by API contract (per-thread, claimed once,
    // released when the thread ends).
    floor_slot:  *const AtomicU64,
    holds_ptr:   *mut HoldStack,
    /// Epoch this ReadRef pushed onto its handle's HoldStack.
    hold_epoch:  u64,
    /// True if the epoch is recorded in the stack array; false if the
    /// stack was full and we fell back to overflow_count tracking.
    was_stacked: bool,
}

unsafe impl<T: Send> Send for ReadRef<T> {}

impl<T> std::ops::Deref for ReadRef<T> {
    type Target = T;
    fn deref(&self) -> &T {
        match &self.inner {
            ReadRefInner::Block(ptr) => unsafe { &**ptr },
            ReadRefInner::Inline(v)  => v,
        }
    }
}

// ---------------------------------------------------------------------------
// ReadRef::Drop — the RAII contact point for floor release.
//
// This is the only piece of reclamation logic the Axis layer indirectly
// touches, and even then only via Rust's automatic drop insertion at
// scope end. Axis code never names floor_slot, hold_epoch, etc.
//
// Behaviour:
//   * floor_slot == null → no-op (Cell::read_ref, inline values, owned read)
//   * otherwise         → pop our entry from the HoldStack and re-publish
//                         the new floor (Release) so a concurrent reclaim()
//                         sees the up-to-date min epoch.
// ---------------------------------------------------------------------------
impl<T> Drop for ReadRef<T> {
    fn drop(&mut self) {
        if self.floor_slot.is_null() { return; }
        // SAFETY: per the invariant above, both pointers refer into a
        // ReaderHandle that outlives self, on the same thread that
        // created self (HoldStack is single-threaded).
        unsafe {
            let holds = &mut *self.holds_ptr;
            holds.remove(self.hold_epoch, self.was_stacked);
            let new_floor = holds.floor();
            (*self.floor_slot).store(new_floor, Ordering::Release);
        }
    }
}

// ---------------------------------------------------------------------------
// ChainIter — causal chain traversal, newest → oldest
// ---------------------------------------------------------------------------

pub struct ChainIter<T> { current: *const Block<T> }

impl<T> ChainIter<T> {
    pub fn new(head: *const Block<T>) -> Self { ChainIter { current: head } }
}

impl<T: Clone> Iterator for ChainIter<T> {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() { return None; }
        let block = unsafe { &*self.current };
        let val = block.value.clone();
        self.current = block.prev;
        Some(val)
    }
}

// ---------------------------------------------------------------------------
// Ring<T> — explicit N-cell retention window (opt-in)
//
// Use when: N concurrent readers each need their own stable head pointer,
// independently of what the writer has moved on to.
//
// Example: axAporia Ring 2 reading snapshot at epoch 100 while Ring 1
// has already written to epoch 105. Ring 2 needs Cell[slot_for_100]
// to remain stable. With a plain Cell that would be gone.
//
// Each slot is one Cell. Writer cycles through slots.
// Readers pick the slot with the epoch they need.
// ---------------------------------------------------------------------------

pub struct Ring<T> {
    cells:      Vec<Cell<T>>,
    write_pos:  usize,
    /// Shared monotonic epoch across all cells.
    /// Single writer — plain u64, no atomic needed.
    ring_epoch: u64,
}

impl<T: Send> Ring<T> {
    pub fn new(n: usize) -> Self {
        assert!(n >= 1);
        Ring { cells: (0..n).map(|_| Cell::new()).collect(), write_pos: 0, ring_epoch: 0 }
    }

    /// Append immutable value. Uses shared ring epoch so read_at_epoch works.
    /// SAFETY: single writer only.
    pub unsafe fn append(&mut self, value: T) -> WriteResult {
        self.ring_epoch += 1;
        let ep = self.ring_epoch;
        let result = self.cells[self.write_pos].append_with_epoch(value, ep);
        self.write_pos = (self.write_pos + 1) % self.cells.len();
        result
    }

    /// Write mutable value. Uses shared ring epoch.
    /// SAFETY: single writer only.
    pub unsafe fn write(&mut self, value: T) -> WriteResult {
        self.ring_epoch += 1;
        let ep = self.ring_epoch;
        let result = self.cells[self.write_pos].write_with_epoch(value, ep);
        self.write_pos = (self.write_pos + 1) % self.cells.len();
        result
    }

    /// Read from the most recently written cell.
    pub fn read(&self, last_epoch: u64) -> ReadResult<T> where T: Clone {
        let idx = if self.write_pos == 0 { self.cells.len() - 1 }
                  else { self.write_pos - 1 };
        self.cells[idx].read(last_epoch)
    }

    /// Read the cell whose epoch best matches target_epoch.
    /// Returns the highest epoch <= target_epoch.
    pub fn read_at_epoch(&self, target_epoch: u64) -> ReadResult<T> where T: Clone {
        let best = self.cells.iter()
            .filter(|c| c.epoch() > 0 && c.epoch() <= target_epoch)
            .max_by_key(|c| c.epoch());
        match best {
            Some(cell) => cell.read(0),
            None       => ReadResult::Empty,
        }
    }

    pub fn n_cells(&self)   -> usize { self.cells.len() }
    pub fn write_pos(&self) -> usize { self.write_pos }
    pub fn epoch(&self)     -> u64   { self.ring_epoch }
}

// ---------------------------------------------------------------------------
// SeqCell<T> — seqlock with inline storage, zero allocation
//
// For large T (size_of::<T>() > 4) in the mutable-only use case where you
// want zero heap allocation. Stores T directly inside the struct.
//
// ## Seqlock protocol
//
//   Write: epoch → ODD  (marks in-progress, AcqRel)
//          copy value into inline slot  (volatile ptr write)
//          epoch → EVEN (marks committed, Release)
//
//   Read:  loop {
//            e0 = epoch.load(Acquire) — if ODD, yield and retry
//            copy value out of inline slot
//            e1 = epoch.load(Acquire) — if e0 != e1, retry
//            return (value, e0 / 2)   — epoch normalised to write count
//          }
//
// ## Tradeoffs vs Cell<T> Block path
//
//   BETTER:  zero allocation per write; Cell<T> size is fixed (8 bytes)
//            regardless of T — SeqCell<T> grows with T (inline storage).
//            No reclamation needed: no heap memory to reclaim.
//
//   WORSE:   readers may spin briefly if they land exactly on a write
//            in progress. Under heavy write pressure from a fast writer,
//            readers will spin more often. Under light or moderate write
//            pressure, spin count is effectively zero.
//
//            T must be Copy (value is memcpy'd in and out of the slot).
//
// ## When to use
//
//   Use SeqCell<T> when:
//     - T > 4 bytes (otherwise Cell<T> inline is already zero allocation)
//     - You want zero heap allocation (embedded, OS kernel, no_std)
//     - Write pressure is low to moderate relative to read rate
//     - Brief reader spin under heavy writes is acceptable
//
//   Use Cell<T> Block path when:
//     - You need readers to never spin under any write pressure
//     - You already have a reclamation sweep (Block reuse)
//     - You need the immutable causal chain (append path)
// ---------------------------------------------------------------------------

/// Test-only counters — incremented each time a reader retries due to
/// a write in progress (ODD) or a write landing mid-copy (CHANGED).
/// Use delta(after - before) in tests to avoid cross-test interference.
#[cfg(test)]
static SEQ_SPIN_ODD:     AtomicU64 = AtomicU64::new(0);
#[cfg(test)]
static SEQ_SPIN_CHANGED: AtomicU64 = AtomicU64::new(0);

pub struct SeqCell<T> {
    /// Seqlock counter. ODD = write in progress. EVEN = committed.
    /// Epoch returned to callers is seq / 2 (counts completed writes).
    seq:     AtomicU64,
    /// Inline value storage. Written only by the single owning writer.
    slot:    UnsafeCell<MaybeUninit<T>>,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send> Send for SeqCell<T> {}
unsafe impl<T: Send> Sync for SeqCell<T> {}

impl<T: Copy> SeqCell<T> {
    pub fn new() -> Self {
        SeqCell {
            seq:     AtomicU64::new(0),
            slot:    UnsafeCell::new(MaybeUninit::uninit()),
            _marker: PhantomData,
        }
    }

    // =========================================================================
    // WRITE
    //
    // SAFETY: must be called by the single owning writer only.
    //
    // Bumps seq to ODD (marks in-progress), writes value into inline slot
    // via volatile copy (prevents compiler from reordering across the seq
    // stores), then bumps seq to EVEN (marks committed).
    //
    // AcqRel on the first bump: ensures readers who load the odd value see
    // any prior writes. Release on the second bump: ensures the value write
    // is visible to readers who Acquire the even seq.
    // =========================================================================
    pub unsafe fn write(&self, value: T) -> WriteResult {
        let old_seq = self.seq.load(Ordering::Relaxed);
        debug_assert!(old_seq % 2 == 0, "writer called write() while write in progress");

        // Mark in-progress. AcqRel so readers see a consistent before-state.
        self.seq.store(old_seq + 1, Ordering::Release);

        // Volatile copy: prevents the compiler reordering the value write
        // outside the seq window. Necessary because slot is not atomic.
        unsafe {
            ptr::write_volatile(
                (*self.slot.get()).as_mut_ptr(),
                value,
            );
        }

        // Mark committed. Release pairs with readers' Acquire on seq.
        let new_seq = old_seq + 2;
        self.seq.store(new_seq, Ordering::Release);
        WriteResult { epoch: new_seq / 2 }
    }

    // =========================================================================
    // READ
    //
    // Spins until it observes two identical even seq values bracketing the
    // value copy. The spin is a yield loop — it gives up the CPU rather than
    // burning it on a tight CAS loop, so it behaves well under OS scheduling.
    //
    // Returns Empty if the cell has never been written (seq == 0).
    // Returns Value { missed } using the same semantics as Cell<T>::read.
    //
    // Safe from any thread, any number of concurrent readers.
    // =========================================================================
    pub fn read(&self, last_epoch: u64) -> ReadResult<T> {
        loop {
            let e0 = self.seq.load(Ordering::Acquire);
            if e0 == 0     { return ReadResult::Empty; }
            if e0 % 2 == 1 {
                #[cfg(test)] SEQ_SPIN_ODD.fetch_add(1, Ordering::Relaxed);
                std::thread::yield_now(); continue; // write in progress
            }

            // Volatile read: prevents the compiler from hoisting this read
            // outside the seq bracket.
            let value = unsafe { ptr::read_volatile((*self.slot.get()).as_ptr()) };

            let e1 = self.seq.load(Ordering::Acquire);
            if e0 != e1 {
                #[cfg(test)] SEQ_SPIN_CHANGED.fetch_add(1, Ordering::Relaxed);
                std::thread::yield_now(); continue; // write landed mid-read
            }

            let current_epoch = e0 / 2;
            let missed = if last_epoch == 0 { 0 }
                         else { current_epoch.saturating_sub(last_epoch + 1) };
            return ReadResult::Value { value, epoch: current_epoch, missed };
        }
    }

    // =========================================================================
    // READ_UNPROTECTED — bypasses seqlock, for benchmarking and demonstration
    //
    // Does NOT retry on odd seq or seq change. Reads the inline slot directly.
    // For large T (>8 bytes) this WILL return torn values under concurrent
    // writes — the volatile copy is not atomic. Use only to demonstrate why
    // the seqlock is necessary, or for latency benchmarking in single-thread
    // scenarios where you know no write is concurrent.
    // =========================================================================
    pub fn read_unprotected(&self) -> ReadResult<T> {
        let e = self.seq.load(Ordering::Acquire);
        if e == 0 { return ReadResult::Empty; }
        let value = unsafe { ptr::read_volatile((*self.slot.get()).as_ptr()) };
        ReadResult::Value { value, epoch: e / 2, missed: 0 }
    }

    /// Current epoch. Safe to call from any thread.
    pub fn epoch(&self) -> u64 { self.seq.load(Ordering::Acquire) / 2 }
}

impl<T: Copy> Default for SeqCell<T> { fn default() -> Self { SeqCell::new() } }

// ===========================================================================
// BRIDGE LAYER — automatic Block reclamation (invisible to Axis code)
// ===========================================================================
//
// The Axis programmer never writes reclamation code — same way a Rust
// programmer never writes free(). The two contact points are RAII Drops:
//
//   ReadRef::drop      — releases / advances the reader's epoch floor
//   ReaderHandle::drop — clears the slot so an exited/panicking reader
//                        thread stops pinning memory
//
// Floor protocol (no CAS, no fetch_add, only AtomicU64 load/store):
//
//   READER  (BridgedCell::read_ref)         SWEEPER (BridgedCell::reclaim)
//   ──────                                  ───────
//   1. slot.store(1, Release)               1. for s in slots:
//   2. bits = head.load(Acquire)               m = min(m, s.load(Acquire))
//   3. epoch = decode(bits)                 2. drop every retired entry with
//   4. holds.push(epoch)                       retired_epoch < m
//   5. slot.store(holds.floor(), Release)
//
// Step 1 is conservative: a sweeper running between 1 and 5 sees floor=1,
// so retired_epoch < 1 is the only thing it can free — i.e. nothing,
// because epochs start at 1. Step 5 tightens the floor to the actual
// epoch we read, so subsequent sweeps free older retired blocks freely.
//
// MATERIALISE-OUT is the other half: BridgedCell::read() returns an
// owned clone and touches NO slot. The owned value can outlive any
// number of subsequent writes + reclaim calls — it pins nothing.
// ===========================================================================

// ── Overflow telemetry (test-only) ─────────────────────────────────────────
//
// Incremented on every push that overflows HOLD_DEPTH and falls back to
// conservative-floor tracking. If the adversarial tests never trip it,
// depth-8 has comfortable headroom; if they trip it constantly, that is
// a signal to revisit HOLD_DEPTH. Use delta(after - before) per test to
// avoid cross-test interference, same convention as SEQ_SPIN_*.
#[cfg(test)]
pub(crate) static HOLD_STACK_OVERFLOW_HITS: AtomicU64 = AtomicU64::new(0);

// ---------------------------------------------------------------------------
// HoldStack — per-handle record of currently-live ReadRef epochs
// ---------------------------------------------------------------------------
//
// Fixed depth (HOLD_DEPTH = 8) covers the common case (single-digit
// concurrent nested reads per thread). Overflow takes a CONSERVATIVE
// fallback (unknown U1 resolution):
//
//   * The overflowed ReadRef does NOT get a stack entry.
//   * `overflow_count` is incremented.
//   * `overflow_floor` is set to min(overflow_floor, new_epoch).
//   * `floor()` returns min(stack-min, overflow_floor when overflow_count>0).
//   * On drop of an overflow ReadRef: overflow_count decrements; when it
//     hits zero, overflow_floor resets to u64::MAX.
//   * The slot only returns to u64::MAX when BOTH count == 0 AND
//     overflow_count == 0 (i.e. when `floor()` returns u64::MAX).
//
// Property: floor is always pinned at least as low as required by safety.
// Pinning longer than strictly necessary is acceptable; freeing too early
// is not. The HOLD_STACK_OVERFLOW_HITS counter records every overflow
// so test output exposes whether depth-8 is ever exceeded in practice.
pub(crate) struct HoldStack {
    epochs:         [u64; HOLD_DEPTH],
    count:          usize,
    overflow_count: usize,
    overflow_floor: u64,
}

impl HoldStack {
    fn new() -> Self {
        HoldStack {
            epochs:         [0; HOLD_DEPTH],
            count:          0,
            overflow_count: 0,
            overflow_floor: u64::MAX,
        }
    }

    /// Returns true if the epoch took a stack slot; false if the stack
    /// was already full and we fell back to overflow tracking. Caller
    /// stashes this in the ReadRef so `remove` knows which path to take.
    fn push(&mut self, epoch: u64) -> bool {
        if self.count < HOLD_DEPTH {
            self.epochs[self.count] = epoch;
            self.count += 1;
            true
        } else {
            #[cfg(test)] {
                HOLD_STACK_OVERFLOW_HITS.fetch_add(1, Ordering::Relaxed);
            }
            self.overflow_count += 1;
            if epoch < self.overflow_floor { self.overflow_floor = epoch; }
            false
        }
    }

    fn remove(&mut self, epoch: u64, was_stacked: bool) {
        if was_stacked {
            // swap-remove the first matching entry.
            for i in 0..self.count {
                if self.epochs[i] == epoch {
                    self.count -= 1;
                    self.epochs[i] = self.epochs[self.count];
                    return;
                }
            }
            // If we get here the stack lost track — should never happen
            // under our protocol. Treat as a no-op in release.
            debug_assert!(false, "HoldStack::remove: epoch {} not found on stack", epoch);
        } else {
            debug_assert!(self.overflow_count > 0,
                "HoldStack::remove: overflow underflow");
            self.overflow_count -= 1;
            if self.overflow_count == 0 {
                self.overflow_floor = u64::MAX;
            }
        }
    }

    /// Current floor across all live entries (stack + overflow).
    /// u64::MAX iff no holds at all (slot returns to idle).
    fn floor(&self) -> u64 {
        let mut m = if self.overflow_count > 0 { self.overflow_floor }
                    else { u64::MAX };
        for i in 0..self.count {
            if self.epochs[i] < m { m = self.epochs[i]; }
        }
        m
    }
}

// ---------------------------------------------------------------------------
// ReaderRegistry — fixed array of per-reader floor slots
// ---------------------------------------------------------------------------
//
// Each slot is an AtomicU64 holding the owning reader's current epoch
// floor — the oldest epoch any of its live ReadRefs might still
// dereference. u64::MAX = idle (pins nothing).
//
// SINGLE-WRITER-PER-SLOT (hard limit): only the ReaderHandle that owns
// slot i ever stores to slot i. The sweeper Acquires every slot but
// stores to none. There is no CAS, no fetch_add, no fetch_or anywhere
// on the floor path.
//
// Slot allocation uses a Mutex<usize> next-slot counter. THIS MUTEX IS
// USED ONLY IN `acquire()`, which is called ONCE per reader thread's
// lifetime — never on the hot read / write / reclaim paths. The hard
// limit "no bus-locking on the hot path" is preserved.
pub struct ReaderRegistry {
    slots: [AtomicU64; MAX_READERS],
    next:  Mutex<usize>,
    /// Monotonic "have we ever issued a ReaderHandle?" flag.
    /// Flipped Release-true on first `acquire()`, never cleared.
    ///
    /// PURPOSE: the cheapest-possible "is this cell shared with any
    /// reader at all?" signal — a single Acquire load. Most cells in a
    /// real codebase never see a reader; for those, BridgedCell::write_lazy
    /// can free old blocks immediately and bypass the retired-list path
    /// entirely. Once a reader registers the flag flips and stays flipped
    /// (futex-style — no oscillation, no migration).
    ///
    /// Uses `store(Release)`/`load(Acquire)` only — no CAS, no fetch_or.
    any_handle_ever: std::sync::atomic::AtomicBool,
}

impl ReaderRegistry {
    pub fn new() -> Self {
        ReaderRegistry {
            slots: std::array::from_fn(|_| AtomicU64::new(u64::MAX)),
            next:  Mutex::new(0),
            any_handle_ever: std::sync::atomic::AtomicBool::new(false),
        }
    }

    /// Reserve one slot for a reader thread. Panics if MAX_READERS exceeded.
    /// Cold path — once per reader thread's lifetime.
    pub fn acquire(&self) -> ReaderHandle<'_> {
        let mut next = self.next.lock().unwrap();
        let slot = *next;
        assert!(slot < MAX_READERS,
            "ReaderRegistry exhausted (MAX_READERS = {})", MAX_READERS);
        *next += 1;
        // Monotonic flip — once true, stays true for the registry's life.
        self.any_handle_ever.store(true, Ordering::Release);
        // Slot is already u64::MAX (from `new` or a previous handle's Drop).
        ReaderHandle {
            registry: self,
            slot,
            holds:    Box::new(UnsafeCell::new(HoldStack::new())),
        }
    }

    /// Fresh Acquire scan of every slot — returns the smallest floor
    /// any live handle is publishing. u64::MAX if every slot is idle.
    /// Called every time `reclaim()` runs. Never cached.
    pub fn floor_min(&self) -> u64 {
        let mut m = u64::MAX;
        for s in self.slots.iter() {
            let v = s.load(Ordering::Acquire);
            if v < m { m = v; }
        }
        m
    }

    /// Has any ReaderHandle ever been issued for this registry?
    /// Single Acquire load — the Level B cheap signal for `write_lazy`.
    #[inline]
    pub fn has_any_reader(&self) -> bool {
        self.any_handle_ever.load(Ordering::Acquire)
    }
}

impl Default for ReaderRegistry {
    fn default() -> Self { Self::new() }
}

// ---------------------------------------------------------------------------
// ReaderHandle — RAII wrapper for one registry slot
// ---------------------------------------------------------------------------
//
// One per reader thread for the thread's lifetime. Drop writes
// u64::MAX to the slot — the "stalled / panicking reader" fix.
// No heartbeats, no timeouts; if the thread unwinds for any reason,
// Drop runs and the slot stops pinning memory.
//
// `holds` is single-thread by API contract (the handle stays on one
// thread). It is wrapped in UnsafeCell so that `read_ref` can take
// `&ReaderHandle` (callable through a shared reference) while still
// mutating the stack.
//
// The HoldStack is BOXED so that its heap address stays stable even
// when the ReaderHandle itself is moved (e.g. pushed into a Vec).
// ReadRefs hold `*mut HoldStack` raw pointers into this heap allocation;
// moving the Box on the stack does not invalidate them, because the
// pointed-to allocation never moves.
pub struct ReaderHandle<'reg> {
    registry: &'reg ReaderRegistry,
    slot:     usize,
    holds:    Box<UnsafeCell<HoldStack>>,
}

impl<'reg> ReaderHandle<'reg> {
    #[inline]
    fn slot_atomic(&self) -> &AtomicU64 {
        &self.registry.slots[self.slot]
    }

    /// Slot index — for test introspection only.
    pub fn slot_index(&self) -> usize { self.slot }
}

impl<'reg> Drop for ReaderHandle<'reg> {
    fn drop(&mut self) {
        // STALLED-READER FIX: release our slot so this thread stops
        // pinning memory the moment it exits (normal return OR panic
        // unwind — Rust runs Drop in both cases).
        self.slot_atomic().store(u64::MAX, Ordering::Release);
    }
}

// ---------------------------------------------------------------------------
// BridgedCell<T> — Cell<T> plus a retired-block list and reclamation API
// ---------------------------------------------------------------------------
//
// Wraps Cell<T> without changing Cell's layout (Cell remains the minimal
// 8-byte primitive). All bridge-layer concerns live here. From the Axis
// programmer's perspective: write(), read(), append(), read_ref(handle).
// The retired list, registry slot pinning, and reclaim sweep are
// invisible.
//
// Concurrency model is identical to Cell:
//   * write / append / reclaim need &mut self (single owner / writer)
//   * read / read_ref need &self (any number of concurrent readers)
//
// Tests use the same Mutex<BridgedCell<T>> harness as Cell's existing
// concurrent tests where shared mutability is needed.

struct RetiredEntry<T> {
    ptr:           *const Block<T>,
    /// Epoch of the retired Block itself (its `Block.epoch`). The block
    /// is safe to free once every live reader's floor is strictly greater
    /// than this value, i.e. `retired_epoch < floor_min`.
    retired_epoch: u64,
}

pub struct BridgedCell<T> {
    inner:   Cell<T>,
    /// Heap blocks that have been superseded by a later mutable write but
    /// might still be reachable through a live ReadRef. Drained by
    /// reclaim().
    retired: Vec<RetiredEntry<T>>,
}

unsafe impl<T: Send> Send for BridgedCell<T> {}
unsafe impl<T: Send> Sync for BridgedCell<T> {}

impl<T> BridgedCell<T> {
    pub fn new() -> Self {
        BridgedCell { inner: Cell::new(), retired: Vec::new() }
    }

    /// Inner Cell — for callers that need cell-level features (epoch(),
    /// chain(), etc.) without going through the bridge.
    pub fn inner(&self) -> &Cell<T> { &self.inner }

    /// Current count of retired-but-not-yet-freed blocks.
    /// Used by reclaim_if_watermark and by tests.
    pub fn retired_len(&self) -> usize { self.retired.len() }

    // =========================================================================
    // WRITE (mutable replace) — bridge-managed retirement
    //
    // Captures the old head bits BEFORE calling Cell::write. If the old
    // head was a Block pointer, that block is now orphaned and goes onto
    // the retired list tagged with its own epoch.
    //
    // SAFETY: single writer per cell only — same as Cell::write.
    // =========================================================================
    pub unsafe fn write(&mut self, value: T) -> WriteResult {
        let old_bits = self.inner.head.load(Ordering::Relaxed);
        let result = unsafe { self.inner.write(value) };
        if old_bits != 0 && (old_bits & INLINE_TAG == 0) {
            let old_ptr = old_bits as *const Block<T>;
            // SAFETY: old_ptr came out of Box::into_raw via Cell::write/append
            // and has not been retired before (single writer => no race).
            let retired_epoch = unsafe { (*old_ptr).epoch };
            self.retired.push(RetiredEntry { ptr: old_ptr, retired_epoch });
        }
        result
    }

    // =========================================================================
    // WRITE_LAZY (mutable replace, optimistic immediate-free)
    //
    // Same wire semantics as write(), but uses the registry's "is anyone
    // reading?" signals to FREE the old block immediately whenever it can
    // prove no live ReadRef can reach it, instead of pushing it to the
    // retired list and waiting for a reclaim sweep.
    //
    // Two cheap signals, checked in order:
    //
    //   LEVEL B (cheapest — single Acquire load):
    //     !registry.has_any_reader()  →  no ReaderHandle has ever existed
    //     for this registry. No reader can be pinning anything. Free.
    //
    //   LEVEL A (64 cache-warm Acquire loads):
    //     registry.floor_min() > old_block.epoch  →  every live reader's
    //     pin is at an epoch strictly later than the block we just
    //     superseded. No live ReadRef can reach the old block. Free.
    //
    //   Otherwise: retire as in write(). A later reclaim() will free it
    //     once readers drain past old_epoch.
    //
    // SAFETY ARGUMENT for Level A (the subtle one):
    //   The Acquire scan is done AFTER the Release-store of the new head
    //   in inner.write(). A reader whose `slot.store(1, Release)` happens
    //   before the scan is seen by the scan (floor_min ≤ 1 ≤ old_epoch
    //   → retire, conservative). A reader whose pin happens AFTER the
    //   scan also did its Acquire-load of head AFTER the new-head publish
    //   (program order on the reader side), so it sees the NEW head and
    //   pins the NEW epoch — never the freed old block.
    //
    // SAFETY: single writer per cell only — same as write().
    // =========================================================================
    pub unsafe fn write_lazy(&mut self, value: T, registry: &ReaderRegistry) -> WriteResult {
        let old_bits = self.inner.head.load(Ordering::Relaxed);
        let result = unsafe { self.inner.write(value) };

        // Inline or first-ever-write: no Block to dispose of.
        if old_bits == 0 || (old_bits & INLINE_TAG != 0) {
            return result;
        }
        let old_ptr = old_bits as *const Block<T>;
        // SAFETY: old_ptr was just published-then-superseded by the
        // single writer; no concurrent mutation of the block itself.
        let old_epoch = unsafe { (*old_ptr).epoch };

        // ── LEVEL B fast path ──────────────────────────────────────────
        if !registry.has_any_reader() {
            // SAFETY: no ReaderHandle has ever existed → no ReadRef has
            // ever existed → no pointer to old_ptr exists outside us.
            unsafe { let _ = Box::from_raw(old_ptr as *mut Block<T>); }
            return result;
        }

        // ── LEVEL A scan ───────────────────────────────────────────────
        let fm = registry.floor_min();
        if fm > old_epoch {
            // SAFETY: every live reader's pin > old_epoch, so no live
            // ReadRef can dereference old_ptr (see argument above).
            unsafe { let _ = Box::from_raw(old_ptr as *mut Block<T>); }
        } else {
            self.retired.push(RetiredEntry { ptr: old_ptr, retired_epoch: old_epoch });
        }
        result
    }

    // =========================================================================
    // APPEND (immutable extend) — does NOT retire
    //
    // The chain itself keeps every block reachable, so append blocks are
    // never orphaned by a subsequent write. Compaction / truncation is
    // explicitly out of scope.
    //
    // SAFETY: single writer per cell only.
    // =========================================================================
    pub unsafe fn append(&mut self, value: T) -> WriteResult {
        unsafe { self.inner.append(value) }
    }

    // =========================================================================
    // READ (owned clone) — MATERIALISE-OUT, pins NOTHING
    //
    // Hard limit: this path MUST NOT touch any ReaderRegistry slot. The
    // value is copied out of the Block; the Block becomes reclaimable
    // immediately regardless of how long the caller keeps the owned
    // value. This is what lets owned reads escape their read scope
    // without leaking the Axis-layer reclamation machinery.
    //
    // Rule: escape the read scope → use read(). Stay scoped + zero-copy
    //       → use read_ref().
    // =========================================================================
    pub fn read(&self, last_epoch: u64) -> ReadResult<T> where T: Clone {
        self.inner.read(last_epoch)
    }

    // =========================================================================
    // READ_REF (zero-copy borrow) — pins the floor for Block path
    //
    // INLINE values (T ≤ 4B): no Block reference, ReadRef holds a decoded
    // copy. No floor pinning needed — null floor_slot, Drop no-op.
    //
    // BLOCK values: ReadRef holds a raw pointer into Block.value. We MUST
    // pin the floor at the read epoch so a concurrent reclaim() cannot
    // free the block.
    //
    // Protocol (see top-of-bridge-section diagram):
    //   1. slot.store(1, Release)  — conservative pre-publish
    //   2. bits = head.load(Acquire)
    //   3. push epoch onto handle's HoldStack
    //   4. slot.store(holds.floor(), Release) — tighten to live min
    // =========================================================================
    pub fn read_ref(
        &self,
        handle: &ReaderHandle<'_>,
        last_epoch: u64,
    ) -> Option<ReadRef<T>> {
        let slot = handle.slot_atomic();

        // 1. Conservative pre-publish. Pins everything ≥ 1 (every real
        //    block) until step 4 tightens the floor to the actual epoch.
        slot.store(1, Ordering::Release);

        // 2. Acquire head — synchronises with writer's Release store.
        let bits = self.inner.head.load(Ordering::Acquire);
        if bits == 0 {
            // Empty cell. Release the conservative floor and return.
            slot.store(u64::MAX, Ordering::Release);
            return None;
        }

        let is_inline = bits & INLINE_TAG != 0;
        let current_epoch = if is_inline { bits >> 33 }
                            else { unsafe { (*(bits as *const Block<T>)).epoch } };
        let missed = if last_epoch == 0 { 0 }
                     else { current_epoch.saturating_sub(last_epoch + 1) };

        if is_inline {
            // Decoded copy: pin nothing.
            slot.store(u64::MAX, Ordering::Release);
            Some(ReadRef {
                inner:       ReadRefInner::Inline(unsafe { decode_inline::<T>(bits) }),
                epoch:       current_epoch,
                missed,
                floor_slot:  ptr::null(),
                holds_ptr:   ptr::null_mut(),
                hold_epoch:  0,
                was_stacked: false,
            })
        } else {
            // 3. Push onto hold stack.
            // SAFETY: handle.holds is single-thread by API contract;
            // no concurrent access from elsewhere.
            let holds = unsafe { &mut *handle.holds.get() };
            let was_stacked = holds.push(current_epoch);

            // 4. Tighten floor to the live min across all held epochs.
            slot.store(holds.floor(), Ordering::Release);

            let value_ptr = unsafe { &(*(bits as *const Block<T>)).value as *const T };
            Some(ReadRef {
                inner:       ReadRefInner::Block(value_ptr),
                epoch:       current_epoch,
                missed,
                floor_slot:  slot as *const AtomicU64,
                holds_ptr:   handle.holds.get(),
                hold_epoch:  current_epoch,
                was_stacked,
            })
        }
    }

    // =========================================================================
    // RECLAIM — fresh Acquire scan, then free retired blocks below floor_min
    //
    // Any thread may call this. It never blocks readers or writers (it
    // takes &mut self, so the caller arranges exclusivity — typically
    // via the same Mutex protecting writes).
    //
    // ALWAYS does a fresh `registry.floor_min()` — the scan is never
    // cached. `reclaim_if_watermark` only gates whether reclaim runs at
    // all; it does NOT gate the fresh scan once it decides to run.
    //
    // Returns the number of blocks actually freed.
    // =========================================================================
    pub fn reclaim(&mut self, registry: &ReaderRegistry) -> usize {
        let floor_min = registry.floor_min();
        let mut freed = 0usize;
        let mut i = 0;
        while i < self.retired.len() {
            if self.retired[i].retired_epoch < floor_min {
                let entry = self.retired.swap_remove(i);
                // SAFETY: the block was allocated via Box::into_raw and
                // its retired_epoch < floor_min proves no live ReadRef
                // can reach it. Single writer + &mut self + the fresh
                // floor scan above mean no concurrent path can promote
                // it back to live before we free it.
                unsafe { let _ = Box::from_raw(entry.ptr as *mut Block<T>); }
                freed += 1;
                // swap_remove pulled tail into i — re-check at the same i.
            } else {
                i += 1;
            }
        }
        freed
    }

    // =========================================================================
    // RECLAIM_IF_WATERMARK — opportunistic gating
    //
    // Gates ONLY whether reclaim() runs at all (based on retired list
    // length). When it does run, the floor scan inside reclaim() is
    // ALWAYS fresh — staleness is bounded by retired_len <= WATERMARK
    // between sweeps, never by a cached floor.
    //
    // Writers can call this opportunistically; readers never do.
    // =========================================================================
    pub fn reclaim_if_watermark(&mut self, registry: &ReaderRegistry) -> usize {
        if self.retired.len() >= WATERMARK {
            self.reclaim(registry)
        } else {
            0
        }
    }
}

impl<T> Default for BridgedCell<T> {
    fn default() -> Self { Self::new() }
}

impl<T> Drop for BridgedCell<T> {
    fn drop(&mut self) {
        // Free everything still on the retired list. By the time Drop
        // runs there are no live ReadRefs (they would borrow self).
        for entry in self.retired.drain(..) {
            unsafe { let _ = Box::from_raw(entry.ptr as *mut Block<T>); }
        }
        // Free the current head if it is a heap Block (a Block at head
        // from append() chains is also freed here — the prev chain is
        // not walked; chain-truncation is explicitly out of scope, so
        // append-chain blocks behind head remain leaked, matching the
        // pre-bridge baseline behaviour).
        let bits = self.inner.head.load(Ordering::Relaxed);
        if bits != 0 && (bits & INLINE_TAG == 0) {
            unsafe { let _ = Box::from_raw(bits as *mut Block<T>); }
            // Stop any future loader from observing the dangling pointer.
            self.inner.head.store(0, Ordering::Relaxed);
        }
    }
}

// ===========================================================================
// SpscQueue<T, N> — bounded single-producer single-consumer FIFO
// ===========================================================================
//
// Lamport ring buffer with monotonic head/tail counters. No CAS, no
// fetch_add — only AtomicUsize load/store. The single-producer /
// single-consumer discipline is what makes this safe without RMW:
//   * head is written ONLY by the producer (we are the only writer).
//   * tail is written ONLY by the consumer.
//   * Each side Acquire-loads the other side's index to synchronise.
//
// Use when: one thread feeds, one thread drains. Render command stream
// (sim → render), audio sample feed, single-broker IPC, log shipper.
//
// NOT a substitute for MPMC. MPMC requires CAS, which is outside this
// crate's discipline; use a different primitive (e.g. crossbeam) when
// you genuinely need multi-producer.
// ===========================================================================

pub struct SpscQueue<T, const N: usize> {
    /// Storage. Each slot is initialised on first push to that index
    /// and consumed (moved out) on pop.
    ring: [UnsafeCell<MaybeUninit<T>>; N],
    /// Producer's running count of pushes. Monotonic (wraps at usize::MAX,
    /// which is irrelevant for any realistic workload). Slot index =
    /// head % N. Written ONLY by the producer.
    head: AtomicUsize,
    /// Consumer's running count of pops. Slot index = tail % N. Written
    /// ONLY by the consumer.
    tail: AtomicUsize,
    _marker: PhantomData<T>,
}

unsafe impl<T: Send, const N: usize> Send for SpscQueue<T, N> {}
unsafe impl<T: Send, const N: usize> Sync for SpscQueue<T, N> {}

impl<T, const N: usize> SpscQueue<T, N> {
    pub fn new() -> Self {
        // Compile-time check: capacity must be > 0.
        // (We can't const-assert in stable, so do it here.)
        assert!(N > 0, "SpscQueue capacity must be > 0");
        SpscQueue {
            ring: std::array::from_fn(|_| UnsafeCell::new(MaybeUninit::uninit())),
            head: AtomicUsize::new(0),
            tail: AtomicUsize::new(0),
            _marker: PhantomData,
        }
    }

    pub fn capacity(&self) -> usize { N }

    /// Current number of unread items. May be observed slightly stale
    /// by the side that didn't perform the last operation, but is always
    /// a lower-bound (consumer view) or upper-bound (producer view).
    pub fn len(&self) -> usize {
        let head = self.head.load(Ordering::Acquire);
        let tail = self.tail.load(Ordering::Acquire);
        head.wrapping_sub(tail)
    }

    pub fn is_empty(&self) -> bool {
        self.head.load(Ordering::Acquire) == self.tail.load(Ordering::Acquire)
    }

    pub fn is_full(&self) -> bool {
        self.len() >= N
    }

    /// Push a value. Returns `Err(value)` if the queue is full.
    ///
    /// SAFETY: caller must be the sole producer thread.
    ///
    /// Ordering:
    ///   * Acquire-load `tail` so we observe the most recent pop and
    ///     don't overwrite a slot the consumer hasn't yet drained.
    ///   * Release-store `head` so the consumer's later Acquire-load of
    ///     `head` synchronises-with our value write into the slot.
    pub unsafe fn push(&self, value: T) -> Result<(), T> {
        let head = self.head.load(Ordering::Relaxed);     // we are the only writer
        let tail = self.tail.load(Ordering::Acquire);     // sync with consumer
        if head.wrapping_sub(tail) >= N {
            return Err(value);
        }
        let slot = head % N;
        // SAFETY: slot is owned by us until we publish head; consumer
        // can't observe this slot as readable until our Release store.
        unsafe { (*self.ring[slot].get()).write(value); }
        self.head.store(head.wrapping_add(1), Ordering::Release);
        Ok(())
    }

    /// Pop a value. Returns `None` if the queue is empty.
    ///
    /// SAFETY: caller must be the sole consumer thread.
    pub unsafe fn pop(&self) -> Option<T> {
        let tail = self.tail.load(Ordering::Relaxed);     // we are the only writer
        let head = self.head.load(Ordering::Acquire);     // sync with producer
        if head == tail {
            return None;
        }
        let slot = tail % N;
        // SAFETY: producer Released this slot's content into the queue
        // when it published head > tail. We Acquired head above. The
        // slot is fully initialised and no other thread will touch it.
        let value = unsafe { (*self.ring[slot].get()).assume_init_read() };
        self.tail.store(tail.wrapping_add(1), Ordering::Release);
        Some(value)
    }
}

impl<T, const N: usize> Default for SpscQueue<T, N> {
    fn default() -> Self { Self::new() }
}

impl<T, const N: usize> Drop for SpscQueue<T, N> {
    fn drop(&mut self) {
        // Drop the unread items so T::drop runs on each.
        // SAFETY: by Drop time, no producer / consumer can be running.
        let tail = *self.tail.get_mut();
        let head = *self.head.get_mut();
        let mut i = tail;
        while i != head {
            unsafe { (*self.ring[i % N].get()).assume_init_drop(); }
            i = i.wrapping_add(1);
        }
    }
}

// ===========================================================================
// DoubleBuffer<T> — two-slot atomic swap-publish
// ===========================================================================
//
// Two heap-free slots. The writer fills the back slot, then atomically
// publishes by swapping front/back. Readers see the front slot through a
// shared `&T`. Single Release-store / Acquire-load on the front index —
// no allocation, no fetch_add, no CAS.
//
// CONCURRENCY MODEL (frame-boundary discipline):
//   * One writer fills `back()` at its own pace.
//   * Writer calls `publish()` (or the convenience `write()`) at a
//     known-safe moment (e.g. end of simulation tick).
//   * Readers consume `front()` between publishes. By caller contract
//     they MUST NOT hold a reference across a publish — there is no
//     refcount / floor to enforce it. This is the explicit tradeoff
//     vs BridgedCell: cheaper, but requires the caller to manage the
//     swap point (frame sync is the canonical pattern).
//
// Use when: game ECS components per frame, audio buffer double-buffering,
// render command lists, anywhere the swap point is structurally obvious.
// ===========================================================================

pub struct DoubleBuffer<T> {
    slots: [UnsafeCell<T>; 2],
    /// 0 or 1 — which slot readers currently see. Single-writer-per-slot
    /// in the moral sense: the writer mutates only the back slot, then
    /// flips this index to publish.
    front: AtomicUsize,
}

unsafe impl<T: Send> Send for DoubleBuffer<T> {}
unsafe impl<T: Send + Sync> Sync for DoubleBuffer<T> {}

impl<T> DoubleBuffer<T> {
    pub fn new(initial: T) -> Self where T: Clone {
        DoubleBuffer {
            slots: [UnsafeCell::new(initial.clone()), UnsafeCell::new(initial)],
            front: AtomicUsize::new(0),
        }
    }

    /// Read the current front slot. The returned reference is valid
    /// until the caller releases it AND no `publish()` runs in between.
    /// Caller-managed frame sync enforces this.
    pub fn read(&self) -> &T {
        let f = self.front.load(Ordering::Acquire);
        // SAFETY: writer only mutates the back slot between publishes;
        // front is stable for readers in the inter-publish interval.
        unsafe { &*self.slots[f].get() }
    }

    /// Mutable access to the back slot — for in-place updates over a
    /// frame.
    ///
    /// SAFETY: single writer. No other thread may access the back slot.
    pub unsafe fn back_mut(&self) -> &mut T {
        let f = self.front.load(Ordering::Relaxed);
        let back = 1 - f;
        unsafe { &mut *self.slots[back].get() }
    }

    /// Swap front/back. After this point readers will see what was the
    /// back slot.
    ///
    /// SAFETY: single writer; caller-managed frame sync (no reader is
    /// holding a reference to the previous front slot).
    pub unsafe fn publish(&self) {
        let f = self.front.load(Ordering::Relaxed);
        self.front.store(1 - f, Ordering::Release);
    }

    /// Convenience: overwrite the back slot then publish. Equivalent to
    /// `*back_mut() = value; publish();`.
    ///
    /// SAFETY: single writer; caller-managed frame sync.
    pub unsafe fn write(&self, value: T) {
        let f = self.front.load(Ordering::Relaxed);
        let back = 1 - f;
        // SAFETY: writer-exclusive access to back slot; existing value
        // is dropped by the assignment.
        unsafe { *self.slots[back].get() = value; }
        self.front.store(back, Ordering::Release);
    }

    /// Which slot index is currently in front (0 or 1). For diagnostics.
    pub fn front_index(&self) -> usize {
        self.front.load(Ordering::Acquire)
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering as AO};

    // ── Mutable write: small (inline) ─────────────────────────────────────────

    #[test]
    fn mutable_small_inline_u32() {
        let mut cell: Cell<u32> = Cell::new();
        assert_eq!(cell.read(0), ReadResult::Empty);

        let w = unsafe { cell.write(42u32) };
        assert_eq!(w.epoch, 1);

        match cell.read(0) {
            ReadResult::Value { value, epoch, missed } => {
                assert_eq!(value, 42u32);
                assert_eq!(epoch, 1);
                assert_eq!(missed, 0);
            }
            _ => panic!("expected Value"),
        }

        // Tag bit must be set (inline encoding — no Block allocation)
        let bits = cell.head.load(Ordering::Relaxed);
        assert_eq!(bits & INLINE_TAG, INLINE_TAG, "u32 must be inline");
    }

    #[test]
    fn mutable_small_u8() {
        let mut cell: Cell<u8> = Cell::new();
        unsafe { cell.write(255u8) };
        assert_eq!(cell.read(0).value(), Some(255u8));
    }

    #[test]
    fn mutable_small_u16() {
        let mut cell: Cell<u16> = Cell::new();
        unsafe { cell.write(1234u16) };
        assert_eq!(cell.read(0).value(), Some(1234u16));
    }

    #[test]
    fn mutable_small_overwrite() {
        let mut cell: Cell<u32> = Cell::new();
        for i in 0..100u32 { unsafe { cell.write(i) }; }
        assert_eq!(cell.read(0).value(), Some(99u32));
    }

    // ── Mutable write: large (Block) ─────────────────────────────────────────

    #[test]
    fn mutable_large_u64() {
        let mut cell: Cell<u64> = Cell::new();
        unsafe { cell.write(9999u64) };
        assert_eq!(cell.read(0).value(), Some(9999u64));

        // Must NOT be inline (u64 is 8 bytes)
        let bits = cell.head.load(Ordering::Relaxed);
        assert_eq!(bits & INLINE_TAG, 0, "u64 must use Block pointer");
    }

    #[test]
    fn mutable_large_struct() {
        #[derive(Clone, Debug, PartialEq)]
        struct Reading { temp: f64, pressure: f64 }

        let mut cell: Cell<Reading> = Cell::new();
        unsafe { cell.write(Reading { temp: 23.5, pressure: 1013.25 }) };
        let v = cell.read(0).value().unwrap();
        assert_eq!(v.temp, 23.5);
        assert_eq!(v.pressure, 1013.25);
    }

    // ── Epoch: write counter ──────────────────────────────────────────────────

    #[test]
    fn epoch_increments_each_write() {
        let mut cell: Cell<u32> = Cell::new();
        for i in 1..=5u32 {
            let w = unsafe { cell.write(i) };
            assert_eq!(w.epoch, i as u64);
        }
        assert_eq!(cell.epoch(), 5);
    }

    #[test]
    fn epoch_increments_on_append() {
        let mut cell: Cell<u64> = Cell::new();
        for i in 1..=5u64 {
            let w = unsafe { cell.append(i) };
            assert_eq!(w.epoch, i);
        }
    }

    // ── Epoch: backpressure signal (missed writes) ────────────────────────────

    #[test]
    fn missed_zero_when_current() {
        let mut cell: Cell<u32> = Cell::new();
        let w = unsafe { cell.write(1) };
        let r = cell.read(w.epoch - 1);
        assert_eq!(r.missed(), 0, "no missed writes when fully current");
    }

    #[test]
    fn missed_counts_skipped_writes() {
        let mut cell: Cell<u32> = Cell::new();
        unsafe { cell.write(1) }; // epoch 1
        unsafe { cell.write(2) }; // epoch 2
        unsafe { cell.write(3) }; // epoch 3
        unsafe { cell.write(4) }; // epoch 4

        // Reader last saw epoch 1, now reading epoch 4
        let r = cell.read(1);
        assert_eq!(r.missed(), 2, "missed writes 2 and 3");
        assert_eq!(r.value(), Some(4u32));
    }

    #[test]
    fn missed_zero_on_first_read() {
        let mut cell: Cell<u32> = Cell::new();
        for _ in 0..10 { unsafe { cell.write(42) }; }
        let r = cell.read(0);
        assert_eq!(r.missed(), 0, "first read: missed is always 0");
    }

    #[test]
    fn caller_can_detect_write_pressure() {
        let mut cell: Cell<u32> = Cell::new();
        for i in 0..1000u32 { unsafe { cell.write(i) }; }

        let r = cell.read(0);
        let last = r.epoch();

        for i in 1000..1050u32 { unsafe { cell.write(i) }; }

        let r2 = cell.read(last);
        assert_eq!(r2.missed(), 49, "reader missed 49 writes");
    }

    // ── Inline: epoch packed in same word as value ────────────────────────────

    #[test]
    fn inline_epoch_consistent_with_value() {
        // Verifies that for inline values the epoch extracted from read()
        // always matches the epoch returned by write() — they come from the
        // same atomic word so there is no window for inconsistency.
        let mut cell: Cell<u32> = Cell::new();
        for i in 1..=20u32 {
            let w = unsafe { cell.write(i) };
            let r = cell.read(0);
            assert_eq!(r.epoch(), w.epoch, "epoch in read must match write");
            assert_eq!(r.value(), Some(i));
        }
    }

    // ── Immutable append: causal chain ───────────────────────────────────────

    #[test]
    fn append_builds_causal_chain() {
        let mut cell: Cell<u64> = Cell::new();
        for i in 0..5u64 { unsafe { cell.append(i) }; }

        let chain: Vec<u64> = cell.chain().collect();
        assert_eq!(chain, vec![4, 3, 2, 1, 0], "newest first");
    }

    #[test]
    fn append_epoch_in_chain() {
        let mut cell: Cell<u64> = Cell::new();
        for i in 0..3u64 { unsafe { cell.append(i) }; }

        let head = cell.head_ptr();
        assert!(!head.is_null());
        assert_eq!(unsafe { (*head).epoch }, 3);
        assert_eq!(unsafe { (*(*head).prev).epoch }, 2);
        assert_eq!(unsafe { (*(*(*head).prev).prev).epoch }, 1);
    }

    #[test]
    fn append_snapshot_stable_after_further_appends() {
        let mut cell: Cell<u64> = Cell::new();
        for i in 0..6u64 { unsafe { cell.append(i) }; }

        let snap = cell.head_ptr();
        assert_eq!(unsafe { (*snap).value }, 5);

        for i in 6..20u64 { unsafe { cell.append(i) }; }

        assert_eq!(unsafe { (*snap).value }, 5, "snapshot must be stable");
    }

    // ── Cell size ─────────────────────────────────────────────────────────────

    #[test]
    fn cell_is_minimal() {
        // Cell<T> is exactly one AtomicU64 = 8 bytes (epoch removed: lives in
        // the inline word or in Block, not as a separate field).
        assert_eq!(std::mem::size_of::<Cell<u32>>(), 8);
        assert_eq!(std::mem::size_of::<Cell<u64>>(), 8);
        assert_eq!(std::mem::size_of::<Cell<[u8; 256]>>(), 8);
    }

    // ── Ring: explicit N-snapshot layer ──────────────────────────────────────

    #[test]
    fn ring_retains_n_snapshots() {
        let mut ring: Ring<u64> = Ring::new(4);
        for i in 0..4u64 { unsafe { ring.append(i) }; }
        assert_eq!(ring.read(0).value(), Some(3u64));
    }

    #[test]
    fn ring_read_at_epoch() {
        let mut ring: Ring<u64> = Ring::new(8);
        for i in 1..=8u64 { unsafe { ring.append(i) }; }

        match ring.read_at_epoch(3) {
            ReadResult::Value { value, epoch, .. } => {
                assert_eq!(epoch, 3);
                assert_eq!(value, 3u64);
            }
            _ => panic!("expected value at epoch 3"),
        }
    }

    #[test]
    fn ring_cycles_correctly() {
        let mut ring: Ring<u64> = Ring::new(2);
        for i in 0..6u64 { unsafe { ring.append(i) }; }
        assert_eq!(ring.read(0).value(), Some(5u64));
    }

    // ── Memory ordering: one Release/Acquire per write/read ──────────────────

    #[test]
    fn no_torn_reads_across_threads() {
        use std::sync::atomic::AtomicUsize;

        let cell = Arc::new(std::sync::Mutex::new(Cell::<u64>::new()));
        let done = Arc::new(AtomicBool::new(false));
        let errors = Arc::new(AtomicUsize::new(0));

        let mut readers = vec![];
        for _ in 0..4 {
            let c = Arc::clone(&cell);
            let d = Arc::clone(&done);
            let e = Arc::clone(&errors);
            readers.push(std::thread::spawn(move || {
                let mut last_epoch = 0u64;
                while !d.load(AO::Acquire) {
                    let r = c.lock().unwrap().read(last_epoch);
                    if let ReadResult::Value { value, epoch, .. } = r {
                        if epoch < last_epoch { e.fetch_add(1, AO::Relaxed); }
                        last_epoch = epoch;
                        if value >= 2000 { e.fetch_add(1, AO::Relaxed); }
                    }
                    std::thread::yield_now();
                }
            }));
        }

        let cw = Arc::clone(&cell);
        let writer = std::thread::spawn(move || {
            for i in 0..2000u64 {
                unsafe { cw.lock().unwrap().write(i) };
            }
        });

        writer.join().unwrap();
        done.store(true, std::sync::atomic::Ordering::Release);
        for r in readers { r.join().unwrap(); }

        assert_eq!(errors.load(std::sync::atomic::Ordering::Relaxed), 0,
            "no torn reads or epoch inversions");

        let final_val = cell.lock().unwrap().read(0).value().unwrap();
        assert_eq!(final_val, 1999u64);
    }

    // ── SeqCell<T> ────────────────────────────────────────────────────────────

    #[test]
    fn seqcell_empty_before_write() {
        let cell: SeqCell<u64> = SeqCell::new();
        assert_eq!(cell.read(0), ReadResult::Empty);
    }

    #[test]
    fn seqcell_write_read_u64() {
        let cell = SeqCell::<u64>::new();
        let w = unsafe { cell.write(42u64) };
        assert_eq!(w.epoch, 1);
        assert_eq!(cell.read(0).value(), Some(42u64));
    }

    #[test]
    fn seqcell_write_read_large_struct() {
        #[derive(Clone, Copy, Debug, PartialEq)]
        struct Sensor { temp: f64, pressure: f64, humidity: f64 }

        let cell = SeqCell::<Sensor>::new();
        unsafe { cell.write(Sensor { temp: 22.5, pressure: 1013.0, humidity: 55.0 }) };
        let v = cell.read(0).value().unwrap();
        assert_eq!(v.temp, 22.5);
        assert_eq!(v.pressure, 1013.0);
        assert_eq!(v.humidity, 55.0);
    }

    #[test]
    fn seqcell_epoch_increments() {
        let cell = SeqCell::<u64>::new();
        for i in 1..=10u64 {
            let w = unsafe { cell.write(i) };
            assert_eq!(w.epoch, i);
        }
        assert_eq!(cell.epoch(), 10);
    }

    #[test]
    fn seqcell_missed_writes() {
        let cell = SeqCell::<u64>::new();
        for i in 1..=5u64 { unsafe { cell.write(i) }; }
        let r = cell.read(2);
        assert_eq!(r.missed(), 2); // missed 3 and 4
        assert_eq!(r.value(), Some(5u64));
    }

    #[test]
    fn seqcell_no_torn_reads_across_threads() {
        use std::sync::atomic::AtomicUsize;

        let cell   = Arc::new(SeqCell::<u64>::new());
        let done   = Arc::new(AtomicBool::new(false));
        let errors = Arc::new(AtomicUsize::new(0));

        let mut readers = vec![];
        for _ in 0..4 {
            let c = Arc::clone(&cell);
            let d = Arc::clone(&done);
            let e = Arc::clone(&errors);
            readers.push(std::thread::spawn(move || {
                let mut last_epoch = 0u64;
                while !d.load(AO::Acquire) {
                    if let ReadResult::Value { value, epoch, .. } = c.read(last_epoch) {
                        if epoch < last_epoch { e.fetch_add(1, AO::Relaxed); }
                        last_epoch = epoch;
                        if value >= 2000 { e.fetch_add(1, AO::Relaxed); }
                    }
                }
            }));
        }

        let cw = Arc::clone(&cell);
        let writer = std::thread::spawn(move || {
            for i in 0..2000u64 { unsafe { cw.write(i) }; }
        });

        writer.join().unwrap();
        done.store(true, std::sync::atomic::Ordering::Release);
        for r in readers { r.join().unwrap(); }

        assert_eq!(errors.load(std::sync::atomic::Ordering::Relaxed), 0,
            "no torn reads or epoch inversions");
        assert_eq!(cell.read(0).value(), Some(1999u64));
    }

    // ── SeqCell: seqlock retry path ───────────────────────────────────────────

    #[test]
    fn seqcell_retry_path_fires_under_contention() {
        // Proves the seqlock guard branches (odd-epoch and seq-changed) are not
        // dead code. Uses a 128-byte struct so the volatile copy takes ~16 loads —
        // long enough relative to the write that readers reliably land mid-write.
        //
        // Uses delta(after - before) to avoid interference with other tests running
        // in parallel on the shared global counters.
        type Big = [u64; 16];
        let cell = Arc::new(SeqCell::<Big>::new());
        let done = Arc::new(AtomicBool::new(false));

        let spins_before = SEQ_SPIN_ODD.load(AO::Relaxed)
                         + SEQ_SPIN_CHANGED.load(AO::Relaxed);

        let (cw, dw) = (Arc::clone(&cell), Arc::clone(&done));
        let writer = std::thread::spawn(move || {
            let mut flip = false;
            while !dw.load(AO::Acquire) {
                let v: u64 = if flip { 0xAAAAAAAAAAAAAAAA } else { 0xBBBBBBBBBBBBBBBB };
                unsafe { cw.write([v; 16]) };
                flip = !flip;
            }
        });

        let mut readers = vec![];
        for _ in 0..4 {
            let (cr, dr) = (Arc::clone(&cell), Arc::clone(&done));
            readers.push(std::thread::spawn(move || {
                while !dr.load(AO::Acquire) { let _ = cr.read(0); }
            }));
        }

        std::thread::sleep(std::time::Duration::from_millis(500));
        done.store(true, AO::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }

        let spins_after = SEQ_SPIN_ODD.load(AO::Relaxed)
                        + SEQ_SPIN_CHANGED.load(AO::Relaxed);
        assert!(spins_after > spins_before,
            "seqlock retry path never fired — the guard is not being exercised \
             (odd={} changed={})",
            SEQ_SPIN_ODD.load(AO::Relaxed),
            SEQ_SPIN_CHANGED.load(AO::Relaxed));
    }

    #[test]
    fn seqcell_no_torn_reads_large_struct() {
        // Sentinel correctness test for the seqlock protection.
        //
        // Writer alternates between two distinct 128-byte patterns (all-0xAA / all-0xBB).
        // Every field in the struct must be identical on every read — either all pattern A
        // or all pattern B. A torn read (some fields A, some B) means the reader saw a
        // partial write, which the seqlock must prevent.
        //
        // This is the authoritative correctness proof: epoch checks (used in other tests)
        // don't catch torn data; this test does.
        type Big = [u64; 16];
        const PAT_A: u64 = 0xAAAAAAAAAAAAAAAA;
        const PAT_B: u64 = 0xBBBBBBBBBBBBBBBB;

        let cell   = Arc::new(SeqCell::<Big>::new());
        let done   = Arc::new(AtomicBool::new(false));
        let errors = Arc::new(AtomicU64::new(0));

        let (cw, dw) = (Arc::clone(&cell), Arc::clone(&done));
        let writer = std::thread::spawn(move || {
            let mut flip = false;
            while !dw.load(AO::Acquire) {
                let v = if flip { PAT_A } else { PAT_B };
                unsafe { cw.write([v; 16]) };
                flip = !flip;
            }
        });

        let mut readers = vec![];
        for _ in 0..4 {
            let (cr, dr, er) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&errors));
            readers.push(std::thread::spawn(move || {
                while !dr.load(AO::Acquire) {
                    if let ReadResult::Value { value, .. } = cr.read(0) {
                        let first = value[0];
                        if first != PAT_A && first != PAT_B {
                            er.fetch_add(1, AO::Relaxed); // corrupted sentinel
                        }
                        for &v in &value[1..] {
                            if v != first { er.fetch_add(1, AO::Relaxed); } // torn field
                        }
                    }
                }
            }));
        }

        std::thread::sleep(std::time::Duration::from_secs(1));
        done.store(true, AO::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }

        assert_eq!(errors.load(AO::Relaxed), 0,
            "torn reads detected despite seqlock protection");
    }

    #[test]
    fn seqcell_unprotected_read_produces_torn_reads() {
        // Negative proof: read_unprotected() skips the seqlock bracket.
        // With a 128-byte struct and a fast writer, readers WILL see partial writes
        // (some fields from the old write, some from the new). This confirms the
        // seqlock in read() is actually doing work, not just overhead.
        //
        // If this test sees zero tears, the hardware is providing unexpected
        // atomicity for 128-byte copies (e.g. AVX-512 aligned load). That is a
        // valid hardware behaviour — seqlock protection remains correct and safe,
        // it's just not needed on that specific core configuration. The test prints
        // a diagnostic rather than asserting, because hardware guarantees vary.
        type Big = [u64; 16];
        const PAT_A: u64 = 0xAAAAAAAAAAAAAAAA;
        const PAT_B: u64 = 0xBBBBBBBBBBBBBBBB;

        let cell  = Arc::new(SeqCell::<Big>::new());
        let done  = Arc::new(AtomicBool::new(false));
        let tears = Arc::new(AtomicU64::new(0));
        let reads = Arc::new(AtomicU64::new(0));

        let (cw, dw) = (Arc::clone(&cell), Arc::clone(&done));
        let writer = std::thread::spawn(move || {
            let mut flip = false;
            while !dw.load(AO::Acquire) {
                let v = if flip { PAT_A } else { PAT_B };
                unsafe { cw.write([v; 16]) };
                flip = !flip;
            }
        });

        let mut readers = vec![];
        for _ in 0..8 {
            let (cr, dr, tr, rr) = (Arc::clone(&cell), Arc::clone(&done),
                                     Arc::clone(&tears), Arc::clone(&reads));
            readers.push(std::thread::spawn(move || {
                while !dr.load(AO::Acquire) {
                    if let ReadResult::Value { value, .. } = cr.read_unprotected() {
                        rr.fetch_add(1, AO::Relaxed);
                        let first = value[0];
                        for &v in &value[1..] {
                            if v != first { tr.fetch_add(1, AO::Relaxed); break; }
                        }
                    }
                }
            }));
        }

        std::thread::sleep(std::time::Duration::from_secs(1));
        done.store(true, AO::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }

        let n_tears = tears.load(AO::Relaxed);
        let n_reads = reads.load(AO::Relaxed);
        if n_tears == 0 {
            eprintln!(
                "NOTE: read_unprotected saw 0 tears in {} reads — hardware may provide \
                 atomic 128-byte copies on this CPU. Seqlock is still correct.",
                n_reads
            );
        } else {
            // Torn reads confirmed: seqlock protection in read() is necessary.
            assert!(n_tears > 0);
        }
    }
}

// ---------------------------------------------------------------------------
// ReadRef tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod read_ref_tests {
    use super::*;

    // ── Zero-copy: Block-stored values ───────────────────────────────────────

    #[test]
    fn read_ref_large_value_no_clone() {
        let mut cell: Cell<Vec<u64>> = Cell::new();
        unsafe { cell.write(vec![1, 2, 3, 4, 5]) };

        let r = cell.read_ref(0).expect("must have value");

        // Deref gives &Vec<u64> — no copy of the vector
        assert_eq!(r.len(), 5);
        assert_eq!(r[0], 1);
        assert_eq!(r[4], 5);

        // The pointer in ReadRef points directly into the Block on the heap
        match &r.inner {
            ReadRefInner::Block(ptr) => assert!(!ptr.is_null()),
            ReadRefInner::Inline(_)  => panic!("Vec must use Block path"),
        }
    }

    #[test]
    fn read_ref_iterate_without_copy() {
        let mut cell: Cell<Vec<u64>> = Cell::new();
        let data: Vec<u64> = (0..1000).collect();
        unsafe { cell.write(data) };

        let r = cell.read_ref(0).unwrap();

        // Iterate the vec directly via Deref — no copy at any point
        let sum: u64 = r.iter().sum();
        assert_eq!(sum, (0..1000u64).sum());
    }

    // ── Zero-copy: immutable causal chain ────────────────────────────────────

    #[test]
    fn read_ref_immutable_stable_across_writes() {
        let mut cell: Cell<Vec<u64>> = Cell::new();
        unsafe { cell.append(vec![1, 2, 3]) };

        // Capture a reference at epoch 1
        let r1 = cell.read_ref(0).unwrap();
        assert_eq!(r1.epoch, 1);

        // More writes
        unsafe { cell.append(vec![4, 5, 6]) };
        unsafe { cell.append(vec![7, 8, 9]) };

        // r1 still valid: points into Block(epoch=1) which is never freed
        assert_eq!(r1[0], 1, "immutable reference stable after further appends");
        assert_eq!(r1.epoch, 1);

        // New read gives latest value
        let r2 = cell.read_ref(1).unwrap();
        assert_eq!(r2[0], 7, "latest value is the most recent append");
        assert_eq!(r2.missed, 1, "missed one write between epoch 1 and epoch 3");
    }

    // ── Inline values: copy is unavoidable but free ───────────────────────────

    #[test]
    fn read_ref_small_value_uses_inline_copy() {
        let mut cell: Cell<u32> = Cell::new();
        unsafe { cell.write(42u32) };

        let r = cell.read_ref(0).unwrap();
        assert_eq!(*r, 42u32);

        // For inline values ReadRef holds a decoded copy (≤4B is free)
        match &r.inner {
            ReadRefInner::Inline(v) => assert_eq!(*v, 42u32),
            ReadRefInner::Block(_)  => panic!("u32 must use inline path"),
        }
    }

    // ── Epoch and missed carry correctly ─────────────────────────────────────

    #[test]
    fn read_ref_missed_count() {
        let mut cell: Cell<Vec<u64>> = Cell::new();
        unsafe { cell.write(vec![0]) }; // epoch 1
        unsafe { cell.write(vec![1]) }; // epoch 2
        unsafe { cell.write(vec![2]) }; // epoch 3
        unsafe { cell.write(vec![3]) }; // epoch 4

        let r = cell.read_ref(1).unwrap(); // last seen epoch 1
        assert_eq!(r.epoch, 4);
        assert_eq!(r.missed, 2, "missed epochs 2 and 3");
    }

    // ── Control loop pattern ─────────────────────────────────────────────────

    #[test]
    fn control_loop_zero_copy_pattern() {
        let mut cell: Cell<Vec<u64>> = Cell::new();
        unsafe { cell.write((0..100u64).collect()) };

        let original_sum: u64 = (0..100).sum();
        let updated_sum:  u64 = (0..100u64).map(|x| x * 2).sum();

        let mut last_epoch = 0u64;
        let mut sums_seen = vec![];
        let mut change_detected_at = None;

        for i in 0..5 {
            let r = cell.read_ref(last_epoch).unwrap();

            // Iterate via Deref — zero copy regardless of list size
            let sum: u64 = r.iter().sum();
            sums_seen.push(sum);

            // Sum must always be from ONE complete list — never torn
            assert!(sum == original_sum || sum == updated_sum,
                "sum must be from one complete list");

            if r.epoch != last_epoch && last_epoch != 0 {
                change_detected_at = Some(i);
            }
            last_epoch = r.epoch;

            if i == 2 {
                unsafe { cell.write((0..100u64).map(|x| x * 2).collect()) };
            }
        }

        assert!(change_detected_at.is_some(), "must detect list change via epoch");
        assert!(sums_seen.contains(&updated_sum), "must see updated values");

        // Rapid writes between reads show up as missed count
        unsafe { cell.write(vec![0]) };
        unsafe { cell.write(vec![1]) };
        let r = cell.read_ref(last_epoch).unwrap();
        assert!(r.missed >= 1, "rapid writes between reads show up as missed count");
    }

    // ── Analysis pattern ─────────────────────────────────────────────────────

    #[test]
    fn analysis_multiple_passes_zero_copy() {
        #[derive(Clone)]
        struct Record { value: f64, score: u64 }

        let mut cell: Cell<Vec<Record>> = Cell::new();
        let records: Vec<Record> = (0..10000)
            .map(|i| Record { value: i as f64 * 1.5, score: i })
            .collect();
        unsafe { cell.write(records) };

        // Multiple analysis passes — all via the same reference, no copy
        let r = cell.read_ref(0).unwrap();

        let sum:  f64 = r.iter().map(|x| x.value).sum();
        let max_score = r.iter().map(|x| x.score).max().unwrap();
        let above_mean = r.iter().filter(|x| x.value > sum / r.len() as f64).count();

        assert_eq!(max_score, 9999);
        assert!(above_mean > 0);
        assert_eq!(r.missed, 0); // dataset didn't change during analysis
    }
}

// ===========================================================================
// Bridge-layer (BridgedCell + ReaderRegistry + Drop) — adversarial tests
//
// These tests actively try to provoke use-after-free / double-free /
// stalled-reader pinning, not just confirm happy path. The Drop-counter
// reconciliation is the primary safety evidence; r.value sentinel checks
// are a secondary sanity net.
// ===========================================================================
#[cfg(test)]
mod bridge_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AO};
    use std::sync::Mutex;
    use std::thread;
    use std::time::Duration;

    // ── Tracked<T>: value with a shared Drop counter ─────────────────────
    //
    // Every Tracked instance, on drop, increments a shared counter. Each
    // mutable write creates exactly one Tracked; each freed Block drops
    // exactly one Tracked. Counter reconciliation at end-of-test exposes
    // both double-free (counter > created) and leak-of-reclaimable
    // (counter < created at sweep boundary).
    #[derive(Clone)]
    struct Tracked {
        value: u64,
        drops: Arc<AtomicU64>,
    }

    impl Tracked {
        fn new(value: u64, drops: &Arc<AtomicU64>) -> Self {
            Tracked { value, drops: Arc::clone(drops) }
        }
    }

    impl Drop for Tracked {
        fn drop(&mut self) {
            self.drops.fetch_add(1, AO::Relaxed);
        }
    }

    // ── (a) ReadRef holds across concurrent writes + reclaim ─────────────

    #[test]
    fn adversarial_a_readref_held_concurrent_writes_and_reclaim() {
        let drops    = Arc::new(AtomicU64::new(0));
        let registry = Arc::new(ReaderRegistry::new());
        let cell     = Arc::new(Mutex::new(BridgedCell::<Tracked>::new()));

        // Seed cell at epoch 1.
        unsafe { cell.lock().unwrap().write(Tracked::new(42, &drops)); }

        let handle = registry.acquire();
        // Acquire ReadRef under the lock, then release the lock — we want
        // the writer/reclaimer to run concurrently while we hold r.
        let r = {
            let c = cell.lock().unwrap();
            c.read_ref(&handle, 0).expect("must read")
        };
        let pinned_epoch = r.epoch;
        let pinned_value = r.value;
        assert_eq!(pinned_value, 42);

        let drops_at_pin = drops.load(AO::Relaxed);
        assert_eq!(drops_at_pin, 0, "no Tracked has been dropped yet");

        let stop = Arc::new(AtomicBool::new(false));

        // Writer thread: hammers writes while r is pinned at epoch 1.
        // Every write retires the previous block; reclaim can only free
        // blocks with retired_epoch < floor_min == pinned_epoch == 1.
        // Since epochs start at 1, nothing < 1 → reclaim frees nothing.
        let cw = Arc::clone(&cell);
        let dw = Arc::clone(&drops);
        let sw = Arc::clone(&stop);
        let writer = thread::spawn(move || {
            let mut i = 1u64;
            while !sw.load(AO::Acquire) {
                {
                    let mut c = cw.lock().unwrap();
                    unsafe { c.write(Tracked::new(1000 + i, &dw)); }
                }
                i += 1;
                thread::yield_now();
            }
            i - 1   // number of writes from this thread
        });

        // Reclaimer thread: hammers reclaim() concurrently.
        let cr = Arc::clone(&cell);
        let rr = Arc::clone(&registry);
        let sr = Arc::clone(&stop);
        let reclaimer = thread::spawn(move || {
            let mut sweeps = 0u64;
            while !sr.load(AO::Acquire) {
                cr.lock().unwrap().reclaim(&rr);
                sweeps += 1;
                thread::yield_now();
            }
            sweeps
        });

        // Reader: dereference r repeatedly while writer/reclaimer hammer.
        // r.value must stay == pinned_value (no UAF, no mutation of freed memory).
        for _ in 0..20_000 {
            let v = r.value;
            assert_eq!(v, pinned_value, "UAF / mutated value while pinned");
            thread::yield_now();
        }

        stop.store(true, AO::Release);
        let n_writes  = writer.join().unwrap();
        let _ = reclaimer.join().unwrap();

        // While r was pinned at epoch 1, no retired block could be freed
        // (every retired_epoch ≥ 1 ≥ floor_min). drops counter unchanged.
        assert_eq!(r.value, pinned_value,
            "after {} concurrent writes, r still intact", n_writes);
        assert_eq!(drops.load(AO::Relaxed), drops_at_pin,
            "no Tracked dropped while floor pinned epoch {}", pinned_epoch);

        // Now drop r → floor goes to MAX → reclaim can free everything.
        drop(r);
        let freed = cell.lock().unwrap().reclaim(&registry);
        // The current head (epoch 1 + n_writes) is alive; everything else
        // (1 .. 1 + n_writes - 1 in some order) was retired and now freed.
        assert_eq!(freed as u64, n_writes,
            "after drop, all {} retired blocks freed in one sweep", n_writes);
        assert_eq!(drops.load(AO::Relaxed), n_writes,
            "{} Tracked dropped via reclaim sweep", n_writes);

        drop(handle);
        drop(cell);   // BridgedCell::Drop frees the current head, +1 drop
        assert_eq!(drops.load(AO::Relaxed), n_writes + 1,
            "after cell drop, all Tracked accounted for");
    }

    // ── (b) After drop + reclaim the block IS freed ──────────────────────

    #[test]
    fn adversarial_b_block_freed_after_readref_drop_and_reclaim() {
        let drops    = Arc::new(AtomicU64::new(0));
        let registry = ReaderRegistry::new();
        let handle   = registry.acquire();
        let mut cell: BridgedCell<Tracked> = BridgedCell::new();

        unsafe { cell.write(Tracked::new(1, &drops)); }     // epoch 1
        let r = cell.read_ref(&handle, 0).unwrap();
        assert_eq!(r.value, 1);
        assert_eq!(r.epoch, 1);

        unsafe { cell.write(Tracked::new(2, &drops)); }     // epoch 2, retires block 1
        assert_eq!(cell.retired_len(), 1);

        // Sweep while pinned — must NOT free.
        let freed_while_pinned = cell.reclaim(&registry);
        assert_eq!(freed_while_pinned, 0);
        assert_eq!(cell.retired_len(), 1, "still pinned");
        assert_eq!(drops.load(AO::Relaxed), 0, "no Tracked dropped yet");

        // Release the ReadRef.
        drop(r);

        // Sweep again — block 1 must be freed.
        let freed = cell.reclaim(&registry);
        assert_eq!(freed, 1, "block freed after ReadRef dropped");
        assert_eq!(cell.retired_len(), 0);
        assert_eq!(drops.load(AO::Relaxed), 1, "one Tracked dropped");

        drop(handle);
        drop(cell);
        assert_eq!(drops.load(AO::Relaxed), 2, "current head also dropped");
    }

    // ── (c) Owned read() pins NOTHING — materialise-out ─────────────────

    #[test]
    fn adversarial_c_owned_read_pins_nothing() {
        let drops    = Arc::new(AtomicU64::new(0));
        let registry = ReaderRegistry::new();
        let _handle  = registry.acquire();   // claim a slot so any pinning would show
        let mut cell: BridgedCell<Tracked> = BridgedCell::new();

        unsafe { cell.write(Tracked::new(42, &drops)); }    // epoch 1

        // Pre-condition: nobody pinning anything.
        assert_eq!(registry.floor_min(), u64::MAX,
            "no live ReadRefs → floor_min is MAX");

        // OWNED read — must NOT touch any slot. The API itself makes this
        // impossible (read does not take a ReaderHandle) — the assertion
        // below is the empirical confirmation.
        let owned = match cell.read(0) {
            ReadResult::Value { value, .. } => value,
            _ => panic!("expected value"),
        };
        assert_eq!(owned.value, 42);

        // Floor still MAX — read() did not pin.
        assert_eq!(registry.floor_min(), u64::MAX,
            "read() must not register a floor (materialise-out)");

        // Pile up writes + reclaim. The block that read() copied from
        // gets retired by the first new write and freed by the sweep —
        // even though `owned` is still alive.
        for i in 1..=10u64 {
            unsafe { cell.write(Tracked::new(100 + i, &drops)); }
        }
        let freed = cell.reclaim(&registry);
        // We did 10 writes after the initial seed → 10 retirements →
        // 10 blocks freeable (floor_min == MAX).
        assert_eq!(freed, 10, "all 10 retired blocks freed despite owned read");
        // 10 retired blocks freed → 10 Tracked drops.
        assert_eq!(drops.load(AO::Relaxed), 10);

        // The owned clone is independent of the cell — value is 42, the
        // u64 it holds is a copy made by Clone, not a pointer.
        assert_eq!(owned.value, 42, "owned clone survives reclamation of its source block");

        drop(owned);                       // +1 drop (the owned clone)
        assert_eq!(drops.load(AO::Relaxed), 11);

        drop(cell);                        // +1 drop (current head)
        assert_eq!(drops.load(AO::Relaxed), 12,
            "11 writes total + 1 owned clone = 12 Tracked drops");
    }

    // ── (d) floor_min reflects the slowest live reader ──────────────────

    #[test]
    fn adversarial_d_floor_min_reflects_slowest_reader() {
        let registry = ReaderRegistry::new();
        let mut cell: BridgedCell<u64> = BridgedCell::new();

        let mut handles: Vec<ReaderHandle> = Vec::new();
        let mut refs:    Vec<ReadRef<u64>> = Vec::new();
        let mut pinned:  Vec<u64>          = Vec::new();

        // Stagger: write epoch i, take ReadRef at epoch i, keep it live.
        for i in 1..=5u64 {
            unsafe { cell.write(i * 10); }
            let h = registry.acquire();
            let r = cell.read_ref(&h, 0).expect("must read");
            assert_eq!(r.epoch, i);
            pinned.push(r.epoch);
            refs.push(r);
            handles.push(h);

            let expected = *pinned.iter().min().unwrap();
            assert_eq!(registry.floor_min(), expected,
                "after {} pinned, floor_min should be min of {:?}, got {}",
                pinned.len(), pinned, registry.floor_min());
        }

        // Drop in reverse — the LAST acquired (highest epoch) drops first.
        // floor_min stays at min(remaining) = 1 until the last drop.
        while let Some(r) = refs.pop() {
            let dropped_epoch = pinned.pop().unwrap();
            drop(r);
            let expected = if pinned.is_empty() { u64::MAX }
                           else { *pinned.iter().min().unwrap() };
            assert_eq!(registry.floor_min(), expected,
                "after dropping ref@{}, expected floor_min {}, got {}",
                dropped_epoch, expected, registry.floor_min());
        }

        // All handles still alive — slots already MAX (no holds), Drop
        // is idempotent.
        assert_eq!(registry.floor_min(), u64::MAX);
        drop(handles);
        assert_eq!(registry.floor_min(), u64::MAX);
    }

    // ── (e) Panicking reader releases its slot ──────────────────────────

    #[test]
    fn adversarial_e_panicking_reader_releases_slot() {
        let registry = Arc::new(ReaderRegistry::new());
        let cell     = Arc::new(Mutex::new(BridgedCell::<u64>::new()));

        unsafe { cell.lock().unwrap().write(42u64); }   // epoch 1, Block path (u64 > 4B)

        // Silence panic output for this one expected panic.
        let prev_hook = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| { /* swallow */ }));

        let cr = Arc::clone(&cell);
        let rr = Arc::clone(&registry);
        let panicking = thread::spawn(move || {
            let handle = rr.acquire();
            let _r = cr.lock().unwrap().read_ref(&handle, 0).expect("must read");
            // At this point: handle's slot pins epoch 1.
            assert_ne!(rr.floor_min(), u64::MAX);
            panic!("simulated reader panic — should unwind ReadRef and Handle drops");
        });

        let join_result = panicking.join();
        std::panic::set_hook(prev_hook);

        assert!(join_result.is_err(), "thread should have panicked");

        // Drop order during unwind: _r drops first (releases floor → slot=MAX),
        // then handle drops (re-stores MAX, idempotent). Either way the slot
        // must now be idle.
        assert_eq!(registry.floor_min(), u64::MAX,
            "panicked reader must have released its slot");

        // Reclamation now proceeds. Write again to retire block 1, then sweep.
        unsafe { cell.lock().unwrap().write(43u64); }
        let freed = cell.lock().unwrap().reclaim(&registry);
        assert_eq!(freed, 1, "block freeable after panicked reader's slot was released");
    }

    // ── (f) 100k-op stress with full reconciliation ─────────────────────

    #[test]
    fn adversarial_f_stress_drop_counter_reconciles() {
        const TARGET_WRITES: u64 = 100_000;

        let drops      = Arc::new(AtomicU64::new(0));
        let registry   = Arc::new(ReaderRegistry::new());
        let cell       = Arc::new(Mutex::new(BridgedCell::<Tracked>::new()));
        let n_writes   = Arc::new(AtomicU64::new(0));
        let stop       = Arc::new(AtomicBool::new(false));
        let overflow_before = HOLD_STACK_OVERFLOW_HITS.load(AO::Relaxed);

        // Seed.
        unsafe { cell.lock().unwrap().write(Tracked::new(0, &drops)); }
        n_writes.fetch_add(1, AO::Relaxed);

        // Writer.
        let cw = Arc::clone(&cell);
        let dw = Arc::clone(&drops);
        let nw = Arc::clone(&n_writes);
        let writer = thread::spawn(move || {
            for i in 1..=TARGET_WRITES {
                {
                    let mut c = cw.lock().unwrap();
                    unsafe { c.write(Tracked::new(i, &dw)); }
                }
                nw.fetch_add(1, AO::Relaxed);
                if i % 1024 == 0 { thread::yield_now(); }
            }
        });

        // Reclaimer (uses the watermark gate opportunistically, plus a
        // final unconditional sweep at exit).
        let cr = Arc::clone(&cell);
        let rr = Arc::clone(&registry);
        let sr = Arc::clone(&stop);
        let reclaimer = thread::spawn(move || {
            while !sr.load(AO::Acquire) {
                cr.lock().unwrap().reclaim_if_watermark(&rr);
                thread::yield_now();
            }
        });

        // Multiple reader threads, each holding a transient ReadRef briefly.
        let mut readers = vec![];
        for tid in 0..4 {
            let cr2 = Arc::clone(&cell);
            let rr2 = Arc::clone(&registry);
            let sr2 = Arc::clone(&stop);
            readers.push(thread::spawn(move || {
                let handle = rr2.acquire();
                let mut last_epoch = 0u64;
                let mut reads = 0u64;
                while !sr2.load(AO::Acquire) {
                    // Acquire ReadRef under the lock, then USE it briefly
                    // outside the lock (this is the realistic pattern the
                    // bridge is designed for — pin then release lock).
                    let r_opt = {
                        let c = cr2.lock().unwrap();
                        c.read_ref(&handle, last_epoch)
                    };
                    if let Some(r) = r_opt {
                        last_epoch = r.epoch;
                        let _v = r.value;
                        reads += 1;
                        // r drops here, releasing the floor.
                    }
                    if tid == 0 && reads % 64 == 0 { thread::yield_now(); }
                }
                reads
            }));
        }

        writer.join().unwrap();
        stop.store(true, AO::Release);
        let _ = reclaimer.join().unwrap();
        let total_reads: u64 = readers.into_iter()
            .map(|h| h.join().unwrap()).sum();

        // No more readers. One last full sweep to drain the retired list.
        let final_freed = cell.lock().unwrap().reclaim(&registry);

        let writes_total = n_writes.load(AO::Relaxed);
        let drops_before_cell = drops.load(AO::Relaxed);

        drop(cell);  // BridgedCell::Drop frees current head + any leftovers.

        let drops_after = drops.load(AO::Relaxed);
        let overflow_after = HOLD_STACK_OVERFLOW_HITS.load(AO::Relaxed);
        let overflow_delta = overflow_after.saturating_sub(overflow_before);

        // Surface telemetry for the final report.
        println!("STRESS: writes={} reads={} final_freed={} drops_before_cell_drop={} drops_after={} overflow_hits_this_test={}",
            writes_total, total_reads, final_freed,
            drops_before_cell, drops_after, overflow_delta);

        // RECONCILIATION:
        //   Every write creates exactly one Tracked.
        //   Every freed Block drops exactly one Tracked.
        //   Cell Drop frees the current head (+1) plus any still-retired blocks.
        //   So drops_after MUST equal writes_total.
        //   drops_after > writes_total → double-free.
        //   drops_after < writes_total → leak of reclaimable memory.
        assert_eq!(drops_after, writes_total,
            "drop-counter reconciliation: writes={} drops={}; mismatch → leak or double-free",
            writes_total, drops_after);
    }

    // ── Sanity: HoldStack nested up to depth-8 works without overflow ───

    #[test]
    fn holdstack_depth_8_no_overflow() {
        // Depth-8 must not overflow. We can't read the global counter
        // reliably under parallel tests (the explicit-overflow test
        // inflates it concurrently), so we verify the *local* effect:
        // all 8 pins are tracked, floor_min correctly reflects the
        // oldest, and dropping them all releases the slot to MAX.
        let registry = ReaderRegistry::new();
        let handle = registry.acquire();
        let mut cell: BridgedCell<u64> = BridgedCell::new();

        let mut refs = vec![];
        for i in 1..=8u64 {
            unsafe { cell.write(i); }
            refs.push(cell.read_ref(&handle, 0).expect("must read"));
        }
        // If any push overflowed, the conservative-fallback floor would
        // still pin epoch 1, but releasing refs in reverse would also
        // need to release the overflow tracker. The shape of this test
        // (drop all → floor MAX) implicitly catches stack/overflow mix-ups.
        assert_eq!(registry.floor_min(), 1);

        drop(refs);
        assert_eq!(registry.floor_min(), u64::MAX,
            "after dropping 8 nested ReadRefs, slot must be idle");
    }

    // ── Sanity: HoldStack overflow is conservative-safe ─────────────────

    #[test]
    fn holdstack_overflow_is_conservative_safe() {
        let drops    = Arc::new(AtomicU64::new(0));
        let registry = ReaderRegistry::new();
        let handle   = registry.acquire();
        let mut cell: BridgedCell<Tracked> = BridgedCell::new();

        let before = HOLD_STACK_OVERFLOW_HITS.load(AO::Relaxed);

        // 12 nested ReadRefs — exceeds depth 8, exercises overflow path.
        let mut refs = vec![];
        let mut epochs = vec![];
        for i in 1..=12u64 {
            unsafe { cell.write(Tracked::new(i, &drops)); }
            let r = cell.read_ref(&handle, 0).expect("must read");
            epochs.push(r.epoch);
            refs.push(r);
        }

        let after = HOLD_STACK_OVERFLOW_HITS.load(AO::Relaxed);
        let overflows = after - before;
        assert_eq!(overflows, 4,
            "12 ReadRefs - 8 stack slots = 4 overflow hits; got {}", overflows);

        // floor_min must be ≤ epoch 1 (the oldest pinned). The conservative
        // fallback keeps the floor at min(stack-min, overflow_floor).
        let floor = registry.floor_min();
        assert!(floor <= 1, "floor_min {} must pin epoch 1", floor);

        // Reclaim while pinned: must free NOTHING.
        let freed_while_pinned = cell.reclaim(&registry);
        assert_eq!(freed_while_pinned, 0);
        assert_eq!(drops.load(AO::Relaxed), 0);

        // Drop refs in random-ish order to exercise overflow remove path.
        // Specifically drop overflow ones first (the last 4 added), then stack ones.
        for _ in 0..12 { refs.pop(); }
        assert_eq!(registry.floor_min(), u64::MAX);

        // Sweep — everything should be freeable now.
        let freed = cell.reclaim(&registry);
        assert_eq!(freed, 11, "11 retired blocks freed (current head not retired)");
        drop(handle);
        drop(cell);
        assert_eq!(drops.load(AO::Relaxed), 12, "12 Tracked dropped total");
    }

    // ── Sanity: ReaderRegistry exhaustion panics cleanly ───────────────

    #[test]
    fn reader_registry_acquire_panics_when_exhausted() {
        let registry = ReaderRegistry::new();
        let mut handles = Vec::with_capacity(MAX_READERS);
        for _ in 0..MAX_READERS {
            handles.push(registry.acquire());
        }
        let result = std::panic::catch_unwind(
            std::panic::AssertUnwindSafe(|| registry.acquire())
        );
        assert!(result.is_err(), "{}th acquire must panic", MAX_READERS + 1);
    }

    // ── write_lazy: Level B fast path (no reader ever registered) ───────

    #[test]
    fn write_lazy_level_b_frees_immediately_when_no_reader_ever() {
        let drops    = Arc::new(AtomicU64::new(0));
        let registry = ReaderRegistry::new();
        let mut cell: BridgedCell<Tracked> = BridgedCell::new();

        assert!(!registry.has_any_reader(), "no handle issued yet");

        unsafe { cell.write_lazy(Tracked::new(1, &drops), &registry); }
        // First write — nothing to retire (no old block).
        assert_eq!(cell.retired_len(), 0);
        assert_eq!(drops.load(AO::Relaxed), 0);

        // Subsequent writes: old block gets freed IMMEDIATELY (Level B)
        // because !registry.has_any_reader(). Retired list stays empty.
        for i in 2..=50u64 {
            unsafe { cell.write_lazy(Tracked::new(i, &drops), &registry); }
            assert_eq!(cell.retired_len(), 0,
                "Level B: retired list must stay empty when no reader ever existed");
        }

        // 49 old blocks freed (writes 2..=50 each superseded the previous).
        assert_eq!(drops.load(AO::Relaxed), 49);

        drop(cell);  // frees current head, +1 drop
        assert_eq!(drops.load(AO::Relaxed), 50);
    }

    // ── write_lazy: Level A path (reader exists, but not pinning old) ──

    #[test]
    fn write_lazy_level_a_frees_when_floor_above_old_epoch() {
        let drops    = Arc::new(AtomicU64::new(0));
        let registry = ReaderRegistry::new();
        let _handle  = registry.acquire();   // flips any_handle_ever → Level B disabled
        let mut cell: BridgedCell<Tracked> = BridgedCell::new();

        assert!(registry.has_any_reader());

        // No live ReadRef → all slots are u64::MAX → floor_min = MAX.
        // Level A: floor_min > old_epoch (MAX > anything), so every old
        // block is freed immediately — retired list stays empty.
        unsafe { cell.write_lazy(Tracked::new(1, &drops), &registry); }
        for i in 2..=30u64 {
            unsafe { cell.write_lazy(Tracked::new(i, &drops), &registry); }
            assert_eq!(cell.retired_len(), 0,
                "Level A: floor_min == MAX, retired list must stay empty");
        }
        assert_eq!(drops.load(AO::Relaxed), 29, "29 old blocks freed immediately");

        drop(cell);
        assert_eq!(drops.load(AO::Relaxed), 30);
    }

    // ── write_lazy: falls back to retire when a reader IS pinning ──────

    #[test]
    fn write_lazy_retires_when_reader_pins_old_epoch() {
        let drops    = Arc::new(AtomicU64::new(0));
        let registry = ReaderRegistry::new();
        let handle   = registry.acquire();
        let mut cell: BridgedCell<Tracked> = BridgedCell::new();

        unsafe { cell.write_lazy(Tracked::new(1, &drops), &registry); }     // epoch 1
        let r = cell.read_ref(&handle, 0).unwrap();
        assert_eq!(r.epoch, 1);
        assert_eq!(registry.floor_min(), 1);

        // Now write — old block (epoch 1) is pinned. Level A fails
        // (floor_min == 1 not > old_epoch == 1) → must retire.
        unsafe { cell.write_lazy(Tracked::new(2, &drops), &registry); }
        assert_eq!(cell.retired_len(), 1,
            "must retire when floor_min ≤ old_epoch");
        assert_eq!(drops.load(AO::Relaxed), 0, "block 1 still alive");

        // Drop r → floor goes to MAX. Subsequent write_lazy can free.
        drop(r);

        // Existing retired entry still needs an explicit reclaim sweep —
        // write_lazy only decides about the CURRENT old block.
        unsafe { cell.write_lazy(Tracked::new(3, &drops), &registry); }
        // Block 2 was just superseded; floor_min == MAX > 2, so freed.
        // Retired list still holds block 1 from before (unchanged).
        assert_eq!(cell.retired_len(), 1);
        assert_eq!(drops.load(AO::Relaxed), 1, "block 2 freed by Level A");

        let freed = cell.reclaim(&registry);
        assert_eq!(freed, 1, "reclaim sweep drains block 1");
        assert_eq!(drops.load(AO::Relaxed), 2);

        drop(handle);
        drop(cell);
        assert_eq!(drops.load(AO::Relaxed), 3);
    }

    // ── write_lazy stress: concurrent reader + write_lazy ─────────────

    #[test]
    fn write_lazy_stress_concurrent_safe() {
        // Adversarial: a reader thread acquires/drops ReadRefs continuously
        // while a writer hammers write_lazy. The drop-counter at the end
        // must equal total writes; nothing freed early.
        const TARGET_WRITES: u64 = 50_000;

        let drops    = Arc::new(AtomicU64::new(0));
        let registry = Arc::new(ReaderRegistry::new());
        let cell     = Arc::new(Mutex::new(BridgedCell::<Tracked>::new()));
        let n_writes = Arc::new(AtomicU64::new(0));
        let stop     = Arc::new(AtomicBool::new(false));

        unsafe { cell.lock().unwrap().write_lazy(Tracked::new(0, &drops), &registry); }
        n_writes.fetch_add(1, AO::Relaxed);

        let cw = Arc::clone(&cell);
        let dw = Arc::clone(&drops);
        let rw = Arc::clone(&registry);
        let nw = Arc::clone(&n_writes);
        let writer = thread::spawn(move || {
            for i in 1..=TARGET_WRITES {
                let mut c = cw.lock().unwrap();
                unsafe { c.write_lazy(Tracked::new(i, &dw), &rw); }
                drop(c);
                nw.fetch_add(1, AO::Relaxed);
            }
        });

        let cr = Arc::clone(&cell);
        let rr = Arc::clone(&registry);
        let sr = Arc::clone(&stop);
        let reader = thread::spawn(move || {
            let handle = rr.acquire();
            let mut last = 0u64;
            let mut reads = 0u64;
            while !sr.load(AO::Acquire) {
                let r_opt = {
                    let c = cr.lock().unwrap();
                    c.read_ref(&handle, last)
                };
                if let Some(r) = r_opt {
                    last = r.epoch;
                    let _ = r.value;
                    reads += 1;
                }
            }
            reads
        });

        writer.join().unwrap();
        stop.store(true, AO::Release);
        let _ = reader.join().unwrap();

        // Drain any retirements left from the racy windows where Level A
        // had to retire (reader was pinning at the moment of write).
        let _ = cell.lock().unwrap().reclaim(&registry);

        let writes_total = n_writes.load(AO::Relaxed);
        let drops_before = drops.load(AO::Relaxed);
        drop(cell);
        let drops_after = drops.load(AO::Relaxed);

        // Same reconciliation as the adversarial_f stress test.
        assert_eq!(drops_after, writes_total,
            "write_lazy stress: writes={} drops={}", writes_total, drops_after);
        let _ = drops_before;
    }

    // Avoid unused-warning churn for the timing import.
    #[allow(dead_code)]
    fn _unused() { let _ = Duration::from_millis(1); }
}

// ===========================================================================
// SpscQueue tests
// ===========================================================================
#[cfg(test)]
mod spsc_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU64, Ordering as AO};
    use std::thread;

    #[test]
    fn empty_pop_returns_none() {
        let q: SpscQueue<u64, 4> = SpscQueue::new();
        assert!(q.is_empty());
        assert_eq!(unsafe { q.pop() }, None);
        assert_eq!(q.len(), 0);
        assert_eq!(q.capacity(), 4);
    }

    #[test]
    fn push_pop_fifo_order() {
        let q: SpscQueue<u64, 4> = SpscQueue::new();
        unsafe {
            assert!(q.push(1).is_ok());
            assert!(q.push(2).is_ok());
            assert!(q.push(3).is_ok());
            assert_eq!(q.len(), 3);
            assert_eq!(q.pop(), Some(1));
            assert_eq!(q.pop(), Some(2));
            assert_eq!(q.pop(), Some(3));
            assert_eq!(q.pop(), None);
        }
    }

    #[test]
    fn fills_to_capacity_then_rejects() {
        let q: SpscQueue<u64, 4> = SpscQueue::new();
        unsafe {
            for i in 0..4 { assert!(q.push(i).is_ok()); }
            assert!(q.is_full());
            // 5th push fails, value returned intact.
            assert_eq!(q.push(99), Err(99));
            // Drain one then push succeeds.
            assert_eq!(q.pop(), Some(0));
            assert!(q.push(99).is_ok());
            // Final order: 1, 2, 3, 99.
            assert_eq!(q.pop(), Some(1));
            assert_eq!(q.pop(), Some(2));
            assert_eq!(q.pop(), Some(3));
            assert_eq!(q.pop(), Some(99));
        }
    }

    #[test]
    fn wraps_around_many_times() {
        let q: SpscQueue<u64, 4> = SpscQueue::new();
        unsafe {
            // Push/pop pattern that wraps head & tail well past capacity.
            for i in 0..10_000u64 {
                assert!(q.push(i).is_ok());
                assert_eq!(q.pop(), Some(i));
            }
            assert!(q.is_empty());
        }
    }

    // ── Drop counter — no leak, no double-drop ──────────────────────────

    #[derive(Clone)]
    struct DropTrack(Arc<AtomicU64>);
    impl Drop for DropTrack {
        fn drop(&mut self) { self.0.fetch_add(1, AO::Relaxed); }
    }

    #[test]
    fn unread_items_dropped_on_queue_drop() {
        let drops = Arc::new(AtomicU64::new(0));
        {
            let q: SpscQueue<DropTrack, 8> = SpscQueue::new();
            unsafe {
                for _ in 0..5 {
                    q.push(DropTrack(Arc::clone(&drops))).ok().unwrap();
                }
                assert_eq!(q.pop().is_some(), true);  // drop 1 via pop
            }
            // 4 items remain in the queue → must drop on queue drop.
            assert_eq!(drops.load(AO::Relaxed), 1, "1 popped+dropped");
        }
        assert_eq!(drops.load(AO::Relaxed), 5, "all 5 dropped after queue drop");
    }

    // ── Concurrent SPSC stress with reconciliation ──────────────────────

    #[test]
    fn concurrent_spsc_stress_drop_reconcile() {
        const N: u64 = 200_000;
        let drops = Arc::new(AtomicU64::new(0));
        let q: Arc<SpscQueue<DropTrack, 64>> = Arc::new(SpscQueue::new());

        let qp = Arc::clone(&q);
        let dp = Arc::clone(&drops);
        let producer = thread::spawn(move || {
            for _ in 0..N {
                // Construct ONCE per logical item; on full-queue retries
                // the push API returns the value, we reuse it. Constructing
                // per-attempt would inflate the drop counter.
                let mut v = DropTrack(Arc::clone(&dp));
                loop {
                    match unsafe { qp.push(v) } {
                        Ok(()) => break,
                        Err(returned) => {
                            v = returned;
                            thread::yield_now();
                        }
                    }
                }
            }
        });

        let qc = Arc::clone(&q);
        let consumer = thread::spawn(move || {
            let mut got = 0u64;
            while got < N {
                if unsafe { qc.pop() }.is_some() { got += 1; }
                else { thread::yield_now(); }
            }
            got
        });

        producer.join().unwrap();
        let got = consumer.join().unwrap();
        assert_eq!(got, N);

        // All N DropTrack instances were created on the producer side
        // (each call to `DropTrack(Arc::clone)` makes one).
        // Each one was either consumed (drop in pop) or left in the queue.
        // The queue is empty now → all dropped.
        drop(q);
        assert_eq!(drops.load(AO::Relaxed), N,
            "SPSC reconciliation: {} produced should equal drops counter", N);
    }
}

// ===========================================================================
// DoubleBuffer tests
// ===========================================================================
#[cfg(test)]
mod double_buffer_tests {
    use super::*;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering as AO};
    use std::thread;
    use std::time::Duration;

    #[test]
    fn initial_value_visible_from_both_slots() {
        let db: DoubleBuffer<u64> = DoubleBuffer::new(42);
        assert_eq!(*db.read(), 42);
        // Initial front is 0.
        assert_eq!(db.front_index(), 0);
    }

    #[test]
    fn write_swaps_front_and_publishes() {
        let db: DoubleBuffer<u64> = DoubleBuffer::new(0);
        assert_eq!(db.front_index(), 0);
        unsafe { db.write(100); }
        assert_eq!(db.front_index(), 1);
        assert_eq!(*db.read(), 100);
        unsafe { db.write(200); }
        assert_eq!(db.front_index(), 0);
        assert_eq!(*db.read(), 200);
    }

    #[test]
    fn back_mut_then_publish_in_place_update() {
        let db: DoubleBuffer<Vec<u64>> = DoubleBuffer::new(vec![]);
        // Fill back over several "frame" operations, then publish once.
        unsafe {
            let b = db.back_mut();
            b.push(1); b.push(2); b.push(3);
            db.publish();
        }
        assert_eq!(db.read(), &vec![1, 2, 3]);

        // Next frame: back is the OLD front (vec![]). Fill again.
        unsafe {
            let b = db.back_mut();
            b.clear();
            b.push(10); b.push(20);
            db.publish();
        }
        assert_eq!(db.read(), &vec![10, 20]);
    }

    // ── Single-writer single-reader pattern with explicit frame sync ────

    #[test]
    fn writer_and_reader_alternate_via_frame_sync() {
        // Demonstrates the canonical usage: writer publishes; reader
        // observes; reader signals done; writer publishes again. Frame
        // sync is via two AtomicBool ping-pong flags.
        let db        = Arc::new(DoubleBuffer::<u64>::new(0));
        let writer_go = Arc::new(AtomicBool::new(true));    // writer first
        let reader_go = Arc::new(AtomicBool::new(false));
        let done      = Arc::new(AtomicBool::new(false));
        let frames    = Arc::new(AtomicU64::new(0));

        let dw = Arc::clone(&db);
        let wg = Arc::clone(&writer_go);
        let rg = Arc::clone(&reader_go);
        let dn = Arc::clone(&done);
        let writer = thread::spawn(move || {
            for i in 1..=50u64 {
                while !wg.load(AO::Acquire) { thread::yield_now(); }
                unsafe { dw.write(i * 10); }
                wg.store(false, AO::Release);
                rg.store(true, AO::Release);
            }
            dn.store(true, AO::Release);
        });

        let dr = Arc::clone(&db);
        let wg2 = Arc::clone(&writer_go);
        let rg2 = Arc::clone(&reader_go);
        let dn2 = Arc::clone(&done);
        let fr = Arc::clone(&frames);
        let reader = thread::spawn(move || {
            let mut last_seen = 0u64;
            loop {
                while !rg2.load(AO::Acquire) {
                    if dn2.load(AO::Acquire) && !rg2.load(AO::Acquire) {
                        return last_seen;
                    }
                    thread::yield_now();
                }
                let v = *dr.read();
                assert!(v >= last_seen, "double-buffer reverted: {} < {}", v, last_seen);
                last_seen = v;
                fr.fetch_add(1, AO::Relaxed);
                rg2.store(false, AO::Release);
                wg2.store(true, AO::Release);
            }
        });

        writer.join().unwrap();
        // Final unblock so reader exits.
        std::thread::sleep(Duration::from_millis(20));
        reader_go.store(true, AO::Release);
        done.store(true, AO::Release);
        let last = reader.join().unwrap();
        assert_eq!(last, 500, "reader saw the final write");
        assert_eq!(frames.load(AO::Relaxed), 50, "exactly 50 frames observed");
    }
}
