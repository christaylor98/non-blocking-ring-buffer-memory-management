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
use std::sync::atomic::{AtomicU64, Ordering};

/// Tag bit: head field encodes inline value when bit 0 is set.
/// Block pointers are always ≥8-byte aligned so bit 0 is always 0 for pointers.
const INLINE_TAG: u64 = 1;

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
}

impl<T> Default for Cell<T> { fn default() -> Self { Cell::new() } }

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
            if e0 == 0        { return ReadResult::Empty; }
            if e0 % 2 == 1    { std::thread::yield_now(); continue; } // write in progress

            // Volatile read: prevents the compiler from hoisting this read
            // outside the seq bracket.
            let value = unsafe {
                ptr::read_volatile((*self.slot.get()).as_ptr())
            };

            let e1 = self.seq.load(Ordering::Acquire);
            if e0 != e1       { std::thread::yield_now(); continue; } // write landed mid-read

            let current_epoch = e0 / 2;
            let missed = if last_epoch == 0 { 0 }
                         else { current_epoch.saturating_sub(last_epoch + 1) };
            return ReadResult::Value { value, epoch: current_epoch, missed };
        }
    }

    /// Current epoch. Safe to call from any thread.
    pub fn epoch(&self) -> u64 { self.seq.load(Ordering::Acquire) / 2 }
}

impl<T: Copy> Default for SeqCell<T> { fn default() -> Self { SeqCell::new() } }

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

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
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AO};

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
        use std::sync::Arc;
        use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering as AO};

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
}
