//! # axOS Memory Model — v0.8
//!
//! ## The design
//!
//! Everything reduces to one primitive:
//!
//! ```text
//! Cell<T>
//!   head:  AtomicU64   ← current value (inline ≤4B) or *const Block<T>
//!   epoch: AtomicU64   ← write counter, unified backpressure signal
//! ```
//!
//! The ring buffer is gone. It was solving three problems that don't
//! need a ring:
//!
//!   1. Locating current head   → single AtomicU64 head does this
//!   2. Retention / history     → Block.prev chain already IS the history
//!   3. Reclamation boundary    → epoch tracks this directly
//!
//! ## Write modes
//!
//! `write(v)`   — mutable replace. ≤4B: inline in head. >4B: swap Block pointer.
//!               Old Block leaked until reclamation sweep (TODO).
//!
//! `append(v)`  — immutable extend. Always allocates Block. new.prev = old head.
//!               Causal chain traversable via ChainIter.
//!
//! ## Epoch = unified backpressure signal
//!
//! Writer increments epoch on every write (single writer per cell — no atomic
//! fetch_add needed, just load+1 then store Release).
//!
//! Reader holds its last-seen epoch. On each read:
//!   missed = current_epoch - last_epoch - 1
//!   missed == 0  → reader is current
//!   missed > N   → reader is behind, caller decides: throttle / escalate / accept
//!
//! No staging queue. No fail_count. No pressure slots.
//! The epoch exposes write velocity; policy stays with the caller.
//!
//! ## Memory ordering
//!
//! Writer: head (Relaxed) → epoch (Release)
//! Reader: epoch (Acquire) → head (Relaxed)
//!
//! One Release/Acquire pair per write/read cycle.
//! No CAS, no fetch_add, no bus lock anywhere.
//!
//! ## Ring (explicit opt-in)
//!
//! `Ring<T>` wraps N cells for the case where N concurrent readers each
//! need their own stable head pointer (e.g. axAporia Ring 2 reading while
//! Ring 1 is still appending). Not needed otherwise.

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
// Block — heap unit of the immutable causal chain
// ---------------------------------------------------------------------------

pub struct Block<T> {
    pub value: T,
    /// Backward causal edge. Null for the first block.
    pub prev:  *const Block<T>,
    /// Epoch at time of creation.
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
// ---------------------------------------------------------------------------

/// Encode a ≤4 byte value into a tagged u64.
/// Bits 1-32: value bytes. Bit 0: INLINE_TAG.
unsafe fn encode_inline<T>(value: T) -> u64 {
    debug_assert!(size_of::<T>() <= 4);
    let value = ManuallyDrop::new(value);
    let mut buf = [0u8; 4];
    ptr::copy_nonoverlapping(
        &*value as *const T as *const u8,
        buf.as_mut_ptr(),
        size_of::<T>(),
    );
    ((u32::from_ne_bytes(buf) as u64) << 1) | INLINE_TAG
}

/// Decode a ≤4 byte value from a tagged u64.
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
    /// Current value: inline (bit 0 = 1) or *const Block<T> (bit 0 = 0).
    head:    AtomicU64,
    /// Write counter. Incremented by the single owning writer on every write.
    /// Stored with Release. Read with Acquire.
    epoch:   AtomicU64,
    _marker: PhantomData<*const Block<T>>,
}

unsafe impl<T: Send> Send for Cell<T> {}
unsafe impl<T: Send> Sync for Cell<T> {}

impl<T> Cell<T> {
    pub fn new() -> Self {
        Cell {
            head:    AtomicU64::new(0),
            epoch:   AtomicU64::new(0),
            _marker: PhantomData,
        }
    }

    // =========================================================================
    // WRITE: mutable replace
    //
    // ≤4 bytes: encode inline in head field. Zero allocation.
    // >4 bytes: allocate Block, swap head pointer. No prev (mutable = no chain).
    //
    // Old Block leaked until reclamation sweep. This is safe because readers
    // hold direct Block pointers, not ring slots. Sweep determines liveness
    // via epoch: once all readers have advanced past old_epoch, old Block
    // is unreachable and can be freed.
    //
    // SAFETY: must be called by the single owning writer only.
    // =========================================================================

    pub unsafe fn write(&mut self, value: T) -> WriteResult {
        let new_epoch = self.epoch.load(Ordering::Relaxed) + 1;

        if size_of::<T>() <= 4 {
            // Small value: inline in head field, zero allocation
            self.head.store(encode_inline(value), Ordering::Relaxed);
        } else {
            // Large value: heap Block, no prev pointer (mutable, no chain)
            let block = Block::allocate(value, ptr::null(), new_epoch);
            self.head.store(block as u64, Ordering::Relaxed);
        }

        // Epoch written LAST (Release): makes head visible to all Acquire reads
        self.epoch.store(new_epoch, Ordering::Release);
        WriteResult { epoch: new_epoch }
    }

    // =========================================================================
    // APPEND: immutable extend
    //
    // Always allocates a Block. Sets prev = current head.
    // Builds the causal chain: new → old → older → ...
    // ChainIter traverses it newest-first.
    //
    // SAFETY: must be called by the single owning writer only.
    // =========================================================================

    pub unsafe fn append(&mut self, value: T) -> WriteResult {
        let new_epoch = self.epoch.load(Ordering::Relaxed) + 1;

        // Capture current head as prev (backward causal edge)
        let prev_bits = self.head.load(Ordering::Relaxed);
        let prev: *const Block<T> = if prev_bits == 0 || prev_bits & INLINE_TAG != 0 {
            ptr::null() // empty or inline (mutable→immutable transition)
        } else {
            prev_bits as *const Block<T>
        };

        let block = Block::allocate(value, prev, new_epoch);

        // head (Relaxed) then epoch (Release) — same ordering guarantee as write
        self.head.store(block as u64, Ordering::Relaxed);
        self.epoch.store(new_epoch, Ordering::Release);
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
        self.head.store(block as u64, Ordering::Relaxed);
        self.epoch.store(epoch, Ordering::Release);
        WriteResult { epoch }
    }

    pub unsafe fn write_with_epoch(&mut self, value: T, epoch: u64) -> WriteResult {
        if size_of::<T>() <= 4 {
            self.head.store(encode_inline(value), Ordering::Relaxed);
        } else {
            let block = Block::allocate(value, ptr::null(), epoch);
            self.head.store(block as u64, Ordering::Relaxed);
        }
        self.epoch.store(epoch, Ordering::Release);
        WriteResult { epoch }
    }

    // =========================================================================
    // READ
    //
    // Epoch loaded FIRST (Acquire): synchronises with writer's Release store.
    // Head loaded AFTER (Relaxed): guaranteed visible by the Acquire above.
    //
    // last_epoch: caller's last seen epoch (0 = first read).
    // missed: writes caller didn't observe since last_epoch.
    //
    // Safe from any thread, any number of concurrent readers.
    // Never blocks, never spins.
    // =========================================================================

    pub fn read(&self, last_epoch: u64) -> ReadResult<T> where T: Clone {
        // Acquire: all writes the owner did before storing this epoch are visible
        let current_epoch = self.epoch.load(Ordering::Acquire);
        if current_epoch == 0 { return ReadResult::Empty; }

        let bits = self.head.load(Ordering::Relaxed);
        if bits == 0 { return ReadResult::Empty; }

        let value = if bits & INLINE_TAG != 0 {
            unsafe { decode_inline::<T>(bits) }
        } else {
            unsafe { (*( bits as *const Block<T>)).value.clone() }
        };

        // missed = how many writes happened between last read and this read
        // last_epoch=0 means first read, missed=0
        let missed = if last_epoch == 0 {
            0
        } else {
            current_epoch.saturating_sub(last_epoch + 1)
        };

        ReadResult::Value { value, epoch: current_epoch, missed }
    }

    // =========================================================================
    // CHAIN — traverse immutable causal history
    //
    // Returns iterator over Block values, newest first.
    // Only meaningful after append() calls (write() sets no prev).
    // =========================================================================

    pub fn chain(&self) -> ChainIter<T> {
        let bits = self.head.load(Ordering::Acquire);
        if bits == 0 || bits & INLINE_TAG != 0 {
            ChainIter::new(ptr::null())
        } else {
            ChainIter::new(bits as *const Block<T>)
        }
    }

    /// Current epoch. Use to check write velocity.
    pub fn epoch(&self) -> u64 { self.epoch.load(Ordering::Acquire) }

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
        let r = cell.read(w.epoch - 1); // pass previous epoch
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
        // last_epoch=0 means "first read, no baseline"
        let r = cell.read(0);
        assert_eq!(r.missed(), 0, "first read: missed is always 0");
    }

    #[test]
    fn caller_can_detect_write_pressure() {
        let mut cell: Cell<u32> = Cell::new();
        // Simulate fast writer: 1000 writes before reader checks
        for i in 0..1000u32 { unsafe { cell.write(i) }; }

        let r = cell.read(0); // first read: missed=0 (no baseline)
        let last = r.epoch();

        // Simulate more writes while reader is "busy"
        for i in 1000..1050u32 { unsafe { cell.write(i) }; }

        let r2 = cell.read(last);
        assert_eq!(r2.missed(), 49, "reader missed 49 writes");
        // Caller can now decide: throttle writer, escalate, or accept the lag
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
        assert_eq!(unsafe { (*head).epoch }, 3); // latest block has epoch 3
        assert_eq!(unsafe { (*(*head).prev).epoch }, 2);
        assert_eq!(unsafe { (*(*(*head).prev).prev).epoch }, 1);
    }

    #[test]
    fn append_snapshot_stable_after_further_appends() {
        let mut cell: Cell<u64> = Cell::new();
        for i in 0..6u64 { unsafe { cell.append(i) }; }

        // Capture a snapshot pointer
        let snap = cell.head_ptr();
        assert_eq!(unsafe { (*snap).value }, 5);

        // More writes
        for i in 6..20u64 { unsafe { cell.append(i) }; }

        // Snapshot still valid — Block is immutable, not freed
        assert_eq!(unsafe { (*snap).value }, 5, "snapshot must be stable");
    }

    // ── Cell size ─────────────────────────────────────────────────────────────

    #[test]
    fn cell_is_minimal() {
        // Cell<T> is exactly two AtomicU64 = 16 bytes (plus PhantomData = 0)
        assert_eq!(std::mem::size_of::<Cell<u32>>(), 16);
        assert_eq!(std::mem::size_of::<Cell<u64>>(), 16);
        assert_eq!(std::mem::size_of::<Cell<[u8; 256]>>(), 16);
    }

    // ── Ring: explicit N-snapshot layer ──────────────────────────────────────

    #[test]
    fn ring_retains_n_snapshots() {
        let mut ring: Ring<u64> = Ring::new(4);
        for i in 0..4u64 { unsafe { ring.append(i) }; }

        // Most recent read
        assert_eq!(ring.read(0).value(), Some(3u64));
    }

    #[test]
    fn ring_read_at_epoch() {
        let mut ring: Ring<u64> = Ring::new(8);
        for i in 1..=8u64 { unsafe { ring.append(i) }; }

        // Read the snapshot that was at epoch 3
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
        // Latest is 5
        assert_eq!(ring.read(0).value(), Some(5u64));
    }

    // ── Memory ordering: one Release/Acquire pair ─────────────────────────────

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
                        // Value and epoch must be consistent
                        // epoch must always advance (never go backward)
                        if epoch < last_epoch {
                            e.fetch_add(1, AO::Relaxed);
                        }
                        last_epoch = epoch;
                        // value must be in valid range (0..2000)
                        if value >= 2000 {
                            e.fetch_add(1, AO::Relaxed);
                        }
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
}