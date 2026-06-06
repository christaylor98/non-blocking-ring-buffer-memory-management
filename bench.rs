// bench.rs — Cell<T> performance benchmarks
//
// Build:  rustc -O bench.rs -o bench
// Run:    ./bench
//
// Compares Cell<u32> (inline path: 1 Release store / 1 Acquire load)
// against Mutex<u32> and RwLock<u32> across four scenarios:
//
//   1. Single-thread baseline
//   2. Sparse writes  — writer sleeps 1 ms between writes, readers flat-out
//   3. Heavy writes   — writer flat-out, readers sleep 100 µs between reads
//   4. Full stress    — writer + all readers flat-out, correctness checked

#[path = "non-blocking-memory.rs"]
mod mem;

use mem::{Cell, SeqCell, ReadResult, BridgedCell, ReaderRegistry, SpscQueue, DoubleBuffer};
use std::collections::VecDeque;
use std::cell::UnsafeCell;
use std::sync::{Arc, Mutex, RwLock};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::{Duration, Instant};
use std::thread;

const BENCH_SECS: u64 = 2;
fn bench_dur() -> Duration { Duration::from_secs(BENCH_SECS) }

fn fmt_rate(ops: u64, elapsed: f64) -> String {
    let r = ops as f64 / elapsed;
    if r >= 1e9      { format!("{:6.2}G", r / 1e9) }
    else if r >= 1e6 { format!("{:6.1}M", r / 1e6) }
    else if r >= 1e3 { format!("{:6.0}K", r / 1e3) }
    else             { format!("{:6.0} ", r)        }
}

// ---------------------------------------------------------------------------
// SharedCell — lets a single writer and N readers share a Cell without a lock.
// Safety is provided by Cell's atomic head field (Release/Acquire ordering).
// ---------------------------------------------------------------------------
struct SharedCell<T>(UnsafeCell<Cell<T>>);
unsafe impl<T: Send> Send for SharedCell<T> {}
unsafe impl<T: Send> Sync for SharedCell<T> {}
impl<T> SharedCell<T> {
    fn new() -> Self { SharedCell(UnsafeCell::new(Cell::new())) }
    fn reader(&self) -> &Cell<T>           { unsafe { &*self.0.get() } }
    unsafe fn writer(&self) -> &mut Cell<T> { &mut *self.0.get() }
}

// ---------------------------------------------------------------------------
// Stats collected by each benchmark run
// ---------------------------------------------------------------------------
struct Stats { reads: u64, writes: u64, errors: u64, elapsed: f64 }

// ---------------------------------------------------------------------------
// 1. Single-thread baseline
// ---------------------------------------------------------------------------
fn bench_single() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  1. Single-thread baseline (2s each)                        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    macro_rules! row {
        ($label:expr, $ops:expr, $elapsed:expr, $baseline:expr) => {{
            let r = $ops as f64 / $elapsed;
            let pct = r / $baseline * 100.0;
            println!("║  {:<36}  {:>8} ops/s  {:>5.0}%  ║",
                $label, fmt_rate($ops, $elapsed), pct);
            r
        }};
    }

    // Plain u32 — no atomics, absolute floor
    let baseline = {
        let mut v: u32 = 0;
        let mut sink = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { v = v.wrapping_add(1); sink ^= v as u64; }
        let _ = sink;
        let e = t.elapsed().as_secs_f64();
        let r = v as f64 / e;
        println!("║  {:<36}  {:>8} ops/s  {:>5}  ║",
            "plain u32  (no atomics — baseline)", fmt_rate(v as u64, e), "100%");
        r
    };

    // Cell<u32> write — 1 Release store, value+epoch in one word
    {
        let mut cell: Cell<u32> = Cell::new();
        let mut i = 0u32;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { unsafe { cell.write(i) }; i = i.wrapping_add(1); }
        row!("Cell<u32>  write  (inline, 1 Release)", i as u64, t.elapsed().as_secs_f64(), baseline);
    }

    // Cell<u32> read — 1 Acquire load
    {
        let mut cell: Cell<u32> = Cell::new();
        unsafe { cell.write(42u32) };
        let mut ops = 0u64;
        let mut last = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            if let ReadResult::Value { epoch, .. } = cell.read(last) { last = epoch; }
            ops += 1;
        }
        row!("Cell<u32>  read   (inline, 1 Acquire)", ops, t.elapsed().as_secs_f64(), baseline);
    }

    // SeqCell<u32> write — seqlock on a small type (3 atomic ops vs Cell's 1)
    {
        let cell = SeqCell::<u32>::new();
        let mut i = 0u32;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { unsafe { cell.write(i) }; i = i.wrapping_add(1); }
        row!("SeqCell<u32> write (seqlock, small T)", i as u64, t.elapsed().as_secs_f64(), baseline);
    }

    // SeqCell<u32> read
    {
        let cell = SeqCell::<u32>::new();
        unsafe { cell.write(42u32) };
        let mut ops = 0u64;
        let mut last = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            if let ReadResult::Value { epoch, .. } = cell.read(last) { last = epoch; }
            ops += 1;
        }
        row!("SeqCell<u32> read  (seqlock, small T)", ops, t.elapsed().as_secs_f64(), baseline);
    }

    // Cell<u64> write — Block path: heap alloc per write
    {
        let mut cell: Cell<u64> = Cell::new();
        let mut i = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { unsafe { cell.write(i) }; i += 1; }
        row!("Cell<u64>  write  (Block, alloc/write)", i, t.elapsed().as_secs_f64(), baseline);
    }

    // Cell<u64> read — Block path: Acquire load + pointer deref
    {
        let mut cell: Cell<u64> = Cell::new();
        unsafe { cell.write(42u64) };
        let mut ops = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { let _ = cell.read(0); ops += 1; }
        row!("Cell<u64>  read   (Block, Acq+deref)", ops, t.elapsed().as_secs_f64(), baseline);
    }

    // SeqCell<u64> write — seqlock, inline storage, zero allocation
    {
        let cell = SeqCell::<u64>::new();
        let mut i = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { unsafe { cell.write(i) }; i += 1; }
        row!("SeqCell<u64> write (seqlock, no alloc)", i, t.elapsed().as_secs_f64(), baseline);
    }

    // SeqCell<u64> read — seqlock, bracketed copy
    {
        let cell = SeqCell::<u64>::new();
        unsafe { cell.write(42u64) };
        let mut ops = 0u64;
        let mut last = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            if let ReadResult::Value { epoch, .. } = cell.read(last) { last = epoch; }
            ops += 1;
        }
        row!("SeqCell<u64> read  (seqlock, no alloc)", ops, t.elapsed().as_secs_f64(), baseline);
    }

    println!("║  {:<50}  ║", "");

    // Mutex<u32> write / read
    {
        let mu = Mutex::new(0u32);
        let mut i = 0u32;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { *mu.lock().unwrap() = i; i = i.wrapping_add(1); }
        row!("Mutex<u32> write", i as u64, t.elapsed().as_secs_f64(), baseline);
    }
    {
        let mu = Mutex::new(42u32);
        let mut ops = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { let _ = *mu.lock().unwrap(); ops += 1; }
        row!("Mutex<u32> read", ops, t.elapsed().as_secs_f64(), baseline);
    }

    // RwLock<u32> write / read
    {
        let rw = RwLock::new(0u32);
        let mut i = 0u32;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { *rw.write().unwrap() = i; i = i.wrapping_add(1); }
        row!("RwLock<u32> write", i as u64, t.elapsed().as_secs_f64(), baseline);
    }
    {
        let rw = RwLock::new(42u32);
        let mut ops = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { let _ = *rw.read().unwrap(); ops += 1; }
        row!("RwLock<u32> read", ops, t.elapsed().as_secs_f64(), baseline);
    }

    println!("╚══════════════════════════════════════════════════════════════╝");
}

// ---------------------------------------------------------------------------
// Multi-thread benchmark runners
// ---------------------------------------------------------------------------

// Scenario A: writer sleeps 1 ms between writes, readers flat-out.
// Tests reader throughput when writes are rare (sensor / config update pattern).
fn run_sparse(n_readers: usize) -> (Stats, Stats, Stats, Stats)
{
    fn go_seqcell(n: usize) -> Stats {
        let cell  = Arc::new(SeqCell::<u64>::new());
        let done  = Arc::new(AtomicBool::new(false));
        let wc    = Arc::new(AtomicU64::new(0));
        let rc    = Arc::new(AtomicU64::new(0));
        let (cw, dw, wcw) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !dw.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(1));
                unsafe { cw.write(i) };
                i += 1; wcw.fetch_add(1, Ordering::Relaxed);
            }
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (cr, dr, rcr) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut last = 0u64; let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    if let ReadResult::Value { epoch, .. } = cr.read(last) { last = epoch; }
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_cell(n: usize) -> Stats {
        let cell  = Arc::new(SharedCell::<u32>::new());
        let done  = Arc::new(AtomicBool::new(false));
        let wc    = Arc::new(AtomicU64::new(0));
        let rc    = Arc::new(AtomicU64::new(0));

        let (cw, dw, wcw) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u32;
            while !dw.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(1));
                unsafe { cw.writer().write(i) };
                i = i.wrapping_add(1);
                wcw.fetch_add(1, Ordering::Relaxed);
            }
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (cr, dr, rcr) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut last = 0u64; let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    if let ReadResult::Value { epoch, .. } = cr.reader().read(last) {
                        last = epoch;
                    }
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_mutex(n: usize) -> Stats {
        let mu   = Arc::new(Mutex::new(0u32));
        let done = Arc::new(AtomicBool::new(false));
        let wc   = Arc::new(AtomicU64::new(0));
        let rc   = Arc::new(AtomicU64::new(0));

        let (mw, dw, wcw) = (Arc::clone(&mu), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u32;
            while !dw.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(1));
                *mw.lock().unwrap() = i;
                i = i.wrapping_add(1);
                wcw.fetch_add(1, Ordering::Relaxed);
            }
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (mr, dr, rcr) = (Arc::clone(&mu), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    let _ = *mr.lock().unwrap();
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_rwlock(n: usize) -> Stats {
        let rw   = Arc::new(RwLock::new(0u32));
        let done = Arc::new(AtomicBool::new(false));
        let wc   = Arc::new(AtomicU64::new(0));
        let rc   = Arc::new(AtomicU64::new(0));

        let (rw_w, dw, wcw) = (Arc::clone(&rw), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u32;
            while !dw.load(Ordering::Acquire) {
                thread::sleep(Duration::from_millis(1));
                *rw_w.write().unwrap() = i;
                i = i.wrapping_add(1);
                wcw.fetch_add(1, Ordering::Relaxed);
            }
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (rw_r, dr, rcr) = (Arc::clone(&rw), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    let _ = *rw_r.read().unwrap();
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    (go_cell(n_readers), go_seqcell(n_readers), go_mutex(n_readers), go_rwlock(n_readers))
}

// Scenario B: writer flat-out, readers sleep 100 µs between reads.
// Tests write throughput when the writer is never waiting on readers.
fn run_heavy_write(n_readers: usize) -> (Stats, Stats, Stats, Stats) {

    fn go_seqcell(n: usize) -> Stats {
        let cell  = Arc::new(SeqCell::<u64>::new());
        let done  = Arc::new(AtomicBool::new(false));
        let wc    = Arc::new(AtomicU64::new(0));
        let rc    = Arc::new(AtomicU64::new(0));
        let (cw, dw, wcw) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64; let mut cnt = 0u64;
            while !dw.load(Ordering::Acquire) { unsafe { cw.write(i) }; i += 1; cnt += 1; }
            wcw.store(cnt, Ordering::Relaxed);
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (cr, dr, rcr) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut last = 0u64; let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_micros(100));
                    if let ReadResult::Value { epoch, .. } = cr.read(last) { last = epoch; }
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_cell(n: usize) -> Stats {
        let cell  = Arc::new(SharedCell::<u32>::new());
        let done  = Arc::new(AtomicBool::new(false));
        let wc    = Arc::new(AtomicU64::new(0));
        let rc    = Arc::new(AtomicU64::new(0));

        let (cw, dw, wcw) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u32; let mut cnt = 0u64;
            while !dw.load(Ordering::Acquire) {
                unsafe { cw.writer().write(i) };
                i = i.wrapping_add(1); cnt += 1;
            }
            wcw.store(cnt, Ordering::Relaxed);
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (cr, dr, rcr) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut last = 0u64; let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_micros(100));
                    if let ReadResult::Value { epoch, .. } = cr.reader().read(last) {
                        last = epoch;
                    }
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_mutex(n: usize) -> Stats {
        let mu   = Arc::new(Mutex::new(0u32));
        let done = Arc::new(AtomicBool::new(false));
        let wc   = Arc::new(AtomicU64::new(0));
        let rc   = Arc::new(AtomicU64::new(0));

        let (mw, dw, wcw) = (Arc::clone(&mu), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u32; let mut cnt = 0u64;
            while !dw.load(Ordering::Acquire) {
                *mw.lock().unwrap() = i;
                i = i.wrapping_add(1); cnt += 1;
            }
            wcw.store(cnt, Ordering::Relaxed);
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (mr, dr, rcr) = (Arc::clone(&mu), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_micros(100));
                    let _ = *mr.lock().unwrap();
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_rwlock(n: usize) -> Stats {
        let rw   = Arc::new(RwLock::new(0u32));
        let done = Arc::new(AtomicBool::new(false));
        let wc   = Arc::new(AtomicU64::new(0));
        let rc   = Arc::new(AtomicU64::new(0));

        let (rw_w, dw, wcw) = (Arc::clone(&rw), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u32; let mut cnt = 0u64;
            while !dw.load(Ordering::Acquire) {
                *rw_w.write().unwrap() = i;
                i = i.wrapping_add(1); cnt += 1;
            }
            wcw.store(cnt, Ordering::Relaxed);
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (rw_r, dr, rcr) = (Arc::clone(&rw), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    thread::sleep(Duration::from_micros(100));
                    let _ = *rw_r.read().unwrap();
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    (go_cell(n_readers), go_seqcell(n_readers), go_mutex(n_readers), go_rwlock(n_readers))
}

// Scenario C: writer + all readers flat-out.
// Epochs must never go backward — any inversion is a correctness failure.
fn run_stress(n_readers: usize) -> (Stats, Stats, Stats, Stats) {

    fn go_seqcell(n: usize) -> Stats {
        let cell   = Arc::new(SeqCell::<u64>::new());
        let done   = Arc::new(AtomicBool::new(false));
        let wc     = Arc::new(AtomicU64::new(0));
        let rc     = Arc::new(AtomicU64::new(0));
        let errors = Arc::new(AtomicU64::new(0));
        let (cw, dw, wcw) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64; let mut cnt = 0u64;
            while !dw.load(Ordering::Acquire) { unsafe { cw.write(i) }; i += 1; cnt += 1; }
            wcw.store(cnt, Ordering::Relaxed);
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (cr, dr, rcr, er) = (Arc::clone(&cell), Arc::clone(&done),
                                      Arc::clone(&rc), Arc::clone(&errors));
            readers.push(thread::spawn(move || {
                let mut last = 0u64; let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    match cr.read(last) {
                        ReadResult::Value { epoch, .. } => {
                            if epoch < last { er.fetch_add(1, Ordering::Relaxed); }
                            last = epoch;
                        }
                        ReadResult::Empty => {}
                    }
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: errors.load(Ordering::Relaxed), elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_cell(n: usize) -> Stats {
        let cell   = Arc::new(SharedCell::<u32>::new());
        let done   = Arc::new(AtomicBool::new(false));
        let wc     = Arc::new(AtomicU64::new(0));
        let rc     = Arc::new(AtomicU64::new(0));
        let errors = Arc::new(AtomicU64::new(0));

        let (cw, dw, wcw) = (Arc::clone(&cell), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u32; let mut cnt = 0u64;
            while !dw.load(Ordering::Acquire) {
                unsafe { cw.writer().write(i) };
                i = i.wrapping_add(1); cnt += 1;
            }
            wcw.store(cnt, Ordering::Relaxed);
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (cr, dr, rcr, er) = (Arc::clone(&cell), Arc::clone(&done),
                                      Arc::clone(&rc), Arc::clone(&errors));
            readers.push(thread::spawn(move || {
                let mut last = 0u64; let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    match cr.reader().read(last) {
                        ReadResult::Value { epoch, .. } => {
                            if epoch < last { er.fetch_add(1, Ordering::Relaxed); }
                            last = epoch;
                        }
                        ReadResult::Empty => {}
                    }
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: errors.load(Ordering::Relaxed), elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_mutex(n: usize) -> Stats {
        let mu   = Arc::new(Mutex::new(0u32));
        let done = Arc::new(AtomicBool::new(false));
        let wc   = Arc::new(AtomicU64::new(0));
        let rc   = Arc::new(AtomicU64::new(0));

        let (mw, dw, wcw) = (Arc::clone(&mu), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u32; let mut cnt = 0u64;
            while !dw.load(Ordering::Acquire) {
                *mw.lock().unwrap() = i;
                i = i.wrapping_add(1); cnt += 1;
            }
            wcw.store(cnt, Ordering::Relaxed);
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (mr, dr, rcr) = (Arc::clone(&mu), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    let _ = *mr.lock().unwrap();
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_rwlock(n: usize) -> Stats {
        let rw   = Arc::new(RwLock::new(0u32));
        let done = Arc::new(AtomicBool::new(false));
        let wc   = Arc::new(AtomicU64::new(0));
        let rc   = Arc::new(AtomicU64::new(0));

        let (rw_w, dw, wcw) = (Arc::clone(&rw), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u32; let mut cnt = 0u64;
            while !dw.load(Ordering::Acquire) {
                *rw_w.write().unwrap() = i;
                i = i.wrapping_add(1); cnt += 1;
            }
            wcw.store(cnt, Ordering::Relaxed);
        });
        let mut readers = vec![];
        for _ in 0..n {
            let (rw_r, dr, rcr) = (Arc::clone(&rw), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    let _ = *rw_r.read().unwrap();
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }
        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    (go_cell(n_readers), go_seqcell(n_readers), go_mutex(n_readers), go_rwlock(n_readers))
}

// ---------------------------------------------------------------------------
// 2. Sparse-writes table
// ---------------------------------------------------------------------------
fn bench_sparse() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  2. Sparse writes — writer sleeps 1 ms, N readers flat-out                          ║");
    println!("║  Metric: total reads/s across all reader threads                                    ║");
    println!("║  Cell<u32>=inline  SeqCell<u64>=seqlock/no-alloc  Mutex/RwLock=blocking             ║");
    println!("╠═══════════╦══════════════╦════════════════╦══════════════╦══════════════╦═══════════╣");
    println!("║  readers  ║  Cell<u32>   ║  SeqCell<u64>  ║  Mutex       ║  RwLock      ║ Seq/Mutex ║");
    println!("╠═══════════╬══════════════╬════════════════╬══════════════╬══════════════╬═══════════╣");

    for &n in &[3usize, 5, 10, 16] {
        let (c, s, m, r) = run_sparse(n);
        let speedup = (s.reads as f64 / s.elapsed) / (m.reads as f64 / m.elapsed);
        println!("║  {:>7}  ║  {:>10}  ║  {:>12}  ║  {:>10}  ║  {:>10}  ║  {:>6.1}x  ║",
            n,
            fmt_rate(c.reads, c.elapsed),
            fmt_rate(s.reads, s.elapsed),
            fmt_rate(m.reads, m.elapsed),
            fmt_rate(r.reads, r.elapsed),
            speedup);
    }
    println!("╚═══════════╩══════════════╩════════════════╩══════════════╩══════════════╩═══════════╝");
}

// ---------------------------------------------------------------------------
// 3. Heavy-writes table
// ---------------------------------------------------------------------------
fn bench_heavy() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  3. Heavy writes — writer flat-out, readers sleep 100 µs                            ║");
    println!("║  Metric: writer ops/s (readers add contention but read infrequently)                ║");
    println!("╠═══════════╦══════════════╦════════════════╦══════════════╦══════════════╦═══════════╣");
    println!("║  readers  ║  Cell<u32>   ║  SeqCell<u64>  ║  Mutex       ║  RwLock      ║ Seq/Mutex ║");
    println!("╠═══════════╬══════════════╬════════════════╬══════════════╬══════════════╬═══════════╣");

    for &n in &[3usize, 5, 10, 16] {
        let (c, s, m, r) = run_heavy_write(n);
        let speedup = (s.writes as f64 / s.elapsed) / (m.writes as f64 / m.elapsed);
        println!("║  {:>7}  ║  {:>10}  ║  {:>12}  ║  {:>10}  ║  {:>10}  ║  {:>6.1}x  ║",
            n,
            fmt_rate(c.writes, c.elapsed),
            fmt_rate(s.writes, s.elapsed),
            fmt_rate(m.writes, m.elapsed),
            fmt_rate(r.writes, r.elapsed),
            speedup);
    }
    println!("╚═══════════╩══════════════╩════════════════╩══════════════╩══════════════╩═══════════╝");
}

// ---------------------------------------------------------------------------
// 4. Full-stress table
// ---------------------------------------------------------------------------
fn bench_stress() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  4. Full stress — writer + all readers flat-out                                                     ║");
    println!("║  errs = epoch inversions detected (must be 0 — correctness check)                                  ║");
    println!("╠═══════════╦═════════════╦══════╦═════════════╦═════════════╦══════╦═════════════╦═════════════╦════╣");
    println!("║  readers  ║  Cell<u32>  ║ errs ║  Cell w/s   ║ SeqCell<64> ║ errs ║  Seq w/s    ║  Mutex r/s  ║ Rx ║");
    println!("╠═══════════╬═════════════╬══════╬═════════════╬═════════════╬══════╬═════════════╬═════════════╬════╣");

    for &n in &[3usize, 5, 10, 16] {
        let (c, s, m, _r) = run_stress(n);
        let speedup = (s.reads as f64 / s.elapsed) / (m.reads as f64 / m.elapsed);
        println!("║  {:>7}  ║  {:>9}  ║ {:>4} ║  {:>9}  ║  {:>9}  ║ {:>4} ║  {:>9}  ║  {:>9}  ║{:>3.0}x║",
            n,
            fmt_rate(c.reads,  c.elapsed), c.errors,
            fmt_rate(c.writes, c.elapsed),
            fmt_rate(s.reads,  s.elapsed), s.errors,
            fmt_rate(s.writes, s.elapsed),
            fmt_rate(m.reads,  m.elapsed),
            speedup);
    }
    println!("╚═══════════╩═════════════╩══════╩═════════════╩═════════════╩══════╩═════════════╩═════════════╩════╝");
    println!("  Rx = SeqCell reads / Mutex reads speedup. Errors must be 0 for both Cell and SeqCell.");
}

// ---------------------------------------------------------------------------
// 5. Read cost by sizeof(T): Cell<T> (SIMD clone) vs SeqCell (volatile scalar)
//
// Answers: why do SeqCell reads get expensive for large T, and how does it
// compare to normal Rust thread-safe reads?
//
// Cell<T> read  = one Acquire load + pointer deref + Clone (compiler uses SIMD)
// SeqCell read  = two Acquire loads + ptr::read_volatile (scalar, no SIMD)
// Mutex<T> read = futex CAS (uncontended) + compiler-SIMD copy of T
// ---------------------------------------------------------------------------
fn bench_seqcell_sizes() {
    println!();
    println!("╔═══════════════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  5. Read cost by sizeof(T) — SIMD clone (Cell/Mutex) vs volatile scalar (SeqCell)       ║");
    println!("║  Single thread, no write contention. % relative to Cell<T> read (100% = fastest).       ║");
    println!("╠════════╦════════════════╦════════════════╦════════════════╦════════════════╦═════════════╣");
    println!("║  size  ║  Cell<T> read  ║  SeqCell read  ║  SeqCell unprot║  Mutex<T> read ║  Cell/Seq   ║");
    println!("║        ║  SIMD clone    ║  seqlock+volat ║  volatile only ║  lock+SIMD cpy ║  speedup    ║");
    println!("╠════════╬════════════════╬════════════════╬════════════════╬════════════════╬═════════════╣");

    macro_rules! row {
        ($sz:literal) => {{
            type T = [u8; $sz];

            // Cell<T> read — inline path for ≤4B, Block path for >4B.
            // Block path: Acquire load of pointer + deref + Clone (SIMD, no volatile).
            let mut cell = Cell::<T>::new();
            unsafe { cell.write([42u8; $sz]) };
            let mut c_ops = 0u64;
            let mut c_sink = 0u64;
            let t = Instant::now();
            while t.elapsed() < bench_dur() {
                if let ReadResult::Value { value, .. } = cell.read(0) {
                    c_sink ^= value[0] as u64;
                }
                c_ops += 1;
            }
            let _ = c_sink;
            let ce = t.elapsed().as_secs_f64();

            // SeqCell read (protected) — 2 Acquire loads + ptr::read_volatile (scalar)
            let cell_s = SeqCell::<T>::new();
            unsafe { cell_s.write([42u8; $sz]) };
            let mut r_ops = 0u64;
            let mut r_sink = 0u64;
            let t = Instant::now();
            while t.elapsed() < bench_dur() {
                if let ReadResult::Value { value, .. } = cell_s.read(0) {
                    r_sink ^= value[0] as u64;
                }
                r_ops += 1;
            }
            let _ = r_sink;
            let re = t.elapsed().as_secs_f64();

            // SeqCell read (unprotected) — 1 Acquire load + ptr::read_volatile (scalar)
            let cell_u = SeqCell::<T>::new();
            unsafe { cell_u.write([42u8; $sz]) };
            let mut u_ops = 0u64;
            let mut u_sink = 0u64;
            let t = Instant::now();
            while t.elapsed() < bench_dur() {
                if let ReadResult::Value { value, .. } = cell_u.read_unprotected() {
                    u_sink ^= value[0] as u64;
                }
                u_ops += 1;
            }
            let _ = u_sink;
            let ue = t.elapsed().as_secs_f64();

            // Mutex<T> read — uncontended futex CAS + compiler-SIMD copy out of lock
            let mu = Mutex::new([42u8; $sz]);
            let mut m_ops = 0u64;
            let mut m_sink = 0u64;
            let t = Instant::now();
            while t.elapsed() < bench_dur() {
                let g = mu.lock().unwrap();
                let v: T = *g;  // SIMD copy (no volatile constraint)
                m_sink ^= v[0] as u64;
                m_ops += 1;
            }
            let _ = m_sink;
            let me = t.elapsed().as_secs_f64();

            let c_rate = c_ops as f64 / ce;
            let r_rate = r_ops as f64 / re;
            let u_rate = u_ops as f64 / ue;
            let m_rate = m_ops as f64 / me;

            println!(
                "║ {:>5}B ║ {:>8}  100%  ║ {:>8}  {:>3.0}%  ║ {:>8}  {:>3.0}%  ║ {:>8}  {:>3.0}%  ║  {:>6.1}x   ║",
                $sz,
                fmt_rate(c_ops, ce),
                fmt_rate(r_ops, re), r_rate / c_rate * 100.0,
                fmt_rate(u_ops, ue), u_rate / c_rate * 100.0,
                fmt_rate(m_ops, me), m_rate / c_rate * 100.0,
                c_rate / r_rate,
            );
        }};
    }

    row!(4);
    row!(8);
    row!(16);
    row!(32);
    row!(64);
    row!(128);
    row!(256);

    println!("╚════════╩════════════════╩════════════════╩════════════════╩════════════════╩═════════════╝");
    println!("  Cell<T> and Mutex both use regular Clone/memcpy — the compiler picks SIMD (AVX2/SSE).");
    println!("  SeqCell uses ptr::read_volatile — the compiler must emit scalar loads, no SIMD.");
    println!("  Seqlock bracket (protected vs unprotected) adds only 1 extra Acquire load on top.");
    println!("  Mutex lock is a single uncontended CAS here; multi-thread contention would hurt it.");
}

// ===========================================================================
// BRIDGE-LAYER BENCHMARKS — BridgedCell vs std::sync::{Mutex, RwLock}
// ===========================================================================
//
// What the bridge buys you, measured against the natural Rust built-ins:
//
//   single-thread cost-per-op   — overhead of write/read/read_ref/reclaim
//   concurrent writer + readers — does read_ref scale where Mutex/RwLock don't?
//   large payload (Vec<u64>)    — zero-copy read_ref vs RwLock guard while a
//                                 writer is hot (this is the killer scenario)
//
// SharedBridgedCell mirrors the existing SharedCell pattern: one writer
// thread owns &mut access (also runs reclaim), N reader threads share &
// access for read_ref. Readers never lock; the writer never blocks on
// readers. This is the deployment pattern the bridge layer is designed for.
// ===========================================================================

struct SharedBridgedCell<T>(UnsafeCell<BridgedCell<T>>);
unsafe impl<T: Send> Send for SharedBridgedCell<T> {}
unsafe impl<T: Send> Sync for SharedBridgedCell<T> {}
impl<T> SharedBridgedCell<T> {
    fn new() -> Self { SharedBridgedCell(UnsafeCell::new(BridgedCell::new())) }
    /// Read-side view. Safe to share across reader threads — read_ref only
    /// touches the head atomic and the caller's own ReaderRegistry slot;
    /// it never touches the retired list.
    fn reader(&self) -> &BridgedCell<T> { unsafe { &*self.0.get() } }
    /// Write-side / reclaim-side view. SAFETY: caller must guarantee a
    /// single thread holds this mutable view at a time (mutates the
    /// retired list non-atomically).
    unsafe fn writer(&self) -> &mut BridgedCell<T> { unsafe { &mut *self.0.get() } }
}

// ---------------------------------------------------------------------------
// 6. Bridge — single-thread cost-per-op
// ---------------------------------------------------------------------------
fn bench_bridge_single() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  6. Bridge layer single-thread cost-per-op (2s each)        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    macro_rules! row {
        ($label:expr, $ops:expr, $elapsed:expr, $baseline:expr) => {{
            let r = $ops as f64 / $elapsed;
            let pct = r / $baseline * 100.0;
            println!("║  {:<36}  {:>8} ops/s  {:>5.0}%  ║",
                $label, fmt_rate($ops, $elapsed), pct);
            r
        }};
    }

    // Baseline: plain u64 (no atomics)
    let baseline = {
        let mut v: u64 = 0;
        let mut sink = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { v = v.wrapping_add(1); sink ^= v; }
        let _ = sink;
        let e = t.elapsed().as_secs_f64();
        let r = v as f64 / e;
        println!("║  {:<36}  {:>8} ops/s  {:>5}  ║",
            "plain u64  (no atomics — baseline)", fmt_rate(v, e), "100%");
        r
    };

    // ── Writes ────────────────────────────────────────────────────────────
    {
        let mu: Mutex<u64> = Mutex::new(0);
        let mut i = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { *mu.lock().unwrap() = i; i += 1; }
        row!("Mutex<u64>        write", i, t.elapsed().as_secs_f64(), baseline);
    }
    {
        let lk: RwLock<u64> = RwLock::new(0);
        let mut i = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { *lk.write().unwrap() = i; i += 1; }
        row!("RwLock<u64>       write", i, t.elapsed().as_secs_f64(), baseline);
    }
    {
        let mut bc: BridgedCell<u64> = BridgedCell::new();
        let registry = ReaderRegistry::new();
        let mut i = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            unsafe { bc.write(i); }
            i += 1;
            // Opportunistic gated sweep — only fires when retired list ≥ WATERMARK.
            if i & 1023 == 0 { bc.reclaim_if_watermark(&registry); }
        }
        row!("BridgedCell<u64>  write+watermark", i, t.elapsed().as_secs_f64(), baseline);
    }

    // ── Reads ─────────────────────────────────────────────────────────────
    {
        let mu: Mutex<u64> = Mutex::new(42);
        let mut ops = 0u64; let mut sink = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { sink ^= *mu.lock().unwrap(); ops += 1; }
        let _ = sink;
        row!("Mutex<u64>        read", ops, t.elapsed().as_secs_f64(), baseline);
    }
    {
        let lk: RwLock<u64> = RwLock::new(42);
        let mut ops = 0u64; let mut sink = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() { sink ^= *lk.read().unwrap(); ops += 1; }
        let _ = sink;
        row!("RwLock<u64>       read", ops, t.elapsed().as_secs_f64(), baseline);
    }
    {
        let mut bc: BridgedCell<u64> = BridgedCell::new();
        unsafe { bc.write(42u64); }
        let mut ops = 0u64; let mut last = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            if let ReadResult::Value { epoch, .. } = bc.read(last) { last = epoch; }
            ops += 1;
        }
        row!("BridgedCell<u64>  read (materialise)", ops, t.elapsed().as_secs_f64(), baseline);
    }
    {
        let mut bc: BridgedCell<u64> = BridgedCell::new();
        let registry = ReaderRegistry::new();
        let handle = registry.acquire();
        unsafe { bc.write(42u64); }
        let mut ops = 0u64; let mut last = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            if let Some(r) = bc.read_ref(&handle, last) {
                last = r.epoch;
                // r drops at end of statement — that's the floor release.
            }
            ops += 1;
        }
        row!("BridgedCell<u64>  read_ref (pin+drop)", ops, t.elapsed().as_secs_f64(), baseline);
    }

    // ── Reclaim sweep — amortised per-block ─────────────────────────────
    {
        let registry = ReaderRegistry::new();
        let mut bc: BridgedCell<u64> = BridgedCell::new();
        let mut freed = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            // Fill a batch of retirements, then sweep them.
            for _ in 0..256u64 { unsafe { bc.write(0); } }
            freed += bc.reclaim(&registry) as u64;
        }
        row!("BridgedCell<u64>  reclaim (per block)", freed, t.elapsed().as_secs_f64(), baseline);
    }

    println!("╚══════════════════════════════════════════════════════════════╝");
}

// ---------------------------------------------------------------------------
// 7. Bridge — concurrent writer + N readers (small T: u64)
// ---------------------------------------------------------------------------
//
// One writer thread hammers writes. N reader threads hammer reads. We
// measure writer-writes/s and reader-reads/s separately so we can see
// whether the scheme serialises one against the other.
//
// BridgedCell uses SharedBridgedCell (lock-free reads; writer owns &mut
// for write+reclaim). Mutex/RwLock are the natural Rust built-ins.
// ---------------------------------------------------------------------------
fn run_bridge_scaling_u64(n: usize) -> (Stats, Stats, Stats) {
    fn go_bridge(n: usize) -> Stats {
        let cell     = Arc::new(SharedBridgedCell::<u64>::new());
        let registry = Arc::new(ReaderRegistry::new());
        let done     = Arc::new(AtomicBool::new(false));
        let wc       = Arc::new(AtomicU64::new(0));
        let rc       = Arc::new(AtomicU64::new(0));

        unsafe { cell.writer().write(0u64); }

        let (cw, rw, dw, wcw) = (Arc::clone(&cell), Arc::clone(&registry),
                                 Arc::clone(&done),  Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !dw.load(Ordering::Acquire) {
                unsafe { cw.writer().write(i); }
                i += 1;
                if i & 511 == 0 {
                    unsafe { cw.writer().reclaim_if_watermark(&rw); }
                }
            }
            wcw.fetch_add(i, Ordering::Relaxed);
            // Final drain so the cell drops cleanly.
            unsafe { cw.writer().reclaim(&rw); }
        });

        let mut readers = vec![];
        for _ in 0..n {
            let (cr, rr, dr, rcr) = (Arc::clone(&cell), Arc::clone(&registry),
                                     Arc::clone(&done),  Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let handle = rr.acquire();
                let mut last = 0u64; let mut cnt = 0u64;
                while !dr.load(Ordering::Acquire) {
                    if let Some(r) = cr.reader().read_ref(&handle, last) {
                        last = r.epoch;
                    }
                    cnt += 1;
                }
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }

        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_rwlock(n: usize) -> Stats {
        let lk   = Arc::new(RwLock::new(0u64));
        let done = Arc::new(AtomicBool::new(false));
        let wc   = Arc::new(AtomicU64::new(0));
        let rc   = Arc::new(AtomicU64::new(0));

        let (lw, dw, wcw) = (Arc::clone(&lk), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !dw.load(Ordering::Acquire) {
                *lw.write().unwrap() = i;
                i += 1;
            }
            wcw.fetch_add(i, Ordering::Relaxed);
        });

        let mut readers = vec![];
        for _ in 0..n {
            let (lr, dr, rcr) = (Arc::clone(&lk), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut cnt = 0u64; let mut sink = 0u64;
                while !dr.load(Ordering::Acquire) {
                    sink ^= *lr.read().unwrap();
                    cnt += 1;
                }
                let _ = sink;
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }

        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_mutex(n: usize) -> Stats {
        let mu   = Arc::new(Mutex::new(0u64));
        let done = Arc::new(AtomicBool::new(false));
        let wc   = Arc::new(AtomicU64::new(0));
        let rc   = Arc::new(AtomicU64::new(0));

        let (mw, dw, wcw) = (Arc::clone(&mu), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !dw.load(Ordering::Acquire) {
                *mw.lock().unwrap() = i;
                i += 1;
            }
            wcw.fetch_add(i, Ordering::Relaxed);
        });

        let mut readers = vec![];
        for _ in 0..n {
            let (mr, dr, rcr) = (Arc::clone(&mu), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut cnt = 0u64; let mut sink = 0u64;
                while !dr.load(Ordering::Acquire) {
                    sink ^= *mr.lock().unwrap();
                    cnt += 1;
                }
                let _ = sink;
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }

        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    (go_bridge(n), go_rwlock(n), go_mutex(n))
}

fn bench_bridge_concurrent_small() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  7. Bridge — concurrent writer + N readers (T = u64, small)                     ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  readers │   BridgedCell read_ref   │      RwLock<u64>         │      Mutex<u64>          ║");
    println!("║          │   writes/s │   reads/s   │   writes/s │   reads/s   │   writes/s │   reads/s   ║");
    println!("╠══════════╪════════════╪═════════════╪════════════╪═════════════╪════════════╪═════════════╣");

    for &n in &[1usize, 2, 4, 8] {
        let (bc, rw, mu) = run_bridge_scaling_u64(n);
        let rate = |ops: u64, e: f64| fmt_rate(ops, e);
        println!("║   {:>2}     │ {:>10} │ {:>11} │ {:>10} │ {:>11} │ {:>10} │ {:>11} ║",
            n,
            rate(bc.writes, bc.elapsed), rate(bc.reads, bc.elapsed),
            rate(rw.writes, rw.elapsed), rate(rw.reads, rw.elapsed),
            rate(mu.writes, mu.elapsed), rate(mu.reads, mu.elapsed));
    }
    println!("╚══════════╧════════════╧═════════════╧════════════╧═════════════╧════════════╧═════════════╝");
    println!("  BridgedCell: readers never lock (atomic head load + slot store).");
    println!("                writer never waits on readers — old block goes to retired list.");
    println!("  RwLock:      readers concurrent; writer waits for all readers to release.");
    println!("  Mutex:       everything serialised; reads and writes contend on the same lock.");
}

// ---------------------------------------------------------------------------
// 8. Bridge — large payload (Vec<u64>, ~1 KB): the zero-copy story
// ---------------------------------------------------------------------------
//
// Here BridgedCell's read_ref returns a raw pointer into the heap Block
// — zero copy regardless of payload size. RwLock's read guard borrows the
// data, also zero-copy, BUT it blocks the writer for the guard's lifetime.
// Mutex<Vec<u64>> would have to lock for every read, blocking the writer.
//
// We give every reader some "work" (an iterate-and-sum loop) so the read
// holds the guard / ReadRef for a non-trivial time — that is the realistic
// case where the lock-vs-no-lock difference shows up.
// ---------------------------------------------------------------------------

const PAYLOAD_LEN: usize = 128;        // ~1 KB

fn make_payload(seed: u64) -> Vec<u64> {
    (0..PAYLOAD_LEN as u64).map(|i| seed.wrapping_add(i)).collect()
}

fn run_bridge_scaling_big(n: usize) -> (Stats, Stats, Stats) {
    fn go_bridge(n: usize) -> Stats {
        let cell     = Arc::new(SharedBridgedCell::<Vec<u64>>::new());
        let registry = Arc::new(ReaderRegistry::new());
        let done     = Arc::new(AtomicBool::new(false));
        let wc       = Arc::new(AtomicU64::new(0));
        let rc       = Arc::new(AtomicU64::new(0));

        unsafe { cell.writer().write(make_payload(0)); }

        let (cw, rw, dw, wcw) = (Arc::clone(&cell), Arc::clone(&registry),
                                 Arc::clone(&done),  Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !dw.load(Ordering::Acquire) {
                unsafe { cw.writer().write(make_payload(i)); }
                i += 1;
                if i & 255 == 0 {
                    unsafe { cw.writer().reclaim_if_watermark(&rw); }
                }
            }
            wcw.fetch_add(i, Ordering::Relaxed);
            unsafe { cw.writer().reclaim(&rw); }
        });

        let mut readers = vec![];
        for _ in 0..n {
            let (cr, rr, dr, rcr) = (Arc::clone(&cell), Arc::clone(&registry),
                                     Arc::clone(&done),  Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let handle = rr.acquire();
                let mut last = 0u64; let mut cnt = 0u64; let mut sink = 0u64;
                while !dr.load(Ordering::Acquire) {
                    if let Some(r) = cr.reader().read_ref(&handle, last) {
                        last = r.epoch;
                        // Realistic reader work: iterate the payload.
                        for v in r.iter() { sink = sink.wrapping_add(*v); }
                    }
                    cnt += 1;
                }
                let _ = sink;
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }

        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_rwlock(n: usize) -> Stats {
        let lk   = Arc::new(RwLock::new(make_payload(0)));
        let done = Arc::new(AtomicBool::new(false));
        let wc   = Arc::new(AtomicU64::new(0));
        let rc   = Arc::new(AtomicU64::new(0));

        let (lw, dw, wcw) = (Arc::clone(&lk), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !dw.load(Ordering::Acquire) {
                *lw.write().unwrap() = make_payload(i);
                i += 1;
            }
            wcw.fetch_add(i, Ordering::Relaxed);
        });

        let mut readers = vec![];
        for _ in 0..n {
            let (lr, dr, rcr) = (Arc::clone(&lk), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut cnt = 0u64; let mut sink = 0u64;
                while !dr.load(Ordering::Acquire) {
                    let g = lr.read().unwrap();
                    for v in g.iter() { sink = sink.wrapping_add(*v); }
                    cnt += 1;
                }
                let _ = sink;
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }

        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    fn go_mutex(n: usize) -> Stats {
        let mu   = Arc::new(Mutex::new(make_payload(0)));
        let done = Arc::new(AtomicBool::new(false));
        let wc   = Arc::new(AtomicU64::new(0));
        let rc   = Arc::new(AtomicU64::new(0));

        let (mw, dw, wcw) = (Arc::clone(&mu), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !dw.load(Ordering::Acquire) {
                *mw.lock().unwrap() = make_payload(i);
                i += 1;
            }
            wcw.fetch_add(i, Ordering::Relaxed);
        });

        let mut readers = vec![];
        for _ in 0..n {
            let (mr, dr, rcr) = (Arc::clone(&mu), Arc::clone(&done), Arc::clone(&rc));
            readers.push(thread::spawn(move || {
                let mut cnt = 0u64; let mut sink = 0u64;
                while !dr.load(Ordering::Acquire) {
                    let g = mr.lock().unwrap();
                    for v in g.iter() { sink = sink.wrapping_add(*v); }
                    cnt += 1;
                }
                let _ = sink;
                rcr.fetch_add(cnt, Ordering::Relaxed);
            }));
        }

        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        for r in readers { r.join().unwrap(); }
        Stats { reads: rc.load(Ordering::Relaxed), writes: wc.load(Ordering::Relaxed),
                errors: 0, elapsed: t.elapsed().as_secs_f64() }
    }

    (go_bridge(n), go_rwlock(n), go_mutex(n))
}

fn bench_bridge_concurrent_big() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════════════════════════╗");
    println!("║  8. Bridge — concurrent writer + N readers (T = Vec<u64; 128>, ~1 KB)           ║");
    println!("║      readers iterate the payload (sum each read) — guard / ReadRef held a while ║");
    println!("╠══════════════════════════════════════════════════════════════════════════════════╣");
    println!("║  readers │  BridgedCell read_ref    │      RwLock<Vec>         │      Mutex<Vec>          ║");
    println!("║          │   writes/s │   reads/s   │   writes/s │   reads/s   │   writes/s │   reads/s   ║");
    println!("╠══════════╪════════════╪═════════════╪════════════╪═════════════╪════════════╪═════════════╣");

    for &n in &[1usize, 2, 4, 8] {
        let (bc, rw, mu) = run_bridge_scaling_big(n);
        let rate = |ops: u64, e: f64| fmt_rate(ops, e);
        println!("║   {:>2}     │ {:>10} │ {:>11} │ {:>10} │ {:>11} │ {:>10} │ {:>11} ║",
            n,
            rate(bc.writes, bc.elapsed), rate(bc.reads, bc.elapsed),
            rate(rw.writes, rw.elapsed), rate(rw.reads, rw.elapsed),
            rate(mu.writes, mu.elapsed), rate(mu.reads, mu.elapsed));
    }
    println!("╚══════════╧════════════╧═════════════╧════════════╧═════════════╧════════════╧═════════════╝");
    println!("  Large payload exposes the lock-cost asymmetry:");
    println!("    BridgedCell — reader iterates a pointer into the heap Block, writer keeps writing;");
    println!("                   old blocks queue on the retired list and reclaim drains them.");
    println!("    RwLock      — reader holds a read guard for the whole iteration; writer is blocked");
    println!("                   until every reader releases — writer/s collapses under reader load.");
    println!("    Mutex       — single lock serialises everything.");
}

// ===========================================================================
// 9. write_lazy vs write — does the Level A + B fast path actually pay?
// ===========================================================================
//
// Single-thread cost-per-write under three regimes:
//   * No reader ever exists  → Level B short-circuit (1 Acquire load)
//   * Reader exists, idle    → Level A full scan (64 Acquire loads)
//   * Reader is pinning      → both Levels fail, falls through to retire
//
// Baseline column = unconditional write() (always retires, batched
// reclaim every 1024 writes). The expectation: Level B is the biggest
// win (no alloc churn at all), Level A is positive (saves the retired
// + reclaim cycle), pinned case should be near-equal (both paths retire,
// write_lazy paid the scan cost for nothing).
fn bench_write_lazy() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  9. write_lazy vs write — Level A+B fast path payoff        ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    macro_rules! row {
        ($label:expr, $ops:expr, $elapsed:expr) => {{
            let r = $ops as f64 / $elapsed;
            println!("║  {:<46}  {:>8} ops/s  ║", $label, fmt_rate($ops, $elapsed));
            r
        }};
    }

    // Baseline: unconditional write + periodic reclaim_if_watermark.
    let base_write = {
        let registry = ReaderRegistry::new();
        let mut bc: BridgedCell<u64> = BridgedCell::new();
        let mut i = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            unsafe { bc.write(i); }
            i += 1;
            if i & 1023 == 0 { bc.reclaim_if_watermark(&registry); }
        }
        row!("BridgedCell<u64>  write  (no reader, +reclaim_if)", i, t.elapsed().as_secs_f64())
    };

    // Level B (no reader ever): write_lazy frees old block immediately.
    let lazy_no_reader = {
        let registry = ReaderRegistry::new();
        let mut bc: BridgedCell<u64> = BridgedCell::new();
        let mut i = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            unsafe { bc.write_lazy(i, &registry); }
            i += 1;
        }
        row!("BridgedCell<u64>  write_lazy  (Level B, no reader)", i, t.elapsed().as_secs_f64())
    };

    // Level A (handle acquired but idle, slot = MAX): floor_min == MAX
    // so old block is freed immediately. Pays the 64-slot scan per write.
    let lazy_idle_reader = {
        let registry = ReaderRegistry::new();
        let _h = registry.acquire();  // flips any_handle_ever, slot stays MAX
        let mut bc: BridgedCell<u64> = BridgedCell::new();
        let mut i = 0u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            unsafe { bc.write_lazy(i, &registry); }
            i += 1;
        }
        row!("BridgedCell<u64>  write_lazy  (Level A, idle reader)", i, t.elapsed().as_secs_f64())
    };

    // Pinning reader: write_lazy MUST retire (and we still call reclaim
    // periodically). Should be close to baseline minus a small scan tax.
    let lazy_pinning_reader = {
        let registry = ReaderRegistry::new();
        let handle   = registry.acquire();
        let mut bc: BridgedCell<u64> = BridgedCell::new();
        unsafe { bc.write(0u64); }
        // Hold a ReadRef the whole time so floor_min stays pinned at 1.
        let _r = bc.read_ref(&handle, 0).expect("must read");
        let mut i = 1u64;
        let t = Instant::now();
        while t.elapsed() < bench_dur() {
            unsafe { bc.write_lazy(i, &registry); }
            i += 1;
            if i & 1023 == 0 { bc.reclaim_if_watermark(&registry); }
        }
        row!("BridgedCell<u64>  write_lazy  (pinning reader)", i, t.elapsed().as_secs_f64())
    };

    println!("╠══════════════════════════════════════════════════════════════╣");
    println!("║  speedups vs baseline:                                       ║");
    println!("║    Level B (no reader)     × {:.2}                            ║", lazy_no_reader / base_write);
    println!("║    Level A (idle reader)   × {:.2}                            ║", lazy_idle_reader / base_write);
    println!("║    pinning reader          × {:.2}                            ║", lazy_pinning_reader / base_write);
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  Level B saves the Box::new + retired.push + future reclaim entirely.");
    println!("  Level A pays the floor scan (~64 cache-warm Acquire loads) for the");
    println!("    same end result. Worth it when readers are idle most of the time.");
    println!("  Pinned case shows the scan-tax: writer can't avoid retire, scan was wasted.");
}

// ===========================================================================
// 10. SpscQueue<T, N> vs Mutex<VecDeque<T>> — bounded FIFO throughput
// ===========================================================================
fn bench_spsc_queue() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  10. SpscQueue vs Mutex<VecDeque> — bounded SPSC throughput  ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    macro_rules! row {
        ($label:expr, $ops:expr, $elapsed:expr) => {{
            println!("║  {:<46}  {:>8} ops/s  ║", $label, fmt_rate($ops, $elapsed));
        }};
    }

    const CAP: usize = 1024;

    // SpscQueue: lock-free producer + consumer.
    {
        let q: Arc<SpscQueue<u64, CAP>> = Arc::new(SpscQueue::new());
        let done = Arc::new(AtomicBool::new(false));
        let pc = Arc::new(AtomicU64::new(0));
        let cc = Arc::new(AtomicU64::new(0));

        let (qp, dp, pcp) = (Arc::clone(&q), Arc::clone(&done), Arc::clone(&pc));
        let producer = thread::spawn(move || {
            let mut i = 0u64;
            while !dp.load(Ordering::Acquire) {
                loop {
                    match unsafe { qp.push(i) } { Ok(()) => break, Err(_) => thread::yield_now() }
                }
                i += 1;
            }
            pcp.fetch_add(i, Ordering::Relaxed);
        });

        let (qc, dc, ccc) = (Arc::clone(&q), Arc::clone(&done), Arc::clone(&cc));
        let consumer = thread::spawn(move || {
            let mut n = 0u64; let mut sink = 0u64;
            while !dc.load(Ordering::Acquire) {
                if let Some(v) = unsafe { qc.pop() } { sink ^= v; n += 1; }
            }
            // Drain anything left after stop signal.
            while let Some(v) = unsafe { qc.pop() } { sink ^= v; n += 1; }
            let _ = sink;
            ccc.fetch_add(n, Ordering::Relaxed);
        });

        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        producer.join().unwrap();
        consumer.join().unwrap();
        let e = t.elapsed().as_secs_f64();
        row!("SpscQueue<u64, 1024>  push (lock-free)", pc.load(Ordering::Relaxed), e);
        row!("SpscQueue<u64, 1024>  pop  (lock-free)", cc.load(Ordering::Relaxed), e);
    }

    // Mutex<VecDeque>: classic locked FIFO.
    {
        let q = Arc::new(Mutex::new(VecDeque::<u64>::with_capacity(CAP)));
        let done = Arc::new(AtomicBool::new(false));
        let pc = Arc::new(AtomicU64::new(0));
        let cc = Arc::new(AtomicU64::new(0));

        let (qp, dp, pcp) = (Arc::clone(&q), Arc::clone(&done), Arc::clone(&pc));
        let producer = thread::spawn(move || {
            let mut i = 0u64;
            while !dp.load(Ordering::Acquire) {
                loop {
                    let mut g = qp.lock().unwrap();
                    if g.len() < CAP { g.push_back(i); break; }
                    drop(g); thread::yield_now();
                }
                i += 1;
            }
            pcp.fetch_add(i, Ordering::Relaxed);
        });

        let (qc, dc, ccc) = (Arc::clone(&q), Arc::clone(&done), Arc::clone(&cc));
        let consumer = thread::spawn(move || {
            let mut n = 0u64; let mut sink = 0u64;
            while !dc.load(Ordering::Acquire) {
                let mut g = qc.lock().unwrap();
                if let Some(v) = g.pop_front() { sink ^= v; n += 1; }
                else { drop(g); }
            }
            // Drain.
            let mut g = qc.lock().unwrap();
            while let Some(v) = g.pop_front() { sink ^= v; n += 1; }
            let _ = sink;
            ccc.fetch_add(n, Ordering::Relaxed);
        });

        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        producer.join().unwrap();
        consumer.join().unwrap();
        let e = t.elapsed().as_secs_f64();
        row!("Mutex<VecDeque<u64>>  push (locked)", pc.load(Ordering::Relaxed), e);
        row!("Mutex<VecDeque<u64>>  pop  (locked)", cc.load(Ordering::Relaxed), e);
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  SpscQueue: each side does 1 Acquire load (other side's index)");
    println!("              + 1 Release store. No lock, no kernel, no allocation.");
    println!("  Mutex<VecDeque>: kernel-backed lock per op + heap (VecDeque internals).");
}

// ===========================================================================
// 11. DoubleBuffer<T> vs RwLock<T> — frame-boundary publish
// ===========================================================================
fn bench_double_buffer() {
    println!();
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  11. DoubleBuffer vs RwLock<T> — frame-boundary publish      ║");
    println!("╠══════════════════════════════════════════════════════════════╣");

    macro_rules! row {
        ($label:expr, $ops:expr, $elapsed:expr) => {{
            println!("║  {:<46}  {:>8} ops/s  ║", $label, fmt_rate($ops, $elapsed));
        }};
    }

    // T = Vec<u64; 128>, the same ~1 KB payload as section 8.
    fn make_payload(seed: u64) -> Vec<u64> {
        (0..128u64).map(|i| seed.wrapping_add(i)).collect()
    }

    // DoubleBuffer<Vec<u64>>: writer does write() (back overwrite + swap),
    // reader iterates front. No lock, no alloc beyond the new payload.
    {
        let db = Arc::new(DoubleBuffer::<Vec<u64>>::new(make_payload(0)));
        let done = Arc::new(AtomicBool::new(false));
        let wc = Arc::new(AtomicU64::new(0));
        let rc = Arc::new(AtomicU64::new(0));

        let (dw, dn, wcw) = (Arc::clone(&db), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !dn.load(Ordering::Acquire) {
                unsafe { dw.write(make_payload(i)); }
                i += 1;
            }
            wcw.fetch_add(i, Ordering::Relaxed);
        });

        let (dr, dn2, rcr) = (Arc::clone(&db), Arc::clone(&done), Arc::clone(&rc));
        let reader = thread::spawn(move || {
            let mut n = 0u64; let mut sink = 0u64;
            while !dn2.load(Ordering::Acquire) {
                let v = dr.read();
                for x in v.iter() { sink = sink.wrapping_add(*x); }
                n += 1;
            }
            let _ = sink;
            rcr.fetch_add(n, Ordering::Relaxed);
        });

        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        reader.join().unwrap();
        let e = t.elapsed().as_secs_f64();
        row!("DoubleBuffer<Vec<u64>>  write+swap", wc.load(Ordering::Relaxed), e);
        row!("DoubleBuffer<Vec<u64>>  read+iterate", rc.load(Ordering::Relaxed), e);
    }

    {
        let lk = Arc::new(RwLock::new(make_payload(0)));
        let done = Arc::new(AtomicBool::new(false));
        let wc = Arc::new(AtomicU64::new(0));
        let rc = Arc::new(AtomicU64::new(0));

        let (lw, dw, wcw) = (Arc::clone(&lk), Arc::clone(&done), Arc::clone(&wc));
        let writer = thread::spawn(move || {
            let mut i = 0u64;
            while !dw.load(Ordering::Acquire) {
                *lw.write().unwrap() = make_payload(i);
                i += 1;
            }
            wcw.fetch_add(i, Ordering::Relaxed);
        });

        let (lr, dr, rcr) = (Arc::clone(&lk), Arc::clone(&done), Arc::clone(&rc));
        let reader = thread::spawn(move || {
            let mut n = 0u64; let mut sink = 0u64;
            while !dr.load(Ordering::Acquire) {
                let g = lr.read().unwrap();
                for x in g.iter() { sink = sink.wrapping_add(*x); }
                n += 1;
            }
            let _ = sink;
            rcr.fetch_add(n, Ordering::Relaxed);
        });

        let t = Instant::now();
        thread::sleep(bench_dur());
        done.store(true, Ordering::Release);
        writer.join().unwrap();
        reader.join().unwrap();
        let e = t.elapsed().as_secs_f64();
        row!("RwLock<Vec<u64>>        write", wc.load(Ordering::Relaxed), e);
        row!("RwLock<Vec<u64>>        read+iterate", rc.load(Ordering::Relaxed), e);
    }
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  DoubleBuffer: no lock, no refcount; caller is responsible for");
    println!("                  not holding a read() across a publish (frame sync).");
    println!("                Writer overwrites the back slot then 1 Release store.");
    println!("  RwLock<Vec>:  safe by construction, but write blocks all readers");
    println!("                and read blocks writer for the iterate duration.");
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
fn main() {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("axOS Cell<T> benchmarks");
    println!("  {} logical CPUs  •  {}s per measurement", cpus, BENCH_SECS);
    println!("  Cell<u32>    = inline path (1 Release store / 1 Acquire load, zero alloc)");
    println!("  SeqCell<u64> = seqlock path (inline storage, zero alloc, possible brief spin)");
    println!("  BridgedCell  = Cell + retired list + epoch-floor reclamation (Block path)");
    println!("  Mutex/RwLock = blocking (OS primitives, all sizes)");

    bench_single();
    bench_sparse();
    bench_heavy();
    bench_stress();
    bench_seqcell_sizes();
    bench_bridge_single();
    bench_bridge_concurrent_small();
    bench_bridge_concurrent_big();
    bench_write_lazy();
    bench_spsc_queue();
    bench_double_buffer();

    println!();
    println!("Done.");
}
