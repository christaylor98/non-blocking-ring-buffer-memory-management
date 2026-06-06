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

use mem::{Cell, SeqCell, ReadResult};
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
// main
// ---------------------------------------------------------------------------
fn main() {
    let cpus = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
    println!("axOS Cell<T> benchmarks");
    println!("  {} logical CPUs  •  {}s per measurement", cpus, BENCH_SECS);
    println!("  Cell<u32>    = inline path (1 Release store / 1 Acquire load, zero alloc)");
    println!("  SeqCell<u64> = seqlock path (inline storage, zero alloc, possible brief spin)");
    println!("  Mutex/RwLock = blocking (OS primitives, all sizes)");

    bench_single();
    bench_sparse();
    bench_heavy();
    bench_stress();

    println!();
    println!("Done.");
}
