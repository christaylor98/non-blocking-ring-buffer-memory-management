# NBM_ADAPTIVE_ADOPTION_V1

Authoritative IS spec: finish adopting the AdaptiveCell prototype into the crate.
Run a Claude Code session with working dir
`/home/chris/dev/non-blocking-ring-buffer-memory-management`.
Status: F1 (shrink ReadRef) DONE + verified (60 + 65 tests green). Remaining:
F4, F5, F2, AdaptiveCell promotion, README.

```lisp
;; ============================================================
;; INTENT DECLARATION — AUTHORITATIVE BLOCK
;; NBM_ADAPTIVE_ADOPTION_V1
;; ============================================================
;; You are operating under INTENT_SYSTEM_SPEC.v1.0.
;; (intent-id NBM_ADAPTIVE_ADOPTION_V1)
;;
;; ------------------------------------------------------------
;; LONG-CONTEXT REHYDRATION ANCHOR
;; ------------------------------------------------------------
;; All constraints remain binding. Absence implies forbidden.
;; Constraint > Priority > Goal.
;; This crate is the single-writer / multi-reader memory substrate the
;; bridge's H1 channels will sit on. Correctness dominates everything.
;;
;; GRAVITY ANCHORS (rehydrate on long context):
;;   ONE_WRITER          — exactly one owning writer (&mut self); any number of
;;                         &self readers. Never weaken this contract.
;;   NO_UNPINNED_DEREF   — no unpinned Block dereference may coexist with any
;;                         free site, EVER. (This rule's violation was the V1
;;                         UAF the stress test caught.)
;;   SLOT_MIRRORED_READ  — the inline slot is kept current in BOTH modes, so
;;                         owned read() is a pure seqlock and never derefs a
;;                         Block. Only floor-pinned read_pinned() touches Blocks.
;;   T_IS_COPY           — these cells are T: Copy. Non-Copy T is a different
;;                         design; do not bolt it on here.
;;   NO_CAS              — demand is advisory; a stale scan costs one wrong-mode
;;                         write, never safety. Keep it CAS-free.
;;   GREEN_OR_REVERT     — every change is gated on the full suite + ASan + the
;;                         stress tests staying green. A red stress run = revert.

(intent-mode (state execution) (authority human) (downgrade-allowed false))

(intent
  (identity
    (name "NBM_ADAPTIVE_ADOPTION_V1")
    (owner "chris")
    (scope "Finish adopting the AdaptiveCell prototype into the crate at
            /home/chris/dev/non-blocking-ring-buffer-memory-management.
            F1 (shrink ReadRef) is DONE and verified (60 + 65 tests green).
            Remaining: F4 (pad floor slots), F5 (reclaim back-off), F2 (read()
            hazard: doc + slot-mirroring), promote AdaptiveCell into
            non-blocking-memory.rs as a sibling of BridgedCell, README. Excludes:
            non-Copy T, the bridge/H1 wiring, any new public API beyond the
            primitives table."))

  (goal
    (primary "Land F4, F5, and the F2 fix; move AdaptiveCell into
              non-blocking-memory.rs as a first-class primitive returning the
              shrunken ReadRef (no Pinned wrapper internally); document it. Keep
              every existing test, the stress tests, and an ASan pass green.")
    (secondary "Confirm the measured wins: F1 closed the pinned-read composition
                gap; F4 improves multi-reader (scenario B) scaling.")
    (type outcome-oriented))

  ;; ----------------------------------------------------------
  ;; ENVIRONMENT — verified
  ;; ----------------------------------------------------------
  ;; cd /home/chris/dev/non-blocking-ring-buffer-memory-management
  ;; Standalone rustc files (no Cargo). Source of truth: HANDOVER-adaptive-cell.md.
  ;;   tests (60):  rustc --edition 2021 --test non-blocking-memory.rs -o nbm-test && ./nbm-test
  ;;   tests (65):  rustc --edition 2021 --test adaptive-cell-prototype.rs -o adaptive-test && ./adaptive-test
  ;;   benches:     rustc --edition 2021 -O adaptive-cell-prototype.rs -o adaptive-bench
  ;;                ./adaptive-bench           # single-thread
  ;;                ./adaptive-bench mt        # multithread
  ;;   ASan:        RUSTFLAGS="-Z sanitizer=address" rustc +nightly --edition 2021 \
  ;;                  --test non-blocking-memory.rs -o nbm-asan && ./nbm-asan
  ;;   F1 size:     printf '#[path="/home/chris/dev/non-blocking-ring-buffer-memory-management/non-blocking-memory.rs"] mod m;\nfn main(){println!("{} B", std::mem::size_of::<m::ReadRef<[u64;16]>>());}\n' > /tmp/sz.rs
  ;;                rustc --edition 2021 -O /tmp/sz.rs -o /tmp/sz && /tmp/sz   # expect ~48 B
  ;;
  ;; KEY LOCATIONS (non-blocking-memory.rs):
  ;;   MAX_READERS = 64                          (~line 86)
  ;;   ReaderRegistry { slots: [AtomicU64; MAX_READERS] }  (~946); init (~966);
  ;;     floor scan `for s in self.slots.iter()` (~995); ReaderHandle.slot_atomic
  ;;     `&self.registry.slots[self.slot]` (~1042)
  ;;   reclaim_if_watermark(...)                 (~1345)
  ;;   BridgedCell owned read() (clones from head Block, pins nothing) — F2 target
  ;;   AdaptiveCell + AdaptiveRegistry/Handle + Pinned live in
  ;;     adaptive-cell-prototype.rs (it #[path]-includes non-blocking-memory.rs)

  ;; ----------------------------------------------------------
  ;; CONSTRAINT — hard limits
  ;; ----------------------------------------------------------
  (constraint
    (hard-limit "DO NOT REDO F1. ReadRefInner::Inline is already u32 with a
                 reinterpret-on-deref. Leave it.")
    (hard-limit "PRESERVE ONE_WRITER, NO_UNPINNED_DEREF, SLOT_MIRRORED_READ,
                 T_IS_COPY, NO_CAS verbatim. Any change that touches the floor
                 protocol must keep floors honoured at every free site.")
    (hard-limit "GREEN GATE. After EACH change run the 60-test suite, the
                 prototype 65-test suite, and the three stress tests. Before
                 declaring done, run the ASan build clean. A red stress or ASan
                 run is a revert, not a debug-in-place.")
    (hard-limit "NO API SURFACE CREEP. Add AdaptiveCell + a README row only. No
                 new public types beyond what the prototype already exposes.")
    (hard-limit "NO non-Copy T support. Out of scope."))

  ;; ----------------------------------------------------------
  ;; THE WORK — ordered, each gated on green
  ;; ----------------------------------------------------------
  ;; W1 — F4: floor-slot false sharing.
  ;;   Wrap each floor slot so neighbouring readers don't share a line:
  ;;     #[repr(align(64))] struct FloorSlot(AtomicU64);
  ;;     slots: [FloorSlot; MAX_READERS]
  ;;   Update init (~966), the floor scan (~995), and slot_atomic (~1042) to go
  ;;   through `.0`. Re-run MT scenario B; expect read_ref scaling to improve.
  ;;
  ;; W2 — F5: reclaim_if_watermark every-write regime.
  ;;   When retired_len >= WATERMARK but floors are starved, the sweep runs every
  ;;   write freeing nothing. Add exponential back-off after a zero-freed sweep
  ;;   (watermark-triggered, but skip-count doubles on each failed drain, capped).
  ;;   Keep retired_len bounded; do not let it ride to MBs. Add a test asserting
  ;;   a starved-floor write storm does not sweep every write.
  ;;
  ;; W3 — F2: BridgedCell::read() reclaim hazard.
  ;;   Owned read() clones from the head Block while pinning nothing — sound only
  ;;   under &mut/&self exclusivity. Adopt the prototype's SLOT_MIRRORED_READ so
  ;;   owned reads serve from the inline slot and never deref a Block (preferred),
  ;;   AND add a loud doc warning on read(). If slot-mirroring is deferred, the
  ;;   doc warning is mandatory.
  ;;
  ;; W4 — Promote AdaptiveCell into non-blocking-memory.rs.
  ;;   Move AdaptiveRegistry / AdaptiveHandle / AdaptiveCell (and the ctrl-word
  ;;   transition logic + COOL_DOWN=64 hysteresis + SWEEP_PERIOD) in as a sibling
  ;;   of BridgedCell. Inside the crate, drop the Pinned wrapper: return the
  ;;   shrunken ReadRef plus a cell-level epoch (extend ReadRef with the cell
  ;;   epoch, or return (ReadRef, epoch)). Decide epoch semantics: ctrl epoch
  ;;   counts ALL writes (both modes); Block epochs count only hot writes and stay
  ;;   internal — if unifying, drive Block epochs from the ctrl count via
  ;;   write_with_epoch (floors only need Block epochs monotone; gaps fine,
  ;;   reordering not). Carry F6 (sticky-demand conditional store) — already in
  ;;   the prototype. Move the 5 adaptive_* tests into the crate's test module.
  ;;
  ;; W5 — README: add the AdaptiveCell row to the primitives table
  ;;   ("demand-driven SeqCell/BridgedCell hybrid; pinned views on demand,
  ;;   seqlock speed otherwise") and a picking-guide line
  ;;   ("pin demand intermittent or unknown -> AdaptiveCell").
  ;;
  ;; DEFERRED (note, do not build): F3 writer-priority signal for bare BridgedCell
  ;;   (AdaptiveCell's ctrl seqlock already yields readers); loom/shuttle model
  ;;   checking of the transition protocol; ctrl epoch-overflow audit (63:2 bits).

  ;; ----------------------------------------------------------
  ;; ERROR / STOP CONDITIONS
  ;; ----------------------------------------------------------
  ;; - Any stress test fails, or ASan reports a fault -> STOP, revert the last
  ;;   change, report verbatim. Treat a torn read containing recycled-heap
  ;;   garbage as a NO_UNPINNED_DEREF violation.
  ;; - A floor-protocol change that can free a Block under a live unpinned read
  ;;   -> forbidden; redesign.

  ;; ----------------------------------------------------------
  ;; DELIVERABLE + VERIFICATION
  ;; ----------------------------------------------------------
  ;; - F4, F5, F2 landed; AdaptiveCell a first-class primitive in
  ;;   non-blocking-memory.rs; README updated.
  ;; - non-blocking-memory.rs test suite green (>= the original 60 + the moved
  ;;   adaptive_* tests).
  ;; - The mode-flapping stress test passes 10/10 runs.
  ;; - ASan build clean.
  ;; - Benches re-run: report single-thread + `mt` numbers; confirm the
  ;;   pinned-read gap (F1) and scenario-B scaling (F4) improvements.
  ;; - F1 size check prints ~48 B for ReadRef<[u64;16]>.
) ;; END NBM_ADAPTIVE_ADOPTION_V1
```
