# F1-R injection race fix (2026-09-22)

## Failure
`ft1-f1-s421-r-r1`: markers `f1-active.json` / `f1-complete.json` present, but no `f1-claimed.json` / descendant / signal. Controller events stop at launcher_registered → result (`technical_invalid`). A-arm r1 armed normally.

## Root cause
`ft1_scheduler_observer.claim` ran only at the **start** of `run_batch` and required trainer-written `f1-complete.json` plus unfinished matching requests.

On R, all k=8 samples for source_row_id 5518 finished within ~249ms (likely one engine step). Trainer `generation_complete` (and thus `f1-complete.json`) is written **after** those results leave the scheduler. By the next `run_batch`, unfinished matching requests were already gone → claim never armed. A-arm got lucky with a staggered window (claim cut while a sibling was still unfinished).

## Fix
In `scripts/ft/ft1_scheduler_observer.py`:
1. Attempt claim **before and after** the real `run_batch`.
2. Arming proof may be trainer `f1-complete.json` **or** in-scheduler sibling state (`>=1` finished and `>=1` unfinished matching `origin_input_ids`).
3. Normalize `origin_input_ids` with `list(...)` before compare.

Backup: `scripts/ft/ft1_scheduler_observer.py.bak-20260922-r-injection`

## Retest (ops)
- Keep A-r1 / R-r1 failure evidence.
- New rid **F1-R only**, `formal_sample=false`, seed 421.
- Expect `f1-claimed.json`, descendant_registered, signal, then acceptance chain.
- Pair remains No-Go until R passes audit.
