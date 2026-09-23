# F2-A async DCP cut fix

```json
{
  "fix_local": "2026-09-22 14:40:01 +0800",
  "file": "scripts/ft/ft1_fault_hooks.py",
  "sha256": "7e35e6da83cf9635f4d1d30ff50d69af1ff11f8a7863845816ddfac93bce272e",
  "bug": "F2 A-arm cut fired on 2nd optimizer_success while async_save step-0 DCP still incomplete (checkpoint_save_returned files=[common.pt] only; disk later had distcp without .metadata). Native recover then failed: not a distributed checkpoint; no final-native-state.",
  "fix": "Before F2 fault_ready: wait_async_saves() and require predecessor recover_checkpoint to contain .metadata + non-empty .distcp.",
  "evidence_refs": [
    "p3_evidence/ft1-f2-s419-a-r3/observer-pilot",
    "p3_evidence/ft1-f2-s419-a-r3/backfill/BLOCKED.json"
  ],
  "repro_new_rid": [
    "Confirm scripts/ft/ft1_fault_hooks.py contains wait_async_saves + _require_complete_dcp",
    "Keep formal_sample=false; new rid e.g. ft1-f2-s419-a-r4; do not overwrite r2/r3",
    "Launch via existing FT1 F2-A path (run_training_fault / run_ft1_faults pattern)",
    "Expect fault_ready.predecessor_files to include .metadata; after recover expect final-native-state.json and acceptance FV/load"
  ],
  "freeze_note": "ft1-fault-freeze.json hash for ft1_fault_hooks.py must be refreshed before formal samples; engineering r4 can proceed with updated working tree"
}
```
