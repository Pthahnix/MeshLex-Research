# 25_objaverse_pipeline_disk_rescue_2026_03_20

## Background

During the Daft-based Objaverse streaming pipeline on 2026-03-20, disk usage on the 80GB container rapidly climbed to 73GB+ (91% full) while only ~20/93 Objaverse batches had been processed. This created an imminent risk of the full dataset build crashing mid-run.

## Root Cause

The root cause was **incorrect Objaverse cache cleanup logic** in `scripts/stream_objaverse_daft.py`.

Old logic tried to reconstruct cached GLB paths as:

```python
objaverse_cache / uid[:2] / f"{uid}.glb"
```

But the actual Objaverse cache layout under:

```text
/home/cc/.objaverse/hf-objaverse-v1/glbs/
```

uses bucket directories like:

```text
000-090/<uid>.glb
000-002/<uid>.glb
...
```

So the cleanup code almost never matched any real file. As a result, processed GLBs accumulated indefinitely.

## Evidence Collected

- `/home/cc/.objaverse` reached **71GB**
- Repository + `results/` were only a few hundred MB, so they were not the main issue
- Age-bucket scan showed:
  - `>30m`: **9000 files / 64.0GB**
- This confirmed that most old GLBs were stale and safe to delete

## Emergency Rescue Performed

A safe emergency cleanup deleted only GLBs older than 30 minutes:

- Deleted files: **9000**
- Freed space: **64.0GB**
- Disk recovered from roughly **73GB used / 7.6GB free**
  to **14GB used / 67GB free**

This was done without killing the current batch, so the running pipeline survived.

## Permanent Fix

A permanent fix was implemented in `scripts/stream_objaverse_daft.py`:

- New helper: `cleanup_objaverse_cache(objects)`
- It now deletes GLBs using the **actual paths returned by `objaverse.load_objects()`**, instead of guessing the path from the UID

### Regression Test Added

New test file:

- `tests/test_stream_objaverse_daft.py`

The regression test proves cleanup removes the real returned path and does not depend on guessed directory structure.

### Verification

Fresh verification command:

```bash
PYTHONPATH=/workspace/MeshLex-Research python3 -m pytest -q \
  tests/test_stream_objaverse_daft.py \
  tests/test_stream_utils.py \
  tests/test_daft_utils.py \
  tests/test_generate_splits.py
```

Result:

- **14 passed**

## Operational Decision Taken

Because the running tmux pipeline had started before the fix existed, it would not automatically benefit from the corrected cleanup logic.

So the chosen recovery procedure was:

1. Wait for the currently running batch boundary (`batch_020`) to finish processing and be recorded in `progress.json`
2. Commit + push the cleanup fix
3. Kill the tmux session only after the batch boundary
4. Restart the pipeline
5. Resume from `progress.json`, skipping `batch_000` through `batch_020`
6. Continue from `batch_021` with the fixed cleanup logic

This ensured:

- no duplicate reprocessing of completed batches
- no corruption of in-flight work
- future batches automatically clean their own GLB cache

## Important Ongoing Rule For Future Tasks

For all later dataset-building / streaming / large-download tasks in this repository:

1. **Always inspect real cache layout before writing deletion logic**
   - never assume UID-derived directory names
   - prefer deleting via actual returned file paths

2. **Monitor disk continuously during long-running pipelines**
   - especially when working with Objaverse / HF caches / extracted archives
   - a log/monitor report alone is not enough; inspect actual disk consumers with `du`

3. **For active pipelines, prefer boundary-safe interventions**
   - do not kill an in-flight batch unless absolutely necessary
   - wait for batch completion / checkpoint / progress file update

4. **When emergency cleanup is needed, use age-based deletion first**
   - deleting only sufficiently old cache files is a good safety-first strategy
   - it avoids touching current batch inputs

5. **If a root-cause fix is made during a live run, restart only after a safe boundary**
   - otherwise the running process will keep using old code

6. **Objaverse GLB cache is disposable after processing**
   - the durable artifact is the uploaded HF parquet + progress metadata
   - the local GLB cache is just transient input material

## Files Relevant To This Incident

- `scripts/stream_objaverse_daft.py`
- `tests/test_stream_objaverse_daft.py`
- `/tmp/dataset_pipeline.log`
- `/tmp/meshlex/objaverse/progress.json`
- `/home/cc/.objaverse/hf-objaverse-v1/glbs/`

## Practical Reminder

If disk usage spikes again during this pipeline, first check:

```bash
df -h /
du -xh --max-depth=2 /home/cc /tmp /workspace/MeshLex-Research 2>/dev/null | sort -h | tail -40
du -xh --max-depth=4 /home/cc/.objaverse 2>/dev/null | sort -h | tail -30
```

Do not assume the repo or `results/` is the culprit; in this incident, the dominant source was Objaverse GLB cache accumulation.
