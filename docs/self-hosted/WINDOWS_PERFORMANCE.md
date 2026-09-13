# Ledger write performance on Windows

Measured, not estimated. Numbers below are from the same 60,000-record workload
(`tests/test_cold_sweep_hardening.py::test_savings_log_is_size_bounded`).

| Platform / mode | Throughput | Wall time | File opens |
|---|---|---|---|
| Linux, synchronous (default) | ~36,000 rec/s | 1.6 s | 60,552 |
| **Windows, synchronous (default)** | **~626 rec/s** | **95.9 s** | 60,552 |
| Linux, buffered (128) | ~58,000 rec/s | 1.0 s | 935 |
| Linux, buffered (512) | ~57,000 rec/s | 1.1 s | 354 |

## Why Windows is ~50x slower here

The synchronous ledger **re-opens the file for every record**. On Linux an
open/close pair costs roughly 25 microseconds; on Windows it costs roughly 1.5
milliseconds once NTFS metadata handling and real-time antivirus scanning are
included. 60,552 opens x ~1.5 ms is ~91 seconds — which matches the 95.9 s
observed almost exactly. The cost is the open count, not the data volume.

**This is a deliberate durability trade-off, not a defect.** Re-opening per
record is what makes the ledger safe under concurrency: another process may trim
the log and `os.replace` it at any moment, and a cached file handle would then
be appending to a dead inode — silently writing records into a file nobody will
ever read. Opening per append always resolves the current file. That property is
why multi-process ledgers reconcile exactly, and it is worth more than the
milliseconds it costs.

## Is this a production problem?

Almost certainly not. One ledger record corresponds to one model call, and a
model call takes 100 ms to several seconds. Even the Windows figure of ~626
records/second is far beyond what a single application process can generate:
sustaining it would require 626 concurrent completions per second from one
process. Typical production rates are 1-50 calls/second, where the ledger is
10x-600x faster than required.

Enable buffered mode if you are genuinely at high volume, on Windows, or on a
slow or network-backed disk:

```python
tokeymeter.set_buffered_savings(True, buffer_size=128)
```

Buffering batches appends, cutting file opens by roughly 65x (60,552 -> 935 in
the measurement above). The trade-off is explicit: records live in memory until
the buffer flushes, so an abrupt process kill can lose the un-flushed tail. The
crash-tail repair still protects the file itself from corruption; what you lose
is the most recent unflushed records, not the integrity of the ones on disk.

## Running the test suite on Windows

The file-open-heavy tests are marked `io_heavy`. For fast local iteration:

```
pytest -m "not io_heavy" -q
```

Run the complete set before any release — those tests cover bounded retention
and post-trim report correctness, and they are exactly the paths a long-running
production ledger exercises.

Two Windows-specific pytest notes, neither of which indicates a product problem:

- Give pytest a per-run private temp directory. A shared one produces
  `access denied` errors on Windows when a prior run's handles are still open:
  `pytest --basetemp=<fresh dir> -p no:cacheprovider`
- Point `TOKEYMETER_HOME` at a writable directory for the run, so ledger, cache,
  and audit state land somewhere predictable and get cleaned up with it.
