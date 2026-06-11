"""Process exit that skips C-runtime finalization.

Why this exists: transformers / mlx-audio / mlx-lm pull in `sentencepiece`, whose
`.so` statically links abseil. On macOS arm64 one of abseil's global
`absl::Flag<bool>` destructors faults (SIGBUS / KERN_PROTECTION_FAILURE) when it
runs during `__cxa_finalize` at normal interpreter shutdown — AFTER our program
and all its cleanup have finished. The result is an intermittent crash report
(~1-2/day) on an otherwise successful run. See the crash signature:
    EXC_BAD_ACCESS (SIGBUS) in _sentencepiece...so  absl::Flag<bool>::~Flag()
    <- __cxa_finalize_ranges <- exit <- Py_Exit  (i.e. process teardown)

`os._exit()` terminates via the `_exit(2)` syscall, which never runs
`__cxa_finalize`, so those C++ static destructors never execute and the crash
class is eliminated by construction — independent of the (stochastic) trigger.

Safety: this is only safe because Rehearse does ALL resource cleanup explicitly
in `finally:` blocks (SQLite `store.close()`, `out_stream.stop()/.close()`), and
registers NO `atexit` handlers. By the time a CLI entry point calls `fast_exit`,
every DB write is already committed (each is `with conn:` → autocommit) and every
connection/stream is closed. So bypassing finalization loses nothing.

IMPORTANT: call this ONLY from CLI entry shims (`_cli`), never from `main()`.
`main()` stays a pure `-> int` so the test suite can call it without the test
process being killed by `os._exit`.
"""

import os
import sys


def fast_exit(code: int | None) -> "NoReturn":  # type: ignore[name-defined]
    """Flush stdio, run Python's own resource cleanup, then terminate via
    `_exit(2)` — which never runs `__cxa_finalize`, so the abseil dtor never fires.

    We deliberately run `multiprocessing.util._exit_function()` first: it is the
    exact cleanup multiprocessing registers with `atexit`, and skipping it (raw
    `os._exit`) leaves a tracked POSIX semaphore behind, which makes the
    resource_tracker print a "leaked semaphore" warning at shutdown. Running it
    explicitly reclaims those resources cleanly; we then `_exit(2)` to skip ONLY
    the broken C-runtime finalization phase. Deterministic — no atexit-ordering
    dependence. Both calls are best-effort; nothing here may block the exit."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    try:
        # Private but stable for years; this is multiprocessing's own atexit hook.
        from multiprocessing.util import _exit_function
        _exit_function()
    except Exception:
        pass
    os._exit(int(code or 0))


def run_cli(main) -> "NoReturn":  # type: ignore[name-defined]
    """Universal CLI funnel: run `main()` and exit via fast_exit on EVERY path —
    normal return, Ctrl-C (even outside main's own try blocks, e.g. during the
    30-60s model warmup), SystemExit, or an uncaught error. This guarantees the
    process never reaches `__cxa_finalize`, so the sentencepiece/abseil SIGBUS
    cannot fire regardless of how the program ends.

    Without this funnel, a KeyboardInterrupt raised in a startup window not
    wrapped by main()'s try would propagate past us into normal interpreter
    shutdown — exactly the crash path we're closing."""
    import traceback
    code: int = 0
    try:
        code = main() or 0
    except KeyboardInterrupt:
        code = 130
    except SystemExit as e:  # preserve an explicit sys.exit(code)
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
    except BaseException:  # noqa: BLE001 — print like the default hook, then clean-exit
        traceback.print_exc()
        code = 1
    fast_exit(code)
