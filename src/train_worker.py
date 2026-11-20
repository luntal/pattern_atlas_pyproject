"""Training worker process, started by the server (TrainingManager).

    python train_worker.py --config <config.json> [--stop-file <path>]

stdout is a pure event channel: one JSON object per line (see modules/training.py for the event types, plus
{"type":"done","stopped":bool,...} and {"type":"error","message":str}). Everything else that is printed
(including output of libraries and C extensions) is redirected to stderr.

Stopping: SIGTERM / SIGINT or the existence of the stop file sets the stop flag. The training checks it once
per batch, then saves model_last_*.pt / model_*.pt and exits cleanly with a "done" event (stopped = true).
"""
import argparse
import json
import os
import signal
import sys
import threading
import traceback

# Keep the real stdout for events and send every other write to stderr, even from C code.
_EVENT_FD = os.dup(1)
os.dup2(2, 1)
_event_out = os.fdopen(_EVENT_FD, 'w', buffering=1, encoding='utf-8')
sys.stdout = sys.stderr
_write_lock = threading.Lock()

_stop_flag = threading.Event()


def emit(event: dict):
    line = json.dumps(event, default=str)
    with _write_lock:
        try:
            _event_out.write(line + '\n')
            _event_out.flush()
        except (BrokenPipeError, ValueError):
            # The server is gone; make sure the training winds down instead of running unattended.
            _stop_flag.set()


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='path to the TrainConfig JSON file')
    parser.add_argument('--stop-file', default=None, help='training stops when this file exists')
    args = parser.parse_args(argv)

    def _handle_signal(signum, _frame):
        _stop_flag.set()

    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    if hasattr(signal, 'SIGBREAK'):  # Windows
        signal.signal(signal.SIGBREAK, _handle_signal)

    def should_stop() -> bool:
        if _stop_flag.is_set():
            return True
        return bool(args.stop_file) and os.path.exists(args.stop_file)

    emit({'type': 'status', 'text': 'worker started', 'pid': os.getpid()})
    try:
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
        # Imported late so that start-up problems are reported as an error event.
        from modules.training import run_training

        result = run_training(config, on_event=emit, should_stop=should_stop)
    except BaseException as exc:  # noqa: BLE001 - report everything to the server
        traceback.print_exc()
        emit({'type': 'error', 'message': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
        return 1

    emit({'type': 'done', 'stopped': bool(result.get('stopped')), 'result': result})
    return 0


if __name__ == '__main__':
    sys.exit(main())
