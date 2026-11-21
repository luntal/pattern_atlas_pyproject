"""Training worker process, started by the server (TrainingManager).

    python train_worker.py --config <config.json> [--stop-file <path>]

stdout is a pure event channel: one JSON object per line (see modules/training.py for the event types, plus
{"type":"done","stopped":bool,...} and {"type":"error","message":str}). Everything else that is printed
(including output of libraries and C extensions) is redirected to stderr.

Stopping: SIGTERM / SIGINT or the existence of the stop file sets the stop flag. The training checks it once
per batch, then saves model_last_*.pt / model_*.pt and exits cleanly with a "done" event (stopped = true).

Multi-GPU: when config.options.gpus selects more than one CUDA device, this process is rank 0 and starts one extra
process per further GPU (same script, RANK / WORLD_SIZE / MASTER_* in the environment, DistributedDataParallel).
Only rank 0 writes events to stdout. The stop request is synchronised between the ranks inside train_vae. If a rank
dies, rank 0 reports the error and terminates the others; extra ranks end when rank 0 disappears.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback

# Keep the real stdout for events and send every other write to stderr, even from C code.
_EVENT_FD = os.dup(1)
os.dup2(2, 1)
_event_out = os.fdopen(_EVENT_FD, 'w', buffering=1, encoding='utf-8')
sys.stdout = sys.stderr
_write_lock = threading.Lock()

_stop_flag = threading.Event()

_RANK = int(os.environ.get('RANK', '0') or 0)
_PARENT_PID = int(os.environ.get('RBF_DDP_PARENT_PID', '0') or 0)
_children: list = []
_finished = threading.Event()


def emit(event: dict):
    if _RANK > 0:
        return  # only rank 0 talks to the server
    line = json.dumps(event, default=str)
    with _write_lock:
        try:
            _event_out.write(line + '\n')
            _event_out.flush()
        except (BrokenPipeError, ValueError):
            # The server is gone; make sure the training winds down instead of running unattended.
            _stop_flag.set()


def _start_extra_ranks(args, world_size: int):
    """Start ranks 1..world_size-1 and set the rendezvous environment for this process (rank 0)."""
    from modules.distributed import find_free_port

    base = {
        'MASTER_ADDR': '127.0.0.1',
        'MASTER_PORT': str(find_free_port()),
        'WORLD_SIZE': str(world_size),
        'RBF_DDP_PARENT_PID': str(os.getpid()),
    }
    script = os.path.abspath(__file__)
    for rank in range(1, world_size):
        env = dict(os.environ)
        env.update(base)
        env['RANK'] = str(rank)
        env['LOCAL_RANK'] = str(rank)
        cmd = [sys.executable, script, '--config', args.config]
        if args.stop_file:
            cmd += ['--stop-file', args.stop_file]
        _children.append(subprocess.Popen(cmd, env=env, stdout=subprocess.DEVNULL, cwd=os.path.dirname(script)))
    os.environ.update(base)
    os.environ['RANK'] = '0'
    os.environ['LOCAL_RANK'] = '0'


def _stop_extra_ranks(wait: float = 30.0):
    deadline = time.time() + wait
    for proc in _children:
        try:
            proc.wait(timeout=max(0.1, deadline - time.time()))
        except subprocess.TimeoutExpired:
            proc.terminate()
    for proc in _children:
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _watch_extra_ranks():
    """Rank 0: a rank that died would leave the others hanging in a collective, so fail fast."""
    while not _finished.wait(1.0):
        for proc in _children:
            code = proc.poll()
            if code not in (None, 0):
                emit({'type': 'error', 'message': f'A training process (GPU rank) exited with code {code}'})
                for other in _children:
                    if other.poll() is None:
                        other.kill()
                os._exit(1)


def _watch_parent():
    """Extra ranks: leave when rank 0 is gone (e.g. killed), so no process keeps a GPU busy."""
    while True:
        time.sleep(2.0)
        if os.getppid() != _PARENT_PID:
            os._exit(1)


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
        from modules.distributed import plan_devices
        from modules.training import run_training

        if _RANK > 0:
            if _PARENT_PID:
                threading.Thread(target=_watch_parent, daemon=True).start()
        else:
            plan = plan_devices(config.get('options'))
            if len(plan) > 1 and 'WORLD_SIZE' not in os.environ:
                emit({'type': 'status', 'text': f'starting {len(plan)} GPU processes (DistributedDataParallel)'})
                _start_extra_ranks(args, len(plan))
                threading.Thread(target=_watch_extra_ranks, daemon=True).start()

        result = run_training(config, on_event=emit if _RANK == 0 else None, should_stop=should_stop)
    except BaseException as exc:  # noqa: BLE001 - report everything to the server
        traceback.print_exc()
        emit({'type': 'error', 'message': f'{type(exc).__name__}: {exc}', 'traceback': traceback.format_exc()})
        _finished.set()
        for proc in _children:
            if proc.poll() is None:
                proc.kill()
        return 1

    _finished.set()
    _stop_extra_ranks()
    emit({'type': 'done', 'stopped': bool(result.get('stopped')), 'result': result})
    return 0


if __name__ == '__main__':
    sys.exit(main())
