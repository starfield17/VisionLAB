"""Explicit subprocess runner with bounded runtime and available-memory guard.

Only executes the command following --. No automatic retry or model selection.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import psutil


def terminate_tree(process):
    try:
        parent = psutil.Process(process.pid)
        children = parent.children(recursive=True)
        for child in children:
            child.terminate()
        parent.terminate()
        _, alive = psutil.wait_procs([*children, parent], timeout=5)
        for child in alive:
            child.kill()
    except psutil.NoSuchProcess:
        pass
    process.wait(timeout=10)


def run(command: list[str], output: Path, *, minimum_available: int = 3 * 1024**3,
        timeout: float = 1800, interval: float = 0.25) -> int:
    if not command or minimum_available < 0 or timeout <= 0 or interval <= 0:
        raise ValueError('command and positive timing limits required')
    output.mkdir(parents=True, exist_ok=False)
    record = {'status': 'not_started', 'peak_process_tree_rss_bytes': 0,
              'minimum_system_available_bytes': psutil.virtual_memory().available,
              'minimum_available_limit_bytes': minimum_available, 'timeout_seconds': timeout}
    if record['minimum_system_available_bytes'] < minimum_available:
        record['status'] = 'memory_limit_before_start'
        (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
        return 1
    started = time.monotonic()
    process = None
    try:
        with (output / 'stdout.log').open('wb') as stdout, (output / 'stderr.log').open('wb') as stderr:
            process = subprocess.Popen(command, stdout=stdout, stderr=stderr)
            record['status'] = 'running'
            while process.poll() is None:
                available = psutil.virtual_memory().available
                record['minimum_system_available_bytes'] = min(record['minimum_system_available_bytes'], available)
                try:
                    parent = psutil.Process(process.pid)
                    rss = sum(p.memory_info().rss for p in [parent, *parent.children(recursive=True)])
                    record['peak_process_tree_rss_bytes'] = max(record['peak_process_tree_rss_bytes'], rss)
                except psutil.NoSuchProcess:
                    pass
                if available < minimum_available or time.monotonic() - started > timeout:
                    record['status'] = 'memory_limit' if available < minimum_available else 'timeout'
                    terminate_tree(process)
                    break
                time.sleep(interval)
            record['exit_code'] = process.wait()
            if record['status'] == 'running':
                record['status'] = 'success' if process.returncode == 0 else 'command_failed'
    except BaseException:
        record['status'] = 'interrupted_or_start_failed'
        if process is not None and process.poll() is None:
            terminate_tree(process)
        raise
    finally:
        record['elapsed_seconds'] = time.monotonic() - started
        (output / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    return 0 if record['status'] == 'success' else 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--timeout', type=float, default=1800)
    parser.add_argument('--minimum-available-gib', type=float, default=3)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    return run(command, args.output, minimum_available=int(args.minimum_available_gib * 1024**3), timeout=args.timeout)


if __name__ == '__main__':
    raise SystemExit(main())
