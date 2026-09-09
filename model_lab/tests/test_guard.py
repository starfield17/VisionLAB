import json
import sys
from types import SimpleNamespace

import pytest

from model_lab.guard import run


def test_guard_success_and_logs(tmp_path):
    output = tmp_path / 'run'
    assert run([sys.executable, '-c', 'print("done")'], output, minimum_available=0) == 0
    assert (output / 'stdout.log').read_text().strip() == 'done'
    assert json.loads((output / 'result.json').read_text())['status'] == 'success'


def test_guard_stops_before_launch_on_memory(tmp_path, monkeypatch):
    monkeypatch.setattr('model_lab.guard.psutil.virtual_memory', lambda: SimpleNamespace(available=1))
    output = tmp_path / 'run'
    assert run(['nonexistent-program'], output, minimum_available=2) == 1
    assert json.loads((output / 'result.json').read_text())['status'] == 'memory_limit_before_start'


def test_guard_timeout_and_failed_command(tmp_path):
    output = tmp_path / 'timeout'
    assert run([sys.executable, '-c', 'import time; time.sleep(30)'], output,
               minimum_available=0, timeout=.1, interval=.02) == 1
    assert json.loads((output / 'result.json').read_text())['status'] == 'timeout'
    failed = tmp_path / 'failed'
    assert run([sys.executable, '-c', 'raise SystemExit(7)'], failed, minimum_available=0) == 1
    assert json.loads((failed / 'result.json').read_text())['exit_code'] == 7


def test_guard_refuses_output_reuse(tmp_path):
    with pytest.raises(FileExistsError):
        run(['anything'], tmp_path)
