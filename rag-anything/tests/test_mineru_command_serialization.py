from __future__ import annotations

import io
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from raganything.parser import MineruParser


class _FakePipe(io.StringIO):
    def close(self):
        return None


class _FakeProcess:
    active = 0
    max_active = 0
    lock = threading.Lock()

    def __init__(self, *args, **kwargs):
        self.stdout = _FakePipe("")
        self.stderr = _FakePipe("")
        self._start = time.monotonic()
        self._done = False
        with self.lock:
            type(self).active += 1
            type(self).max_active = max(type(self).max_active, type(self).active)

    def _finish_if_ready(self):
        if not self._done and time.monotonic() - self._start >= 0.2:
            self._done = True
            with self.lock:
                type(self).active -= 1

    def poll(self):
        self._finish_if_ready()
        return 0 if self._done else None

    def wait(self):
        while self.poll() is None:
            time.sleep(0.01)
        return 0


def test_mineru_commands_are_serialized_by_default(monkeypatch, tmp_path):
    monkeypatch.setenv("RAGANYTHING_SERIALIZE_MINERU", "true")
    monkeypatch.setattr("raganything.parser.subprocess.Popen", _FakeProcess)
    _FakeProcess.active = 0
    _FakeProcess.max_active = 0

    input_file = tmp_path / "sample.pdf"
    input_file.write_text("pdf", encoding="utf-8")
    output_dir = tmp_path / "out"

    def run_once():
        MineruParser._run_mineru_command(input_file, output_dir)

    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda _: run_once(), range(2)))

    assert _FakeProcess.max_active == 1
