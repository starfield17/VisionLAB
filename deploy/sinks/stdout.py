"""JSONL sink; stream ownership remains with the caller."""
import json
import sys
from typing import Any, TextIO


class StdoutSink:
    def __init__(self, stream: TextIO | None = None) -> None:
        self.stream = stream if stream is not None else sys.stdout
        self._opened = False

    def open(self) -> None:
        self._opened = True

    def write(self, event: dict[str, Any]) -> None:
        if not self._opened:
            raise RuntimeError('sink is not open')
        text = json.dumps(event, ensure_ascii=False, allow_nan=False, separators=(',', ':'))
        self.stream.write(text + '\n')
        self.stream.flush()

    def close(self) -> None:
        if self._opened:
            self._opened = False
            self.stream.flush()
