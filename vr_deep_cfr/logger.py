"""Small in-memory logger implementing the interface used by DeepPDCFR."""

from __future__ import annotations

import logging


class Logger:
    def __init__(self, writer_strings=None, *, verbose: bool = False):
        del writer_strings
        self._pending = {}
        self.history = []
        self._logger = logging.getLogger("vr_deep_cfr")
        self._verbose = bool(verbose)

    def record(self, key, value):
        self._pending[key] = value

    def dump(self, step=None):
        row = dict(self._pending)
        if step is not None:
            row["step"] = step
        self.history.append(row)
        self._pending.clear()
        return row

    def info(self, message):
        if self._verbose:
            self._logger.info(message)

    def warn(self, message):
        self._logger.warning(message)
