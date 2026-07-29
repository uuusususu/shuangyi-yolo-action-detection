"""Provide writable standard streams for PyInstaller windowed builds."""
from __future__ import annotations

import os
import sys


if sys.stdout is None:
    sys.stdout = open(os.devnull, "w", encoding="utf-8")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w", encoding="utf-8")
