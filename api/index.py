from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    path_string = str(path)
    if path_string not in sys.path:
        sys.path.insert(0, path_string)

from app.flask_api import app
