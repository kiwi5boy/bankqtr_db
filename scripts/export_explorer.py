"""Build the variable explorer as a static, self-contained WASM site.

The explorer is a marimo app.  Run locally it is a Python process reading
``data/out``; exported here it is a directory of static files that runs the
same notebook in the browser under Pyodide, with no Python installed and no
server beyond a plain file host.

Two things have to be true for that to work, and this script is what makes
the second one true:

*   The notebook's own modules have to travel with it.  marimo handles this
    on its own -- it packages ``apps/explore.py`` and ``apps/viz.py`` as
    wheels under ``public/wheels`` and installs them into the browser kernel.
*   The panels have to travel with it.  Nothing does that automatically, so
    they are copied to ``public/data`` here and the notebook's first cell
    points the loaders at that URL when it detects it is running on WASM.

The export must be *served*, not opened off disk: a ``file://`` page cannot
fetch its own wheels.  The last line printed is the command to serve it.

    uv run --group app python scripts/export_explorer.py
    uv run --group app python scripts/export_explorer.py --out docs/explorer
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NOTEBOOK = ROOT / "apps" / "explorer.py"

# What the explorer reads.  Kept in step with ``apps/explore.py`` -- a name
# added there and not here exports a site that fails on first use.
DATA = (
    "panel.parquet",
    "panel_coverage.csv",
    "call_panel.parquet",
    "call_panel_coverage.csv",
)
SOURCES = (ROOT / "data" / "out", ROOT / "docs" / "data")


def _source(name: str) -> Path:
    for base in SOURCES:
        candidate = base / name
        if candidate.exists():
            return candidate
    raise SystemExit(
        f"{name} not found in {' or '.join(str(b) for b in SOURCES)} -- "
        "build the panels first, or check out docs/data."
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", type=Path, default=ROOT / "build" / "explorer",
        help="output directory (default: build/explorer)",
    )
    args = ap.parse_args()
    out: Path = args.out

    # Resolve every input before the export runs, so a missing panel fails
    # here rather than after a minute of bundling.
    sources = [(_source(name), name) for name in DATA]

    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable, "-m", "marimo", "export", "html-wasm",
            str(NOTEBOOK), "-o", str(out), "--mode", "run", "-f",
            # The notebook carries inline PEP 723 metadata for the browser
            # kernel; without this marimo stops to ask whether to build a
            # sandbox venv from it, which is not what the block is for.
            "--no-sandbox",
        ],
        check=True,
    )

    # After the export: it owns the directory and would clear this first.
    data_dir = out / "public" / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    for path, name in sources:
        shutil.copy2(path, data_dir / name)
        print(f"  staged {name}  ({path.stat().st_size / 1e6:.1f} MB)")

    print(f"\nServe it:\n  python -m http.server --directory {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
