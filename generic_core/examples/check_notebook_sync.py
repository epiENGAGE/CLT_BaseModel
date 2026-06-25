"""
check_notebook_sync.py
=======================

Verifies that model_builder_notebook.py matches what build_notebook.py would
produce from the current _nb_*.py section files — i.e. that the two have not
drifted out of sync (e.g. because model_builder_notebook.py was hand-edited
or edited via the marimo UI without running split_notebook.py afterward).

Run from the repo root:

    python generic_core/examples/check_notebook_sync.py

Exits 0 and prints "in sync" if they match; exits 1 and prints a diff
otherwise. Does not modify model_builder_notebook.py.
"""

import difflib
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).parent
NB = HERE / "model_builder_notebook.py"

sys.path.insert(0, str(HERE))
import build_notebook  # noqa: E402


def main() -> int:
    actual = NB.read_text(encoding="utf-8") if NB.exists() else ""

    with tempfile.TemporaryDirectory() as tmp:
        rebuilt_path = Path(tmp) / "model_builder_notebook.py"
        original_out = build_notebook.OUT
        build_notebook.OUT = rebuilt_path
        try:
            build_notebook.build()
        finally:
            build_notebook.OUT = original_out
        expected = rebuilt_path.read_text(encoding="utf-8")

    if actual == expected:
        print("in sync")
        return 0

    diff = difflib.unified_diff(
        actual.splitlines(keepends=True),
        expected.splitlines(keepends=True),
        fromfile="model_builder_notebook.py (current)",
        tofile="model_builder_notebook.py (rebuilt from _nb_*.py)",
    )
    sys.stdout.writelines(diff)
    print(
        "\nOUT OF SYNC — run `python generic_core/examples/build_notebook.py` "
        "(after split_notebook.py first if the drift came from UI edits)."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
