# Repo notes for Claude

## generic_core/examples/model_builder_notebook.py is a generated file

It is assembled from section files (`_nb_*.py` in `generic_core/examples/`) by
`build_notebook.py`. **Never hand-edit `model_builder_notebook.py` directly** —
edits there are silently lost the next time someone runs `build_notebook.py`.

Workflow:
- Editing notebook behavior: edit the relevant `_nb_*.py` section file, then run
  `python generic_core/examples/build_notebook.py` to regenerate
  `model_builder_notebook.py`. Do this regeneration as part of the same task —
  don't leave the two out of sync.
- If cells were changed live in the marimo browser UI (not by editing
  `_nb_*.py`), run `python generic_core/examples/split_notebook.py` FIRST to
  pull those changes back into the section files, before making further edits
  or running `build_notebook.py` — otherwise `build_notebook.py` will overwrite
  the UI-made changes.
- To verify the two are in sync at any time (e.g. before committing), run:
  `python generic_core/examples/check_notebook_sync.py`

A pre-commit hook (`.githooks/pre-commit`) enforces this automatically: it
runs `check_notebook_sync.py` and blocks the commit if
`model_builder_notebook.py` and `_nb_*.py` have drifted apart. `core.hooksPath`
is a local git config setting, not something checked into the repo, so it
does not apply automatically on a fresh clone — enable it once per clone with:

    git config core.hooksPath .githooks
