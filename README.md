# ZeroCode

ZeroCode is a local coding harness designed around 4B-9B quantized models. The model proposes one bounded action at a time; the host process owns repository access, patch validation, and checks.

## Quick Start

Start an OpenAI-compatible llama.cpp server on port `2247` for coding and one on port `2248` for embeddings, then run:

```powershell
zerocode
```

The harness uses `http://127.0.0.1:2247/v1` for coding and `http://127.0.0.1:2248/v1` for embeddings. Model IDs are discovered from those servers automatically.

The harness supports repository summaries, regex search, symbol lookup, task-relevant context retrieval, bounded file reads, unified patches, automatic check discovery, focused checks, and final diffs. It intentionally does not expose unrestricted shell access. Editing is blocked until inspection, and completion is blocked until verification has been attempted.

## Interactive TUI

The primary executable is built from `zerocode_tui.py`:

```powershell
python -m PyInstaller --onefile --name zerocode zerocode_tui.py
```

Run `dist\zerocode.exe`. Type a task at `you>`. `/index`, `/diff`, `/clear`, `/repo`, and `/quit` are available at any time. `/index` explicitly builds or updates the repository index.

## `zerocode` Command

The primary executable is `dist\zerocode.exe`, built from the TUI. Add its directory to PATH if you want the command available globally. Open a terminal, change to any project directory, and run:

```powershell
zerocode
```

It uses the current directory as `--repo` and connects to port `2247` by default.

## Build A Windows EXE

Install PyInstaller once:

```powershell
python -m pip install -r requirements-build.txt
python -m PyInstaller --onefile --name zerocode zerocode_tui.py
```

The executable is written to `dist\zerocode.exe`. It connects to already-running local llama.cpp servers; model files are not bundled into the executable.

## Existing Local Models

The harness does not require model aliases in its UI. It discovers whichever model is exposed on port `2247` and whichever embedding model is exposed on port `2248`. On startup it performs an incremental index update; unchanged files and existing vectors are skipped. Use `/index` to force an update after making changes.
