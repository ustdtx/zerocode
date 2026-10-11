# ZeroCode

ZeroCode is a local coding harness designed around 4B-9B quantized models. The model proposes one bounded action at a time; the host process owns repository access, patch validation, and checks.

## Quick Start

Start an OpenAI-compatible llama.cpp server on port `2247` for coding and one on port `2248` for embeddings, then run:

```powershell
zerocode
```

The harness uses `http://127.0.0.1:2247/v1` for coding and `http://127.0.0.1:2248/v1` for embeddings. Model IDs are discovered from those servers automatically.

## Start The Model Servers

The repository does not include model weights or llama.cpp binaries. Start two local llama.cpp servers before running ZeroCode. The following profile is tuned for an 8 GB GPU and keeps measured usage below 7.3 GB on the development machine.

Coding server on port `2247`:

```powershell
llama-server.exe `
  -m C:\models\OxCoder-9B.i1-Q4_K_M.gguf `
  --alias OxCoder-9B `
  -ngl 99 -c 16384 -fa on `
  -ctk q4_0 -ctv q4_0 -np 1 `
  --jinja --reasoning on `
  --host 127.0.0.1 --port 2247 `
  --api-key sk-llama-local-key
```

Embedding server on port `2248`:

```powershell
llama-server.exe `
  -m C:\models\Qwen3-Embedding-0.6B-Q8_0.gguf `
  --alias Qwen3-Embedding-0.6B `
  --embedding --pooling mean --embd-normalize 2 `
  --ctx-size 2048 --parallel 1 `
  --batch-size 2048 --ubatch-size 2048 `
  --gpu-layers 0 `
  --host 127.0.0.1 --port 2248 `
  --api-key sk-llama-local-key
```

The embedding server intentionally runs on CPU. It is used during indexing and for small retrieval queries; keeping it GPU-resident competes with OxCoder's model and KV cache. The `/index` command can therefore run while the coding server remains loaded.

If your local server uses a different API key, pass it to ZeroCode with `--api-key` or set `ZEROCODE_API_KEY`.

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

The executable directory must be on PATH for `zerocode` to work from every folder. A clone does not modify PATH automatically.

## Build A Windows EXE

Install PyInstaller once:

```powershell
python -m pip install -r requirements-build.txt
python -m PyInstaller --onefile --name zerocode zerocode_tui.py
```

The executable is written to `dist\zerocode.exe`. It connects to already-running local llama.cpp servers; model files are not bundled into the executable.

After starting both servers, open a project directory and run `zerocode`. ZeroCode incrementally updates its index on startup. Use `/index` after source changes to update it manually. The index is stored in `.zerocode\index.sqlite3`, which is intentionally ignored by Git.

## Existing Local Models

The harness does not require model aliases in its UI. It discovers whichever model is exposed on port `2247` and whichever embedding model is exposed on port `2248`. On startup it performs an incremental index update; unchanged files and existing vectors are skipped. Use `/index` to force an update after making changes.
