"""ZeroCode: a small-model-first local coding agent.

The model is used for decisions; the host process owns repository access,
patch application, and verification.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import os
import re
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from zerocode_index import CodeIndex, EmbeddingClient


IGNORE_DIRS = {".git", ".venv", "venv", "node_modules", "bin", "obj", "dist", "build", "__pycache__"}
TEXT_EXTENSIONS = {
    ".c", ".cc", ".cpp", ".cs", ".css", ".go", ".h", ".hpp", ".html", ".java", ".js",
    ".json", ".jsx", ".md", ".py", ".rs", ".scss", ".sh", ".sql", ".swift", ".toml", ".ts",
    ".tsx", ".txt", ".xml", ".yaml", ".yml",
}
ALLOWED_CHECKS = ("pytest", "python ", "python3 ", "npm ", "npx ", "yarn ", "pnpm ", "dotnet ", "cargo ", "go ")


class AgentError(Exception):
    pass


@dataclass
class TaskState:
    phase: str = "INSPECT"
    inspected: bool = False
    changed: bool = False
    check_available: bool = False
    check_passed: bool = False
    check_attempted: bool = False

    def label(self) -> str:
        return (
            f"phase={self.phase} inspected={self.inspected} changed={self.changed} "
            f"check_available={self.check_available} check_passed={self.check_passed}"
        )


def safe_path(root: Path, value: str) -> Path:
    candidate = (root / value).resolve()
    try:
        candidate.relative_to(root.resolve())
    except ValueError as exc:
        raise AgentError(f"Path is outside repository: {value}") from exc
    return candidate


def display_path(root: Path, path: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def files_in(root: Path) -> list[Path]:
    result = []
    for path in root.rglob("*"):
        if any(part in IGNORE_DIRS for part in path.parts):
            continue
        if path.is_file() and (path.suffix.lower() in TEXT_EXTENSIONS or not path.suffix):
            result.append(path)
    return sorted(result)[:5000]


def detect_checks(root: Path) -> list[str]:
    checks: list[str] = []
    package = root / "package.json"
    if package.is_file():
        try:
            scripts = json.loads(package.read_text(encoding="utf-8")).get("scripts", {})
            for name in ("test", "typecheck", "lint", "build"):
                if name in scripts:
                    checks.append(f"npm run {name}")
        except (OSError, json.JSONDecodeError):
            pass
    if (root / "pyproject.toml").is_file() or (root / "pytest.ini").is_file() or (root / "tests").is_dir():
        checks.append("pytest -q")
    if list(root.glob("*.csproj")) or list(root.glob("*.sln")):
        checks.append("dotnet test")
    if (root / "Cargo.toml").is_file():
        checks.append("cargo test")
    if (root / "go.mod").is_file():
        checks.append("go test ./...")
    return list(dict.fromkeys(checks))


def repo_summary(root: Path) -> str:
    paths = files_in(root)
    checks = detect_checks(root)
    markers = [path.name for path in root.iterdir() if path.name in {
        "package.json", "pyproject.toml", "pytest.ini", "Cargo.toml", "go.mod", "README.md"
    }]
    return "\n".join([
        f"root: {root}",
        f"text_files: {len(paths)}",
        f"top_level_markers: {', '.join(sorted(markers)) or '(none)'}",
        f"suggested_checks: {', '.join(checks) or '(none detected)'}",
        "top_level_entries: " + ", ".join(sorted(path.name for path in root.iterdir())[:80]),
    ])


def list_files(root: Path) -> str:
    paths = [display_path(root, path) for path in files_in(root)]
    if len(paths) > 300:
        return "\n".join(paths[:300]) + f"\n... ({len(paths) - 300} more files; use search)"
    return "\n".join(paths) or "(repository has no text files)"


def find_symbol(root: Path, name: str, max_results: int = 40) -> str:
    if not name.strip():
        raise AgentError("Symbol name cannot be empty")
    pattern = re.compile(
        rf"\b(class|interface|struct|def|function|func|fn|enum|const|let|var)\s+{re.escape(name)}\b"
    )
    results: list[str] = []
    for path in files_in(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        for number, line in enumerate(lines, 1):
            if pattern.search(line):
                results.append(f"{display_path(root, path)}:{number}: {line[:300]}")
                if len(results) >= max_results:
                    return "\n".join(results)
    return "\n".join(results) or "(symbol not found; do not infer its definition)"


def relevant_context(root: Path, query: str, max_files: int = 6) -> str:
    words = [word for word in re.findall(r"[A-Za-z_][A-Za-z0-9_]{2,}", query) if word.lower() not in {
        "the", "and", "for", "with", "from", "that", "this", "into", "then", "must", "should"
    }]
    if not words:
        return "(no searchable task terms)"
    candidates: list[tuple[int, Path, list[str]]] = []
    for path in files_in(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        score = sum(sum(1 for word in words if word.lower() in line.lower()) for line in lines)
        if score:
            candidates.append((score, path, lines))
    sections: list[str] = []
    for _, path, lines in sorted(candidates, key=lambda item: (-item[0], str(item[1])))[:max_files]:
        hits = [index for index, line in enumerate(lines) if any(word.lower() in line.lower() for word in words)]
        if not hits:
            continue
        start = max(0, min(hits) - 8)
        end = min(len(lines), max(hits) + 9)
        excerpt = "\n".join(f"{number + 1:>5}: {lines[number]}" for number in range(start, end))
        sections.append(f"{display_path(root, path)}\n{excerpt}")
    return "\n\n".join(sections) or "(no relevant context found; use search)"


def read_file(root: Path, value: str, start: int = 1, end: int = 160) -> str:
    path = safe_path(root, value)
    if not path.is_file():
        raise AgentError(f"File does not exist: {value}")
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except UnicodeDecodeError as exc:
        raise AgentError(f"File is not UTF-8 text: {value}") from exc
    start = max(1, int(start))
    end = min(len(lines), max(start, int(end)))
    return "\n".join(f"{number:>5}: {lines[number - 1]}" for number in range(start, end + 1))


def search(root: Path, pattern: str, max_results: int = 50) -> str:
    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise AgentError(f"Invalid search pattern: {exc}") from exc
    matches: list[str] = []
    for path in files_in(root):
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            continue
        relative = display_path(root, path)
        for number, line in enumerate(lines, 1):
            if regex.search(line):
                matches.append(f"{relative}:{number}: {line[:300]}")
                if len(matches) >= max_results:
                    return "\n".join(matches)
    return "\n".join(matches) or "(no matches; do not infer a match)"


def run_check(root: Path, command: str) -> str:
    command = command.strip()
    if not command or not command.startswith(ALLOWED_CHECKS):
        raise AgentError("Check command is not allowed. Use pytest, python, npm, dotnet, cargo, or go commands.")
    completed = subprocess.run(command, cwd=root, shell=True, capture_output=True, text=True, timeout=180)
    output = (completed.stdout + "\n" + completed.stderr).strip()
    if len(output) > 8000:
        output = output[-8000:]
    return f"exit_code={completed.returncode}\n{output}"


def apply_patch(root: Path, patch: str) -> str:
    patch = re.sub(r"^```(?:diff)?\s*|\s*```$", "", patch.strip(), flags=re.IGNORECASE | re.DOTALL)
    if not patch.strip().startswith("diff ") and "*** Begin Patch" not in patch:
        raise AgentError("Expected a unified diff beginning with 'diff ' or an apply_patch block.")
    if "*** Begin Patch" in patch:
        patch = patch.replace("*** Begin Patch\n", "").replace("\n*** End Patch", "")
        patch = re.sub(r"\n\*\*\* Update File: ([^\n]+)", lambda m: f"\ndiff --git a/{m.group(1)} b/{m.group(1)}\n--- a/{m.group(1)}\n+++ b/{m.group(1)}", patch)
        patch = re.sub(r"\n\*\*\* Add File: ([^\n]+)", lambda m: f"\ndiff --git a/{m.group(1)} b/{m.group(1)}\n--- /dev/null\n+++ b/{m.group(1)}", patch)
        patch = patch.replace("\n*** Delete File:", "\ndiff --git")
    touched = set()
    for line in patch.splitlines():
        if line.startswith("+++ b/") or line.startswith("--- a/"):
            path = line[6:]
            safe_path(root, path)
            touched.add(path)
    if len(touched) > 5:
        raise AgentError("Patch touches more than five files. Split the change into smaller patches.")
    check = subprocess.run(["git", "apply", "--check", "--whitespace=nowarn", "-"], input=patch, text=True, cwd=root, capture_output=True)
    if check.returncode:
        raise AgentError(f"Patch rejected during validation: {check.stderr.strip()}")
    process = subprocess.run(["git", "apply", "--whitespace=nowarn", "-"], input=patch, text=True, cwd=root, capture_output=True)
    if process.returncode:
        raise AgentError(f"Patch rejected: {process.stderr.strip()}")
    return "Patch applied successfully.\n\n" + diff(root)


def diff(root: Path) -> str:
    process = subprocess.run(["git", "diff", "--", "."], cwd=root, capture_output=True, text=True)
    return process.stdout[-12000:] or "(no changes)"


TOOLS = """Available actions (return exactly one JSON object per turn):
{"action":"build_index"}
{"action":"repo_summary"}
{"action":"list_files"}
{"action":"search","pattern":"regex"}
{"action":"find_symbol","name":"SymbolName"}
{"action":"context","query":"terms from the task"}
{"action":"read_file","path":"relative/path","start":1,"end":200}
{"action":"suggest_check"}
{"action":"apply_patch","patch":"unified git diff"}
{"action":"run_check","command":"pytest -q"}
{"action":"show_diff"}
{"action":"finish","summary":"what changed and what was verified"}

Rules:
- Your only source of truth is tool output. Never invent files, symbols, APIs, errors, or test results.
- Start every task by inspecting the repository. Do not apply a patch until you have read relevant files.
- Use the smallest patch possible and modify only files relevant to the task.
- Apply one coherent patch, then run the narrowest relevant check.
- If a tool says no match or a file is missing, change strategy; do not guess.
- Do not claim success unless a check passed or you clearly state that no check exists.
- Do not explain your reasoning. Return one JSON action only, with no markdown and no extra keys.
- Phase order is INSPECT -> EDIT -> VERIFY -> DONE. The host enforces this order.
"""


def initial_messages(root: Path, task: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are ZeroCode, a precise coding agent operating on an existing repository.\n"
                f"Repository root: {root}\n"
                "The host, not you, controls files and commands. You must use tools for facts.\n"
                + TOOLS
            ),
        },
        {"role": "user", "content": f"New task: {task}\nFirst inspect the repository before proposing any edit."},
    ]


def extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    candidates = [text]
    candidates.extend(re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL))
    decoder = json.JSONDecoder()
    candidates.extend(text[index:] for index, char in enumerate(text) if char == "{")
    for candidate in candidates:
        try:
            value, _ = decoder.raw_decode(candidate.lstrip())
            if isinstance(value, dict) and "action" in value:
                return value
        except json.JSONDecodeError:
            continue
    raise AgentError(f"Model did not return a valid action: {text[:1000]}")


def validate_action(action: dict[str, Any], inspected: bool) -> None:
    allowed = {"build_index", "repo_summary", "list_files", "search", "find_symbol", "context", "read_file", "suggest_check", "apply_patch", "run_check", "show_diff", "finish"}
    name = action.get("action")
    if name not in allowed:
        raise AgentError(f"Invalid action '{name}'. Choose one of: {', '.join(sorted(allowed))}")
    if name == "apply_patch" and not inspected:
        raise AgentError("Inspection required before editing. Use list_files, search, or read_file first.")
    if name in {"search", "find_symbol", "context", "read_file", "apply_patch", "run_check"} and not isinstance(action.get({
        "search": "pattern", "find_symbol": "name", "context": "query", "read_file": "path", "apply_patch": "patch", "run_check": "command"
    }[name]), str):
        raise AgentError(f"Action '{name}' has a missing or invalid argument.")


def validate_phase(action: dict[str, Any], state: TaskState) -> None:
    name = action.get("action")
    if name == "apply_patch" and not state.inspected:
        raise AgentError("Phase violation: inspect the repository before editing.")
    if name == "finish" and state.changed and not state.check_attempted:
        raise AgentError("Phase violation: run or inspect a verification check before finishing.")
    if name == "finish" and state.changed and state.check_available and not state.check_passed:
        raise AgentError("Verification is available but has not passed. Repair the failure before finishing.")


class LocalModel:
    def __init__(self, endpoint: str = "http://127.0.0.1:2247/v1", model: str | None = None, api_key: str = "sk-llama-local-key", timeout: int = 600):
        self.base = endpoint.rstrip("/")
        self.endpoint = self.base + "/chat/completions"
        self.model = model
        self.api_key = api_key
        self.timeout = timeout

    def _model_id(self) -> str:
        if self.model:
            return self.model
        request = urllib.request.Request(self.base + "/models", headers={"Authorization": f"Bearer {self.api_key}"})
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                models = json.loads(response.read()).get("data", [])
            if models and models[0].get("id"):
                self.model = models[0]["id"]
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError):
            pass
        return self.model or "local"

    def ask(self, messages: list[dict[str, str]]) -> str:
        payload = {
            "model": self._model_id(),
            "messages": messages,
            "temperature": 0.1,
            "top_p": 0.9,
            "max_tokens": 8192,
            "reasoning_effort": "low",
            "stream": False,
        }
        request = urllib.request.Request(self.endpoint, data=json.dumps(payload).encode(), headers={
            "Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}",
        })
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                body = json.loads(response.read())
        except (urllib.error.URLError, TimeoutError) as exc:
            raise AgentError(f"Could not reach model at {self.endpoint}: {exc}") from exc
        try:
            message = body["choices"][0]["message"]
            content = message.get("content")
            if content:
                return content
            reasoning = message.get("reasoning_content")
            if reasoning:
                return reasoning
            raise KeyError("message.content")
        except (KeyError, IndexError, TypeError) as exc:
            raise AgentError(f"Unexpected model response: {json.dumps(body)[:1000]}") from exc


def execute(root: Path, action: dict[str, Any], index: CodeIndex | None = None) -> str:
    name = action.get("action")
    if name == "build_index":
        return (index or CodeIndex(root)).build()
    if name == "repo_summary":
        return repo_summary(root)
    if name == "list_files":
        return list_files(root)
    if name == "search":
        return search(root, str(action.get("pattern", "")))
    if name == "find_symbol":
        return index.symbols(str(action.get("name", ""))) if index else find_symbol(root, str(action.get("name", "")))
    if name == "context":
        return index.retrieve(str(action.get("query", ""))) if index else relevant_context(root, str(action.get("query", "")))
    if name == "read_file":
        return read_file(root, str(action.get("path", "")), action.get("start", 1), action.get("end", 160))
    if name == "apply_patch":
        return apply_patch(root, str(action.get("patch", "")))
    if name == "run_check":
        return run_check(root, str(action.get("command", "")))
    if name == "show_diff":
        return diff(root)
    if name == "suggest_check":
        checks = detect_checks(root)
        return "suggested_checks=" + (", ".join(checks) if checks else "(none detected)")
    if name == "finish":
        return "FINISHED: " + str(action.get("summary", ""))
    raise AgentError(f"Unknown action: {name}")


def run_agent(args: argparse.Namespace) -> int:
    root = Path(args.repo).resolve()
    if not root.is_dir():
        print(f"Repository does not exist: {root}", file=sys.stderr)
        return 2
    model = LocalModel(args.endpoint, args.model, args.api_key)
    system = "You are ZeroCode, a precise local coding agent for small language models.\n" + TOOLS
    messages: list[dict[str, str]] = [{"role": "system", "content": system}, {"role": "user", "content": args.task}]
    print(f"Repository: {root}\nModel: {args.model}\n")
    state = TaskState()
    for turn in range(1, args.max_turns + 1):
        print(f"[{turn}/{args.max_turns}] thinking...")
        raw = ""
        try:
            raw = model.ask(messages)
            action = extract_json(raw)
            validate_action(action, inspected=state.inspected)
            validate_phase(action, state)
            print(f"  action: {action.get('action')}")
            if action.get("action") == "finish":
                print(action.get("summary", "Finished."))
                return 0
            result = execute(root, action)
            name = action.get("action")
            if name in {"repo_summary", "list_files", "search", "find_symbol", "context", "read_file"}:
                state.inspected = True
            if name == "suggest_check":
                state.check_attempted = True
                state.check_available = "(none detected)" not in result
            if name == "apply_patch":
                state.changed = True
                state.phase = "VERIFY"
                state.check_attempted = False
                state.check_available = False
                state.check_passed = False
            if name == "run_check":
                state.check_attempted = True
                state.check_available = True
                state.check_passed = "exit_code=0" in result
                state.phase = "DONE" if state.check_passed else "VERIFY"
        except AgentError as exc:
            result = f"TOOL ERROR: {exc}"
            print(f"  {result}")
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Tool result ({state.label()}):\n{result}\n\nContinue with exactly one JSON action."})
    print("Maximum turns reached. Review the working-tree diff.", file=sys.stderr)
    return 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Small-model-first local coding agent")
    parser.add_argument("task", help="Coding task in plain language")
    parser.add_argument("--repo", default=".", help="Repository to edit")
    parser.add_argument("--endpoint", default=os.environ.get("ZEROCODE_ENDPOINT", "http://127.0.0.1:2247/v1"))
    parser.add_argument("--model", default=os.environ.get("ZEROCODE_MODEL", "OxCoder-9B"))
    parser.add_argument("--api-key", default=os.environ.get("ZEROCODE_API_KEY", "sk-llama-local-key"))
    parser.add_argument("--max-turns", type=int, default=16)
    return run_agent(parser.parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
