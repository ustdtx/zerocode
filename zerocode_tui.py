"""Interactive terminal UI for ZeroCode."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from zerocode import AgentError, CodeIndex, EmbeddingClient, LocalModel, TOOLS, TaskState, diff, execute, extract_json, initial_messages, validate_action, validate_phase


RESET = "\033[0m"
CYAN = "\033[36m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"
DIM = "\033[2m"


def colour(code: str, text: str) -> str:
    return f"{code}{text}{RESET}"


def banner(root: Path) -> None:
    print(colour(CYAN, "=" * 68))
    print(colour(CYAN, "  ZeroCode TUI  |  small-model local coding agent"))
    print(colour(CYAN, "=" * 68))
    print("  chat:     http://127.0.0.1:2247/v1")
    print("  retrieval:http://127.0.0.1:2248/v1")
    print(f"  repo:     {root}")
    print()


def compact_messages(messages: list[dict[str, str]], budget: int = 48000) -> list[dict[str, str]]:
    """Keep the system prompt and newest turns inside the 16K-token budget."""
    if not messages:
        return messages
    kept: list[dict[str, str]] = []
    used = len(messages[0].get("content", ""))
    for message in reversed(messages[1:]):
        size = len(message.get("content", ""))
        if kept and used + size > budget:
            break
        kept.append(message)
        used += size
    return [messages[0], *reversed(kept)]


def retrieved_context(index: CodeIndex, task: str) -> str:
    try:
        return index.retrieve(task, limit=6, max_chars=12000)
    except Exception as exc:
        return f"(retrieval unavailable: {exc})"
    print(colour(DIM, "  Type a task. Commands: /help /index /diff /clear /repo /quit"))
    print()


def run_task(root: Path, model: LocalModel, task: str, max_turns: int, messages: list[dict[str, str]] | None, index: CodeIndex) -> list[dict[str, str]]:
    if messages is None:
        messages = initial_messages(root, task)
        messages[1]["content"] += "\n\nHost-provided retrieved context (use tools to verify it):\n" + retrieved_context(index, task)
    else:
        messages.append({"role": "user", "content": f"New task: {task}\nInspect current files before editing.\n\nHost-provided retrieved context (use tools to verify it):\n{retrieved_context(index, task)}"})
    state = TaskState()
    repeated_actions: dict[str, int] = {}
    for turn in range(1, max_turns + 1):
        print(colour(DIM, f"\n[{turn}/{max_turns}] model is thinking..."))
        raw = ""
        name = ""
        try:
            messages = compact_messages(messages)
            raw = model.ask(messages)
            action = extract_json(raw)
            validate_action(action, state.inspected)
            validate_phase(action, state)
            name = action.get("action", "unknown")
            signature = json.dumps(action, sort_keys=True, ensure_ascii=True)
            repeated_actions[signature] = repeated_actions.get(signature, 0) + 1
            if repeated_actions[signature] >= 3:
                raise AgentError("You repeated the same action three times. Stop searching and make the smallest next patch, usually PLAN.md for an empty repository.")
            print(colour(YELLOW, f"  {state.label()} -> {name}"))
            if name == "finish":
                print(colour(GREEN, f"  {action.get('summary', 'Finished.')}"))
                result = "FINISHED: " + str(action.get("summary", ""))
            else:
                result = execute(root, action, index)
                if name in {"build_index", "repo_summary", "list_files", "search", "find_symbol", "context", "read_file"}:
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
            if name == "apply_patch":
                print(colour(GREEN, "  patch applied"))
            elif name == "run_check":
                print(colour(GREEN if "exit_code=0" in result else RED, result))
            elif name == "suggest_check":
                print(colour(YELLOW, result))
            else:
                print(colour(DIM, result[:5000]))
        except AgentError as exc:
            result = f"TOOL ERROR: {exc}"
            print(colour(RED, f"  {result}"))
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Tool result ({state.label()}):\n{result}\n\nContinue with exactly one JSON action."})
        if name == "finish":
            return messages
    print(colour(YELLOW, "Maximum turns reached. Use /diff to inspect the result."))
    return messages


def main() -> int:
    parser = argparse.ArgumentParser(description="ZeroCode interactive terminal UI")
    parser.add_argument("--repo", default=".")
    parser.add_argument("--api-key", default="sk-llama-local-key")
    parser.add_argument("--max-turns", type=int, default=16)
    args = parser.parse_args()

    root = Path(args.repo).resolve()
    if not root.is_dir():
        print(f"Repository does not exist: {root}", file=sys.stderr)
        return 2
    model = LocalModel(api_key=args.api_key)
    index = CodeIndex(root, EmbeddingClient(api_key=args.api_key))
    print(colour(DIM, "updating code index..."))
    print(colour(DIM, index.build()))
    messages = None
    banner(root)

    while True:
        try:
            task = input(colour(CYAN, "you> ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not task:
            continue
        if task in {"/quit", "/exit", "/q"}:
            return 0
        if task == "/help":
            print("  /index      build or update the repository index")
            print("  /diff       show current git diff")
            print("  /clear      clear conversation context")
            print("  /repo       show current repository")
            print("  /quit       exit")
            continue
        if task == "/index":
            print(colour(DIM, "building code index..."))
            print(colour(GREEN, index.build()))
            continue
        if task == "/diff":
            print(diff(root))
            continue
        if task == "/clear":
            print("\033[2J\033[H", end="")
            messages = None
            banner(root)
            continue
        if task == "/repo":
            print(f"  {root}")
            continue
        messages = run_task(root, model, task, args.max_turns, messages, index)


if __name__ == "__main__":
    raise SystemExit(main())
