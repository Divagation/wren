#!/usr/bin/env python3
"""Wren MCP server -- compressed tool output for MCP-capable coding agents."""

import os
import sys
import subprocess
import shlex
import shutil

# Ensure wren.py is importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _find_rg() -> str:
    """Find the ripgrep binary, checking common install locations."""
    found = shutil.which("rg")
    if found:
        return found
    for candidate in ["/opt/homebrew/bin/rg", "/usr/local/bin/rg"]:
        if os.path.isfile(candidate):
            return candidate
    return "rg"  # fall back and let subprocess raise

from mcp.server.fastmcp import FastMCP
from wren import compress_with_stats, get_stats, is_loaded

mcp = FastMCP("wren")

SAFE_EXECUTABLES = {
    "git",
    "pytest",
    "cargo",
    "go",
    "swift",
    "xcodebuild",
    "cmake",
    "ctest",
    "make",
    "just",
    "npm",
    "pnpm",
    "yarn",
    "bun",
    "kubectl",
    "docker",
    "docker-compose",
    "podman",
    "rg",
    "find",
    "fd",
    "ls",
    "tree",
    "du",
    "wc",
    "head",
    "tail",
    "cat",
    "sed",
    "pwd",
}

SAFE_GIT_SUBCOMMANDS = {
    "status",
    "diff",
    "show",
    "log",
    "rev-parse",
    "ls-files",
    "grep",
    "blame",
    "shortlog",
}

SAFE_JS_ACTIONS = {"test", "build", "lint", "check"}
SAFE_MAKE_TARGETS = {"test", "build", "check", "lint"}
SAFE_KUBECTL_SUBCOMMANDS = {"get", "describe", "logs", "top"}
SAFE_DOCKER_SUBCOMMANDS = {"logs", "ps", "images", "inspect"}


def _validate_exec_command(command: str) -> tuple[list[str] | None, str | None]:
    """Parse a command and reject arbitrary or destructive invocations."""
    try:
        argv = shlex.split(command)
    except ValueError as e:
        return None, f"Error: invalid command syntax: {e}"

    if not argv:
        return None, "Error: empty command"

    exe = os.path.basename(argv[0])
    if exe not in SAFE_EXECUTABLES:
        return None, (
            "Error: compressed_exec only supports a limited set of read/build/test/log commands. "
            "Use the native terminal tool for arbitrary commands."
        )

    if exe == "git":
        # Skip flags (e.g. -C /path) to find the actual subcommand
        git_sub = None
        i = 1
        while i < len(argv):
            arg = argv[i]
            if not arg.startswith("-"):
                git_sub = arg
                break
            # Flags that take a value: skip the next arg too
            if arg in ("-C", "-c", "--git-dir", "--work-tree"):
                i += 1
            i += 1
        if git_sub not in SAFE_GIT_SUBCOMMANDS:
            return None, "Error: compressed_exec only allows read-only git subcommands."

    if exe in {"npm", "pnpm", "yarn", "bun"}:
        if len(argv) < 2:
            return None, f"Error: {exe} requires an action."
        action = argv[1]
        if action == "run":
            if len(argv) < 3 or argv[2] not in SAFE_JS_ACTIONS:
                return None, f"Error: {exe} run only allows test/build/lint/check."
        elif action not in SAFE_JS_ACTIONS:
            return None, f"Error: {exe} only allows test/build/lint/check."

    if exe in {"make", "just"}:
        targets = [arg for arg in argv[1:] if not arg.startswith("-")]
        if not targets:
            return None, f"Error: {exe} requires an explicit target."
        if any(target not in SAFE_MAKE_TARGETS for target in targets):
            return None, f"Error: {exe} only allows test/build/check/lint targets."

    if exe == "kubectl":
        if len(argv) < 2 or argv[1] not in SAFE_KUBECTL_SUBCOMMANDS:
            return None, "Error: kubectl is limited to get/describe/logs/top."

    if exe in {"docker", "docker-compose", "podman"}:
        if len(argv) < 2 or argv[1] not in SAFE_DOCKER_SUBCOMMANDS:
            return None, "Error: container commands are limited to logs/ps/images/inspect."

    return argv, None


def _combine_process_output(result: subprocess.CompletedProcess[str]) -> str:
    parts = []
    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if stdout:
        parts.append(stdout)
    if stderr:
        parts.append(f"[stderr]\n{stderr}")
    if result.returncode != 0:
        parts.append(f"[exit: {result.returncode}]")

    return "\n".join(parts)


@mcp.tool()
def compressed_read(path: str, offset: int = 0, limit: int = 2000) -> str:
    """Read a file and return Wren-compressed content. Use for exploring unfamiliar
    code when you need the gist, not exact lines. Do NOT use for files you plan to
    edit -- use native Read for that since Edit needs exact string matching."""
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return f"Error: {path} not found"

    with open(path) as f:
        lines = f.readlines()

    total = len(lines)
    selected = lines[offset : offset + limit]
    content = "".join(selected)

    result, stats = compress_with_stats(content, mode="output")

    meta = f"[{path} | {total} lines | {offset + 1}-{min(offset + limit, total)}]"
    if stats.get("compressed"):
        meta += f" [wren: {stats['savings']} saved]"

    return f"{meta}\n{result}"


@mcp.tool()
def compressed_grep(
    pattern: str,
    path: str = ".",
    glob_filter: str = "",
    max_results: int = 50,
) -> str:
    """Search for a regex pattern and return Wren-compressed results.
    Use for broad exploration where you need the gist, not exact line numbers."""
    path = os.path.expanduser(path)

    cmd = [_find_rg(), "--no-heading", "-n", "-m", str(max_results), pattern, path]
    if glob_filter:
        cmd.extend(["--glob", glob_filter])

    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except Exception as e:
        return f"Error: {e}"

    if r.returncode == 2:
        return f"Error: {r.stderr.strip() or 'rg failed'}"

    output = r.stdout.strip()
    if not output and r.returncode == 1:
        return "No matches found"
    if not output:
        return f"Error: {r.stderr.strip() or f'rg exited with code {r.returncode}'}"

    result, stats = compress_with_stats(output, mode="output")

    meta = f"[grep: /{pattern}/]"
    if stats.get("compressed"):
        meta += f" [wren: {stats['savings']} saved]"

    return f"{meta}\n{result}"


@mcp.tool()
def compressed_exec(command: str) -> str:
    """Run a constrained command and return Wren-compressed output. Use for verbose
    read/build/test/log output, not arbitrary shell access."""
    argv, error = _validate_exec_command(command)
    if error:
        return error

    try:
        r = subprocess.run(argv, capture_output=True, text=True, timeout=120)
    except subprocess.TimeoutExpired:
        return "Error: timed out (120s)"
    except Exception as e:
        return f"Error: {e}"

    output = _combine_process_output(r)
    if not output.strip():
        return "(no output)"

    result, stats = compress_with_stats(output, mode="output")

    meta = f"[exec: {command}]"
    if stats.get("compressed"):
        meta += f" [wren: {stats['savings']} saved]"

    return f"{meta}\n{result}"


@mcp.tool()
def compress_text(text: str, mode: str = "input") -> str:
    """Compress arbitrary text through Wren. General-purpose -- use to shrink
    any large text blob before working with it. mode='input' for prose/instructions,
    mode='output' for code/tool results."""
    result, stats = compress_with_stats(text, mode=mode)
    if stats.get("compressed"):
        return f"[wren: {stats['savings']} saved | {stats['original']}->{stats['result']} chars]\n{result}"
    return result


@mcp.tool()
def wren_status() -> str:
    """Wren MCP server status: model state and session compression stats."""
    loaded = is_loaded()
    s = get_stats()

    lines = [f"Model loaded: {loaded}", f"Compressions: {s['calls']}"]

    if s["calls"] > 0:
        ratio = 1 - (s["chars_out"] / s["chars_in"]) if s["chars_in"] else 0
        lines.extend([
            f"Chars in: {s['chars_in']:,}",
            f"Chars out: {s['chars_out']:,}",
            f"Chars saved: {s['chars_in'] - s['chars_out']:,} ({ratio:.0%})",
        ])

    return "\n".join(lines)


def main() -> None:
    """CLI entrypoint for packaged installs."""
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
