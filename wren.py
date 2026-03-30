#!/usr/bin/env python3
"""Wren -- prompt and tool output compression. Pipe text in, compressed text out."""

import sys
import os
import json
import platform
import importlib.util
import re

# Suppress mlx warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

WREN_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(WREN_DIR, "config.json")
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

PROMPT_INPUT = (
    "You are Wren, a prompt compression model. Compress the input to its shortest "
    "form while preserving all meaning and instruction-following behavior. "
    "Output only the compressed text."
)
PROMPT_OUTPUT = (
    "You are Wren, a tool output compression model. Compress the tool output to its "
    "shortest form while preserving all actionable information: file paths, line numbers, "
    "error messages, function signatures, variable names, status codes, and structural "
    "relationships. Output only the compressed text."
)

MIN_CHARS = 100
MIN_SAVINGS = 0.20
TOKEN_ESTIMATE_CHARS = 4
DEMO_TEXT = [
    {
        "title": "Claude Code system prompt",
        "mode": "input",
        "text": (
            "Before making any changes to the codebase, please read the relevant files "
            "first so you understand the existing structure and patterns. Do not create "
            "new files unless they are clearly necessary. Preserve exact file paths, line "
            "numbers, error messages, negations, and step ordering when summarizing tool "
            "output. If you are unsure whether a command is destructive, stop and ask "
            "before proceeding."
        ),
        "compressed": (
            "Read relevant files first. Do not create new files unless clearly necessary. "
            "Preserve exact paths, line numbers, errors, negations, and step order. "
            "Ask before destructive commands."
        ),
    },
    {
        "title": "Tool output / build log",
        "mode": "output",
        "text": (
            "src/server.py:184: ValueError: invalid status code 299\n"
            "tests/test_api.py:88: AssertionError: expected 201, got 500\n"
            "Command failed: pytest tests/test_api.py -k create_user --maxfail=1\n"
            "Hint: check config.json and .env before retrying.\n"
            "Next steps:\n"
            "1. restore POST /users success path\n"
            "2. keep --maxfail=1 for fast iteration\n"
            "3. re-run pytest tests/test_api.py -k create_user\n"
        ),
        "compressed": (
            "src/server.py:184 ValueError invalid status code 299. "
            "tests/test_api.py:88 expected 201, got 500. "
            "pytest tests/test_api.py -k create_user --maxfail=1 failed. "
            "Check config.json and .env. Next: 1) restore POST /users success path. "
            "2) keep --maxfail=1. 3) re-run pytest tests/test_api.py -k create_user."
        ),
    },
]

# --- Model singleton ---

_model = None
_tokenizer = None
_stats = {"calls": 0, "chars_in": 0, "chars_out": 0}


def load_config() -> dict:
    """Load config.json if present."""
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            return json.load(f)
    return {}


def get_base_model() -> str:
    """Resolve base model from config.json, env var, or project default."""
    cfg = load_config()
    model = cfg.get("base_model", "").strip()
    if model and model != "your-base-model-here":
        return model
    env = os.environ.get("WREN_BASE_MODEL", "")
    if env:
        return env
    return DEFAULT_BASE_MODEL


def get_adapter_path() -> str:
    """Resolve adapter path from config.json or project default."""
    cfg = load_config()
    adapter_path = cfg.get("adapter_path", "adapters")
    if not os.path.isabs(adapter_path):
        adapter_path = os.path.join(WREN_DIR, adapter_path)
    return adapter_path


def get_max_tokens() -> int:
    """Resolve max_tokens from config.json or default."""
    cfg = load_config()
    return int(cfg.get("max_tokens", 2048))


def get_min_chars() -> int:
    """Resolve minimum compression length from config.json or default."""
    cfg = load_config()
    return int(cfg.get("min_compress_chars", MIN_CHARS))


def get_min_savings() -> float:
    """Resolve minimum savings threshold from config.json or default."""
    cfg = load_config()
    return float(cfg.get("min_savings_pct", MIN_SAVINGS * 100)) / 100.0


def _load_model():
    """Load model + LoRA adapter once, return cached pair."""
    global _model, _tokenizer
    if _model is None:
        from mlx_lm import load
        _model, _tokenizer = load(get_base_model(), adapter_path=get_adapter_path())
    return _model, _tokenizer


def is_loaded() -> bool:
    return _model is not None


def get_stats() -> dict:
    return dict(_stats)


def compress(text: str, mode: str = "input") -> str:
    """Compress text. mode='input' for prompts, 'output' for tool results."""
    from mlx_lm import generate

    model, tokenizer = _load_model()
    system_prompt = PROMPT_OUTPUT if mode == "output" else PROMPT_INPUT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=get_max_tokens(),
        verbose=False,
    )


def compress_with_stats(text: str, mode: str = "output") -> tuple[str, dict]:
    """Compress with threshold checks and stats tracking. Default mode is 'output' for MCP use."""
    original_len = len(text)
    min_chars = get_min_chars()
    min_savings = get_min_savings()

    if original_len < min_chars:
        return text, {"compressed": False, "reason": "below minimum length"}

    result = compress(text, mode=mode)
    compressed_len = len(result)
    savings = 1 - (compressed_len / original_len) if original_len else 0

    if savings < min_savings:
        return text, {"compressed": False, "reason": f"savings too low ({savings:.0%})"}

    _stats["calls"] += 1
    _stats["chars_in"] += original_len
    _stats["chars_out"] += compressed_len

    return result, {
        "compressed": True,
        "original": original_len,
        "result": compressed_len,
        "savings": f"{savings:.0%}",
    }


def estimate_tokens(text: str) -> int:
    """Rough token estimate for quick CLI feedback."""
    return max(1, round(len(text) / TOKEN_ESTIMATE_CHARS))


def extract_signals(text: str) -> dict[str, set[str]]:
    """Extract fragile signals users care about preserving."""
    return {
        "paths": set(re.findall(r"(?:~?/)?(?:[\w.\-]+/)+[\w.\-]+(?::\d+)?", text)),
        "flags": set(re.findall(r"--[\w-]+", text)),
        "numbers": set(re.findall(r"\b\d+\b", text)),
        "negations": set(re.findall(r"\b(?:no|not|never|unless|without)\b", text, re.I)),
        "ordered_steps": set(re.findall(r"(?:^|\s)(?:\d+[.)]|[-*])\s+[^.\n]+", text)),
        "errors": set(re.findall(r"\b(?:error|exception|failed|failure|traceback)\b", text, re.I)),
    }


def preservation_report(original: str, compressed: str) -> list[str]:
    """Summarize whether fragile signals survive compression."""
    report = []
    original_signals = extract_signals(original)
    compressed_lower = compressed.lower()

    for label in ("paths", "flags", "numbers", "negations", "ordered_steps", "errors"):
        values = original_signals[label]
        if not values:
            continue
        if label in {"negations", "errors"}:
            preserved = all(value.lower() in compressed_lower for value in values)
        else:
            preserved = values.issubset(extract_signals(compressed)[label])
        report.append(f"{label}: {'ok' if preserved else 'check'}")

    if not report:
        report.append("preservation: no fragile signals detected")
    return report


def format_demo_result(title: str, original: str, compressed: str) -> str:
    """Render an example with before/after and quick savings stats."""
    original_tokens = estimate_tokens(original)
    compressed_tokens = estimate_tokens(compressed)
    savings = 1 - (len(compressed) / len(original)) if original else 0
    lines = [
        f"== {title} ==",
        "BEFORE",
        original.strip(),
        "",
        "AFTER",
        compressed.strip(),
        "",
        f"chars: {len(original)} -> {len(compressed)} ({savings:.0%} saved)",
        f"tokens (est): {original_tokens} -> {compressed_tokens} ({original_tokens - compressed_tokens} saved)",
        "preserved: " + ", ".join(preservation_report(original, compressed)),
    ]
    return "\n".join(lines)


def read_text(args: list[str]) -> str:
    """Read text from --file, stdin, or positional arguments."""
    if len(args) > 1 and args[0] == "--file":
        with open(args[1]) as f:
            return f.read()
    if not sys.stdin.isatty():
        return sys.stdin.read()
    if args:
        return " ".join(args)
    return ""


def parse_mode(args: list[str]) -> tuple[str, list[str]]:
    """Parse optional mode flags used by the demo command."""
    mode = "input"
    remaining: list[str] = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg in {"--mode", "-m"} and i + 1 < len(args):
            mode = args[i + 1]
            i += 2
            continue
        if arg in {"--output", "--tool-output"}:
            mode = "output"
            i += 1
            continue
        if arg == "--input":
            mode = "input"
            i += 1
            continue
        remaining.append(arg)
        i += 1
    return mode, remaining


def doctor(args: list[str]) -> int:
    """Check whether Wren is ready to run locally."""
    checks: list[tuple[bool, str, str]] = []
    verify_model = "--load-model" in args or "--sample" in args

    def add(ok: bool, name: str, detail: str) -> None:
        checks.append((ok, name, detail))

    add(platform.system() == "Darwin", "platform", platform.system())
    add(platform.machine() == "arm64", "architecture", platform.machine())
    add(sys.version_info >= (3, 10), "python", platform.python_version())

    for mod in ("mlx_lm", "mcp"):
        add(importlib.util.find_spec(mod) is not None, f"dependency:{mod}", "installed" if importlib.util.find_spec(mod) else "missing")

    cfg = load_config()
    add(bool(cfg), "config.json", CONFIG_PATH if cfg else "missing, using defaults/env")
    add(bool(get_base_model()), "base model", get_base_model())

    adapter_path = get_adapter_path()
    adapter_exists = os.path.isdir(adapter_path) and any(
        name.endswith(".safetensors") for name in os.listdir(adapter_path)
    )
    add(adapter_exists, "adapters", adapter_path if adapter_exists else f"missing .safetensors in {adapter_path}")

    deps_ok = all(ok for ok, name, _ in checks if name.startswith("dependency:"))
    runtime_ok = deps_ok and adapter_exists and platform.system() == "Darwin" and platform.machine() == "arm64"
    if runtime_ok and verify_model:
        try:
            result = compress(
                "Keep file paths, numbers, and negations. Do not create new files unless necessary.",
                mode="input",
            )
            add(bool(result.strip()), "sample compression", "ok")
        except Exception as exc:
            add(False, "sample compression", str(exc))
    elif runtime_ok:
        add(True, "sample compression", "skipped by default; run `wren doctor --load-model` for a live check")
    else:
        add(False, "sample compression", "skipped until platform, deps, and adapters are ready")

    failures = 0
    for ok, name, detail in checks:
        status = "OK" if ok else "FAIL"
        if not ok:
            failures += 1
        print(f"[{status}] {name}: {detail}")

    if failures:
        print(f"\nWren doctor found {failures} issue(s).")
        return 1

    print("\nWren doctor passed.")
    return 0


def demo(args: list[str]) -> int:
    """Show before/after compression on sample text or user-provided input."""
    mode, remaining = parse_mode(args)
    text = read_text(remaining).strip()

    if text:
        compressed = compress(text, mode=mode)
        print(format_demo_result("Your text", text, compressed))
        return 0

    for i, example in enumerate(DEMO_TEXT):
        compressed = example["compressed"]
        if i:
            print()
        print(format_demo_result(example["title"], example["text"], compressed))
    return 0


def main():
    args = sys.argv[1:]

    if args and args[0] == "doctor":
        sys.exit(doctor(args[1:]))

    if args and args[0] == "demo":
        sys.exit(demo(args[1:]))

    if args and args[0] == "--file" and len(args) < 2:
        print("Usage: wren --file <path>", file=sys.stderr)
        sys.exit(1)

    text = read_text(args)
    if not text:
        print(
            "Usage: wren <text> | echo 'text' | wren | wren --file <path> | wren demo | wren doctor",
            file=sys.stderr,
        )
        sys.exit(1)

    text = text.strip()
    if not text:
        sys.exit(0)

    result, _stats = compress_with_stats(text, mode="input")
    print(result)


if __name__ == "__main__":
    main()
