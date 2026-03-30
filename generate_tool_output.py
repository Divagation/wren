#!/usr/bin/env python3
"""
Wren tool output training data generator.
Mines Claude Code conversation history for tool RESULTS (Read, Grep, Bash outputs),
then uses Claude to generate compressed versions as training pairs.

Usage:
  python3 generate_tool_output.py mine       # Extract tool output from conversations
  python3 generate_tool_output.py compress   # Generate compressions via Claude API
  python3 generate_tool_output.py merge      # Merge into train.jsonl / valid.jsonl
  python3 generate_tool_output.py stats      # Show dataset statistics
"""

import json
import os
import sys
import hashlib
import random
from pathlib import Path

WREN_DIR = Path(__file__).parent
DATA_DIR = WREN_DIR / "data"
CANDIDATES_FILE = DATA_DIR / "tool_output_candidates.jsonl"
COMPRESSED_FILE = DATA_DIR / "tool_output_compressed.jsonl"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"

SYSTEM_PROMPT = (
    "You are Wren, a tool output compression model. Compress the tool output to its "
    "shortest form while preserving all actionable information: file paths, line numbers, "
    "error messages, function signatures, variable names, status codes, and structural "
    "relationships. Output only the compressed text."
)

MIN_CHARS = 200
MAX_CHARS = 5000
MAX_RATIO = 0.75

# Tool results that are trivial / not worth training on
SKIP_PATTERNS = [
    "No matches found",
    "(no output)",
    "No files found",
    "File not found",
    "command not found",
]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def is_trivial(text: str) -> bool:
    """Skip trivial tool output that doesn't benefit from compression."""
    stripped = text.strip()
    if any(stripped.startswith(p) or stripped == p for p in SKIP_PATTERNS):
        return True
    # Skip if it's just a single short line
    lines = stripped.split("\n")
    if len(lines) == 1 and len(stripped) < MIN_CHARS:
        return True
    return False


def is_binary(text: str) -> bool:
    """Detect binary/base64 content."""
    if "base64" in text[:200].lower():
        return True
    # High ratio of non-printable chars
    non_printable = sum(1 for c in text[:500] if not c.isprintable() and c not in "\n\r\t")
    return non_printable > len(text[:500]) * 0.1


def extract_tool_results(filepath: Path) -> list[dict]:
    """Extract tool results from a conversation file.

    Conversation format:
    - assistant messages contain tool_use items with name and id
    - user messages contain tool_result items with tool_use_id and content
    """
    results = []
    tool_names = {}  # tool_use_id -> tool_name

    try:
        with open(filepath) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                msg = obj.get("message", obj)
                content = msg.get("content", [])
                if not isinstance(content, list):
                    continue

                for item in content:
                    if not isinstance(item, dict):
                        continue

                    # Map tool_use_id -> tool_name
                    if item.get("type") == "tool_use":
                        tool_id = item.get("id", "")
                        tool_name = item.get("name", "")
                        if tool_id and tool_name:
                            tool_names[tool_id] = tool_name

                    # Extract tool results
                    if item.get("type") == "tool_result":
                        tool_id = item.get("tool_use_id", "")
                        tool_name = tool_names.get(tool_id, "unknown")

                        # Only care about Read, Grep, Bash
                        if tool_name not in ("Read", "Grep", "Bash"):
                            continue

                        raw = item.get("content", "")

                        # Handle string content
                        if isinstance(raw, str):
                            text = raw
                        # Handle list content (extract text blocks)
                        elif isinstance(raw, list):
                            texts = []
                            for block in raw:
                                if isinstance(block, dict) and block.get("type") == "text":
                                    texts.append(block.get("text", ""))
                            text = "\n".join(texts)
                        else:
                            continue

                        if text:
                            results.append({
                                "text": text,
                                "tool": tool_name,
                            })

    except Exception as e:
        print(f"  Error reading {filepath.name}: {e}", file=sys.stderr)

    return results


def mine_conversations():
    """Extract tool output candidates from Claude Code conversation history."""
    seen_hashes = set()
    candidates = []

    # Load existing candidates
    if CANDIDATES_FILE.exists():
        with open(CANDIDATES_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                seen_hashes.add(d["hash"])
        print(f"Loaded {len(seen_hashes)} existing candidates")

    # Load existing training data hashes
    for path in [TRAIN_FILE, VALID_FILE]:
        if path.exists():
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    h = content_hash(d["messages"][1]["content"])
                    seen_hashes.add(h)

    print(f"Total existing hashes (dedup): {len(seen_hashes)}")

    conv_files = list(CLAUDE_PROJECTS.rglob("*.jsonl"))
    print(f"Found {len(conv_files)} conversation files")

    tool_counts = {"Read": 0, "Grep": 0, "Bash": 0}

    for filepath in conv_files:
        if "wren" in str(filepath).lower():
            continue

        results = extract_tool_results(filepath)

        for r in results:
            text = r["text"]
            tool = r["tool"]

            # Skip binary/trivial
            if is_binary(text) or is_trivial(text):
                continue

            # Trim to max
            if len(text) > MAX_CHARS:
                text = text[:MAX_CHARS]

            text = text.strip()
            if len(text) < MIN_CHARS:
                continue

            h = content_hash(text)
            if h in seen_hashes:
                continue
            seen_hashes.add(h)

            tool_counts[tool] = tool_counts.get(tool, 0) + 1
            candidates.append({
                "hash": h,
                "text": text,
                "chars": len(text),
                "tool": tool,
                "source": str(filepath.relative_to(CLAUDE_PROJECTS)),
            })

    new_count = len(candidates)
    if new_count > 0:
        with open(CANDIDATES_FILE, "a") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")

    print(f"\nMined {new_count} new candidates")
    for tool, count in sorted(tool_counts.items()):
        print(f"  {tool}: {count}")

    total = 0
    if CANDIDATES_FILE.exists():
        with open(CANDIDATES_FILE) as f:
            total = sum(1 for l in f if l.strip())
    print(f"Total candidates: {total}")


COMPRESS_INSTRUCTION = (
    "TASK: Compress the following tool output to its shortest form. "
    "Preserve ALL actionable information: file paths, line numbers, error messages, "
    "function signatures, variable names, status codes, structural relationships. "
    "Output ONLY the compressed text. No explanation, no formatting, no markdown.\n\n"
    "TEXT:\n"
)


def _compress_via_claude(text: str) -> str:
    """Compress text using claude --print (uses subscription, not API credits)."""
    import subprocess

    result = subprocess.run(
        ["claude", "--print", "--tools", "", "--model", "haiku"],
        input=COMPRESS_INSTRUCTION + text,
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"exit {result.returncode}")
    return result.stdout.strip()


def _compress_one(c: dict) -> dict | None:
    """Compress a single candidate. Returns result dict or None on failure."""
    try:
        compressed = _compress_via_claude(c["text"])
        ratio = len(compressed) / len(c["text"]) if len(c["text"]) > 0 else 1.0
        return {
            "hash": c["hash"],
            "input": c["text"],
            "output": compressed,
            "input_chars": len(c["text"]),
            "output_chars": len(compressed),
            "ratio": round(ratio, 3),
            "tool": c.get("tool", "unknown"),
        }
    except Exception as e:
        print(f"  ERROR: {e}", file=sys.stderr)
        return None


def compress_candidates():
    """Generate compressions for tool output candidates via claude --print (8 workers)."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    compressed_hashes = set()
    if COMPRESSED_FILE.exists():
        with open(COMPRESSED_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                compressed_hashes.add(d["hash"])

    candidates = []
    with open(CANDIDATES_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["hash"] not in compressed_hashes:
                candidates.append(d)

    print(f"Candidates to compress: {len(candidates)} (already done: {len(compressed_hashes)})")

    if not candidates:
        print("Nothing to compress.")
        return

    workers = 1
    total_done = 0
    batch_size = 20

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i : i + batch_size]
        print(f"\nBatch {i // batch_size + 1} ({len(batch)} items)...")

        results = []
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_compress_one, c): c for c in batch}
            for future in as_completed(futures):
                r = future.result()
                if r:
                    results.append(r)
                    status = "OK" if r["ratio"] <= MAX_RATIO else "WEAK"
                    print(f"  [{status}] {r['tool']}: {r['input_chars']}->{r['output_chars']} ({r['ratio']:.0%})")

        with open(COMPRESSED_FILE, "a") as out:
            for r in results:
                out.write(json.dumps(r) + "\n")
        total_done += len(results)

        print(f"  Done: {total_done} total")

    print(f"\nCompressed {total_done} candidates total")


def merge_data():
    """Merge compressed tool output into train/valid splits."""
    if not COMPRESSED_FILE.exists():
        print("No compressed data found. Run 'compress' first.")
        return

    pairs = []
    with open(COMPRESSED_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            if d["ratio"] <= MAX_RATIO and d["output_chars"] >= 20:
                pairs.append(d)

    print(f"Good compressed pairs: {len(pairs)}")

    existing_hashes = set()
    existing_train = []
    existing_valid = []

    if TRAIN_FILE.exists():
        with open(TRAIN_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                existing_train.append(d)
                existing_hashes.add(content_hash(d["messages"][1]["content"]))

    if VALID_FILE.exists():
        with open(VALID_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                existing_valid.append(d)
                existing_hashes.add(content_hash(d["messages"][1]["content"]))

    print(f"Existing: {len(existing_train)} train + {len(existing_valid)} valid")

    new_examples = []
    for p in pairs:
        h = content_hash(p["input"])
        if h in existing_hashes:
            continue
        existing_hashes.add(h)
        new_examples.append({
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": p["input"]},
                {"role": "assistant", "content": p["output"]},
            ]
        })

    print(f"New examples to add: {len(new_examples)}")

    if not new_examples:
        print("No new examples. Done.")
        return

    random.shuffle(new_examples)
    split = max(1, len(new_examples) // 10)
    new_valid = new_examples[:split]
    new_train = new_examples[split:]

    all_train = existing_train + new_train
    all_valid = existing_valid + new_valid

    with open(TRAIN_FILE, "w") as f:
        for ex in all_train:
            f.write(json.dumps(ex) + "\n")

    with open(VALID_FILE, "w") as f:
        for ex in all_valid:
            f.write(json.dumps(ex) + "\n")

    print(f"Final: {len(all_train)} train + {len(all_valid)} valid ({len(all_train) + len(all_valid)} total)")


def show_stats():
    """Show tool output dataset statistics."""
    # Candidates by tool
    if CANDIDATES_FILE.exists():
        tool_counts = {}
        total = 0
        with open(CANDIDATES_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                tool = d.get("tool", "unknown")
                tool_counts[tool] = tool_counts.get(tool, 0) + 1
                total += 1
        print(f"Tool output candidates: {total}")
        for tool, count in sorted(tool_counts.items()):
            print(f"  {tool}: {count}")
    else:
        print("No candidates yet. Run 'mine' first.")

    # Compressed stats
    if COMPRESSED_FILE.exists():
        tool_ratios = {}
        with open(COMPRESSED_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                tool = d.get("tool", "unknown")
                if tool not in tool_ratios:
                    tool_ratios[tool] = []
                tool_ratios[tool].append(d["ratio"])

        print(f"\nCompressed pairs by tool:")
        for tool, ratios in sorted(tool_ratios.items()):
            avg = sum(ratios) / len(ratios)
            print(f"  {tool}: {len(ratios)} pairs, avg ratio {avg:.2f}")
    else:
        print("\nNo compressed data yet. Run 'compress' first.")

    # Overall training data
    print()
    for name, path in [("train", TRAIN_FILE), ("valid", VALID_FILE)]:
        if not path.exists():
            continue
        with open(path) as f:
            count = sum(1 for l in f if l.strip())
        print(f"{name}: {count} examples")


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return

    cmd = sys.argv[1]
    if cmd == "mine":
        mine_conversations()
    elif cmd == "compress":
        compress_candidates()
    elif cmd == "merge":
        merge_data()
    elif cmd == "stats":
        show_stats()
    else:
        print(f"Unknown command: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
