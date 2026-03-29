#!/usr/bin/env python3
"""
Wren training data generator.
Mines Claude Code conversation history for compressible text blocks,
then uses Claude Haiku to generate high-quality compressions.

Usage:
  python3 generate_data.py mine          # Extract candidates from conversation history
  python3 generate_data.py compress      # Generate compressions via Claude API
  python3 generate_data.py merge         # Merge into train.jsonl / valid.jsonl
  python3 generate_data.py stats         # Show dataset statistics
"""

import json
import os
import sys
import hashlib
import random
from pathlib import Path

WREN_DIR = Path(__file__).parent
DATA_DIR = WREN_DIR / "data"
CANDIDATES_FILE = DATA_DIR / "candidates.jsonl"
COMPRESSED_FILE = DATA_DIR / "compressed.jsonl"
TRAIN_FILE = DATA_DIR / "train.jsonl"
VALID_FILE = DATA_DIR / "valid.jsonl"

CLAUDE_PROJECTS = Path.home() / ".claude" / "projects"

SYSTEM_PROMPT = "You are Wren, a prompt compression model. Compress the input to its shortest form while preserving all meaning and instruction-following behavior. Output only the compressed text."

# Minimum chars for a text block to be worth compressing
MIN_CHARS = 200
# Maximum chars (avoid extremely long blocks that blow up context)
MAX_CHARS = 3000
# Target compression ratio (output/input) -- reject if above this
MAX_RATIO = 0.75


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def extract_text_blocks(obj: dict) -> list[str]:
    """Extract compressible text blocks from a conversation message."""
    blocks = []

    content = obj.get("content", "")
    if isinstance(content, str) and len(content) >= MIN_CHARS:
        blocks.append(content)
    elif isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                text = item.get("text", "") or item.get("content", "")
                if isinstance(text, str) and len(text) >= MIN_CHARS:
                    blocks.append(text)

    return blocks


def mine_conversations():
    """Extract candidate text blocks from Claude Code conversation history."""
    seen_hashes = set()
    candidates = []

    # Load existing candidates to avoid dupes
    if CANDIDATES_FILE.exists():
        with open(CANDIDATES_FILE) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                seen_hashes.add(d["hash"])
        print(f"Loaded {len(seen_hashes)} existing candidates")

    # Also load existing training data hashes
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

    # Walk all conversation files
    conv_files = list(CLAUDE_PROJECTS.rglob("*.jsonl"))
    print(f"Found {len(conv_files)} conversation files")

    for filepath in conv_files:
        # Skip wren's own data
        if "wren" in str(filepath).lower():
            continue

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

                    blocks = extract_text_blocks(obj)
                    for block in blocks:
                        # Trim to max
                        if len(block) > MAX_CHARS:
                            block = block[:MAX_CHARS]

                        # Clean up
                        block = block.strip()
                        if len(block) < MIN_CHARS:
                            continue

                        h = content_hash(block)
                        if h in seen_hashes:
                            continue
                        seen_hashes.add(h)

                        candidates.append({
                            "hash": h,
                            "text": block,
                            "chars": len(block),
                            "source": str(filepath.relative_to(CLAUDE_PROJECTS)),
                        })
        except Exception as e:
            print(f"  Error reading {filepath.name}: {e}", file=sys.stderr)

    # Append new candidates
    new_count = len(candidates)
    if new_count > 0:
        with open(CANDIDATES_FILE, "a") as f:
            for c in candidates:
                f.write(json.dumps(c) + "\n")

    print(f"Mined {new_count} new candidates")

    # Show stats
    total = 0
    if CANDIDATES_FILE.exists():
        with open(CANDIDATES_FILE) as f:
            total = sum(1 for l in f if l.strip())
    print(f"Total candidates: {total}")


def compress_candidates():
    """Use Claude API to generate compressions for candidates."""
    import anthropic

    client = anthropic.Anthropic()

    # Load candidates that haven't been compressed yet
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

    # Process in batches
    batch_size = 50
    total_done = 0

    for i in range(0, len(candidates), batch_size):
        batch = candidates[i:i + batch_size]
        print(f"\nBatch {i // batch_size + 1} ({len(batch)} items)...")

        with open(COMPRESSED_FILE, "a") as out:
            for c in batch:
                try:
                    response = client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=1024,
                        system=SYSTEM_PROMPT,
                        messages=[{"role": "user", "content": c["text"]}],
                    )
                    compressed = response.content[0].text.strip()

                    ratio = len(compressed) / len(c["text"]) if len(c["text"]) > 0 else 1.0

                    result = {
                        "hash": c["hash"],
                        "input": c["text"],
                        "output": compressed,
                        "input_chars": len(c["text"]),
                        "output_chars": len(compressed),
                        "ratio": round(ratio, 3),
                    }
                    out.write(json.dumps(result) + "\n")
                    total_done += 1

                    status = "OK" if ratio <= MAX_RATIO else "WEAK"
                    print(f"  [{status}] {len(c['text'])}→{len(compressed)} ({ratio:.0%})")

                except Exception as e:
                    print(f"  ERROR: {e}", file=sys.stderr)

        print(f"  Done: {total_done} total")

    print(f"\nCompressed {total_done} candidates total")


def merge_data():
    """Merge compressed data into train/valid splits."""
    if not COMPRESSED_FILE.exists():
        print("No compressed data found. Run 'compress' first.")
        return

    # Load all good compressed pairs
    pairs = []
    with open(COMPRESSED_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            # Filter: good compression ratio and not trivially short
            if d["ratio"] <= MAX_RATIO and d["output_chars"] >= 20:
                pairs.append(d)

    print(f"Good compressed pairs: {len(pairs)} (filtered from compressed.jsonl)")

    # Load existing training data
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

    # Convert new pairs to training format
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

    # Shuffle and split 90/10
    random.shuffle(new_examples)
    split = max(1, len(new_examples) // 10)
    new_valid = new_examples[:split]
    new_train = new_examples[split:]

    # Append
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
    """Show dataset statistics."""
    for name, path in [("train", TRAIN_FILE), ("valid", VALID_FILE)]:
        if not path.exists():
            print(f"{name}: not found")
            continue

        examples = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                inp = d["messages"][1]["content"]
                out = d["messages"][2]["content"]
                examples.append((len(inp), len(out)))

        if not examples:
            print(f"{name}: empty")
            continue

        in_lens = [e[0] for e in examples]
        out_lens = [e[1] for e in examples]
        ratios = [o / i if i > 0 else 0 for i, o in examples]

        print(f"\n{name}: {len(examples)} examples")
        print(f"  Input:  min={min(in_lens)}, max={max(in_lens)}, avg={sum(in_lens)//len(in_lens)}")
        print(f"  Output: min={min(out_lens)}, max={max(out_lens)}, avg={sum(out_lens)//len(out_lens)}")
        print(f"  Ratio:  min={min(ratios):.2f}, max={max(ratios):.2f}, avg={sum(ratios)/len(ratios):.2f}")

    # Candidates / compressed stats
    for name, path in [("candidates", CANDIDATES_FILE), ("compressed", COMPRESSED_FILE)]:
        if path.exists():
            with open(path) as f:
                count = sum(1 for l in f if l.strip())
            print(f"\n{name}: {count} entries")


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
