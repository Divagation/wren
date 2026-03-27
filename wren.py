#!/usr/bin/env python3
"""Wren - prompt compression model. Pipe text in, get compressed text out."""

import sys
import os

# Suppress mlx warnings
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

WREN_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(WREN_DIR, "adapters")
BASE_MODEL = os.environ.get("WREN_BASE_MODEL", "")
CONFIG_PATH = os.path.join(WREN_DIR, "config.json")
SYSTEM_PROMPT = "You are Wren, a prompt compression model. Compress the input to its shortest form while preserving all meaning and instruction-following behavior. Output only the compressed text."


def get_base_model() -> str:
    """Resolve base model from config.json or WREN_BASE_MODEL env var."""
    import json
    if os.path.exists(CONFIG_PATH):
        with open(CONFIG_PATH) as f:
            cfg = json.load(f)
            model = cfg.get("base_model", "")
            if model:
                return model
    if BASE_MODEL:
        return BASE_MODEL
    print("Error: set base_model in config.json or WREN_BASE_MODEL env var", file=sys.stderr)
    sys.exit(1)


def compress(text: str) -> str:
    from mlx_lm import load, generate

    model, tokenizer = load(get_base_model(), adapter_path=ADAPTER_PATH)
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": text},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    return generate(model, tokenizer, prompt=prompt, max_tokens=2048, verbose=False)


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--file":
        if len(sys.argv) < 3:
            print("Usage: wren --file <path>", file=sys.stderr)
            sys.exit(1)
        with open(sys.argv[2]) as f:
            text = f.read()
    elif not sys.stdin.isatty():
        text = sys.stdin.read()
    elif len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        print("Usage: wren <text>  |  echo 'text' | wren  |  wren --file <path>", file=sys.stderr)
        sys.exit(1)

    text = text.strip()
    if not text:
        sys.exit(0)

    result = compress(text)
    print(result)


if __name__ == "__main__":
    main()
