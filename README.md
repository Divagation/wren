<p align="center">
  <img src="logo.png" alt="Wren" width="200">
</p>

<h1 align="center">Wren</h1>

<p align="center">A tiny prompt compression model that makes LLM prompts shorter while preserving meaning. Built on Apple Silicon with <a href="https://github.com/ml-explore/mlx">MLX</a>.</p>

Wren averages **50-80% reduction** on verbose prompts, preserves critical details like numbers, flags, error codes, negations, conditional branches, and step ordering. Short text passes through unchanged.

## Usage

```bash
# Pipe text
echo "When implementing a REST API, make sure to use proper HTTP status codes..." | wren

# Inline
wren "Your verbose prompt here"

# Compress a file
wren --file ./my-system-prompt.md
```

## Examples

| Input | Output | Reduction |
|-------|--------|-----------|
| "When implementing a REST API, make sure to use proper HTTP status codes for all responses. Use 200 for successful GET requests, 201 for successful POST requests, 204 for DELETE, 400 for bad client requests." | "REST API: 200 GET, 201 POST, 204 DELETE, 400 BAD." | 79% |
| "Before making any changes to the codebase, please read the relevant files first to understand the existing code structure. Do not create new files unless they are absolutely necessary. Generally prefer editing an existing file to creating a new one." | "Read existing files, do not create new unless necessary." | 78% |
| "Database migration: 1) pg_dump --format=custom. 2) Maintenance mode. 3) Run db/migrations/0042_add_indexes.sql. 4) Verify with db/verify_schema.py. 5) If fails, pg_restore. 6) Remove maintenance after verify." | "1) pg_dump --format=custom. 2) Maintenance mode. 3) db/migrations/0042_add_indexes.sql. 4) db/verify_schema.py. 5) Restore if fails. 6) Remove after verify." | 40% |

## Install

Requires Python 3.10+ and Apple Silicon (MLX).

```bash
git clone https://github.com/Divagation/wren.git
cd wren
python3 -m venv .venv && source .venv/bin/activate
pip install mlx-lm huggingface-hub

# Configure the base model
cp config.example.json config.json
# Edit config.json and set "base_model" to a 1.5B instruction-tuned model

# Add to PATH
ln -sf $(pwd)/bin/wren ~/.local/bin/wren
```

## Claude Code Hook

Wren can auto-compress long prompts before they hit the API. Add to `~/.claude/settings.json`:

```json
{
  "hooks": {
    "UserPromptSubmit": [
      {
        "hooks": [
          {
            "type": "command",
            "command": "/path/to/wren/hooks/wren-compress.sh"
          }
        ]
      }
    ]
  }
}
```

Only prompts over 300 characters with >20% compression savings are rewritten.

## Training

Wren is fine-tuned with LoRA on 2,693 curated prompt compression pairs across 20+ categories:

- System prompts, tool descriptions, CLI help text
- API docs, config instructions, error messages
- Code comments, git descriptions, architecture docs
- Negation/exception preservation ("NEVER X unless Y")
- Value preservation (numbers, flags, codes, paths)
- Conditional branch preservation (all branches kept)
- Step ordering preservation (procedures stay complete)
- Security/compliance, legal, email/comms, ML/AI docs
- Edge cases (short text, mixed code/prose, embedded JSON)

To retrain:

```bash
source .venv/bin/activate
BASE=$(python3 -c "import json; print(json.load(open('config.json'))['base_model'])")
python3 -m mlx_lm lora \
  --model "$BASE" \
  --train --data data \
  --batch-size 1 --num-layers 8 --iters 800 \
  --learning-rate 1e-5 --adapter-path adapters \
  --steps-per-eval 100 --save-every 100
```

## Eval

Run the eval harness to score compression quality across 31 test cases:

```bash
python3 eval.py           # summary
python3 eval.py -v        # verbose (show each test)
python3 eval.py -c values # filter by category
python3 eval.py -j        # JSON output
```

Scores 6 dimensions: ratio, value preservation, negation preservation, branch completeness, step ordering, and required content.

## Architecture

- **Base**: 1.5B parameter instruction-tuned model
- **Fine-tuning**: LoRA (8 layers, 2.6M trainable params / 0.17%)
- **Training data**: 2,693 pairs (cleaned, constraint-preserving)
- **Inference**: MLX native on Apple Silicon
- **Latency**: ~2-5s per compression on M-series Mac
