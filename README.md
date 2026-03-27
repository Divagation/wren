# Wren

A tiny prompt compression model that makes LLM prompts shorter while preserving meaning. Built on Apple Silicon with [MLX](https://github.com/ml-explore/mlx).

Wren averages **50-75% reduction** on verbose prompts, correctly passes through short text unchanged, and preserves critical details like URLs, file paths, negative constraints, and conditional logic.

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
| "When implementing a REST API, make sure to use proper HTTP status codes for all responses. Use 200 for successful GET requests, 201 for successful POST requests..." | "REST API: 200=GET, 201=POST create, 204=DELETE, 400=bad client." | 74% |
| "NEVER use dark themes. Always light. No exceptions." | "NEVER dark themes. Always light. No exceptions." | 49% |
| "Fix the bug." | "Fix the bug." | 0% |

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

Wren is fine-tuned with LoRA on 183 curated prompt compression pairs covering:

- System prompts and tool descriptions
- Multi-step instructions and conditional logic
- Code context with file paths and URLs
- Negative constraints (NEVER/ALWAYS preserved)
- Short prompt passthrough (no unnecessary changes)
- Domain-specific content (trading, construction, creative)

To retrain with your own data:

```bash
source .venv/bin/activate
BASE=$(python3 -c "import json; print(json.load(open('config.json'))['base_model'])")
python3 -m mlx_lm lora \
  --model "$BASE" \
  --train --data data \
  --batch-size 2 --num-layers 16 --iters 300 \
  --learning-rate 1e-5 --adapter-path adapters \
  --steps-per-eval 50 --save-every 50
```

## Architecture

- **Base**: 1.5B parameter instruction-tuned model
- **Fine-tuning**: LoRA (16 layers, 5.3M trainable params / 0.34%)
- **Inference**: MLX native on Apple Silicon
- **Latency**: ~2-5s per compression on M-series Mac
