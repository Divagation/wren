<p align="center">
  <img src="logo.png" alt="wren" width="200">
</p>

# wren

**say more with less.**

wren is a tiny prompt compression model that makes LLM prompts shorter while preserving meaning. you pipe text in, compressed text comes out. it runs locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx).

it averages 50-80% reduction on verbose prompts. short text passes through unchanged. it preserves the things that matter -- numbers, flags, error codes, negations, conditional branches, step ordering. the stuff that breaks if you lose it.

<br>

## get it

requires Python 3.10+ and Apple Silicon.

```bash
git clone https://github.com/Divagation/wren.git
cd wren
python3 -m venv .venv && source .venv/bin/activate
pip install mlx-lm huggingface-hub

# configure the base model
cp config.example.json config.json
# edit config.json and set "base_model" to a 1.5B instruction-tuned model

# add to PATH
ln -sf $(pwd)/bin/wren ~/.local/bin/wren
```

<br>

## usage

```bash
# pipe text
echo "When implementing a REST API, make sure to use proper HTTP status codes..." | wren

# inline
wren "Your verbose prompt here"

# compress a file
wren --file ./my-system-prompt.md
```

<br>

## what it does

| input | output | saved |
|-------|--------|-------|
| "When implementing a REST API, make sure to use proper HTTP status codes for all responses. Use 200 for successful GET requests, 201 for successful POST requests, 204 for DELETE, 400 for bad client requests." | "REST API: 200 GET, 201 POST, 204 DELETE, 400 BAD." | 79% |
| "Before making any changes to the codebase, please read the relevant files first to understand the existing code structure. Do not create new files unless they are absolutely necessary. Generally prefer editing an existing file to creating a new one." | "Read existing files, do not create new unless necessary." | 78% |
| "Database migration: 1) pg_dump --format=custom. 2) Maintenance mode. 3) Run db/migrations/0042_add_indexes.sql. 4) Verify with db/verify_schema.py. 5) If fails, pg_restore. 6) Remove maintenance after verify." | "1) pg_dump --format=custom. 2) Maintenance mode. 3) db/migrations/0042_add_indexes.sql. 4) db/verify_schema.py. 5) Restore if fails. 6) Remove after verify." | 40% |

the hard stuff (exact numbers, file paths, flags, negations) stays intact. the fluff ("please make sure to", "it is important that") disappears.

<br>

## what it preserves

wren is trained to never drop things that change meaning:

- **negations** -- "NEVER do X unless Y" stays "NEVER X unless Y", not "do X"
- **values** -- numbers, status codes, paths, flags, error codes survive verbatim
- **branches** -- if/else/when/otherwise logic stays complete
- **step ordering** -- numbered procedures keep every step in order
- **constraints** -- limits, thresholds, and requirements don't get softened

<br>

## config

`config.json` controls the model and compression behavior:

```json
{
  "base_model": "your-base-model-here",
  "adapter_path": "adapters",
  "max_tokens": 2048,
  "min_compress_chars": 300,
  "min_savings_pct": 20
}
```

- `base_model` -- HuggingFace model ID (1.5B instruction-tuned models work best)
- `adapter_path` -- path to the LoRA adapter directory
- `max_tokens` -- max generation length
- `min_compress_chars` -- skip compression for text shorter than this
- `min_savings_pct` -- only use the compressed version if savings exceed this threshold

<br>

## training

wren is LoRA fine-tuned on 2,693 curated compression pairs across 20+ categories: system prompts, tool descriptions, CLI help, API docs, error messages, code comments, architecture docs, security/compliance, and edge cases like mixed code/prose and embedded JSON.

training data is mined from real Claude Code conversations and compressed via API, then cleaned:

```bash
source .venv/bin/activate

# mine conversation history for candidate text blocks
python3 generate_data.py mine

# compress candidates via Claude API
python3 generate_data.py compress

# merge into train/valid splits (90/10)
python3 generate_data.py merge

# check dataset stats
python3 generate_data.py stats
```

to retrain:

```bash
BASE=$(python3 -c "import json; print(json.load(open('config.json'))['base_model'])")
python3 -m mlx_lm lora \
  --model "$BASE" \
  --train --data data \
  --batch-size 1 --num-layers 8 --iters 800 \
  --learning-rate 1e-5 --adapter-path adapters \
  --steps-per-eval 100 --save-every 100
```

<br>

## eval

31 test cases across 6 dimensions. run it to see how the model is doing:

```bash
python3 eval.py           # summary with letter grades
python3 eval.py -v        # verbose -- show each test case
python3 eval.py -c values # filter by category
python3 eval.py -j        # JSON output
```

dimensions scored: compression ratio, value preservation, negation preservation, branch completeness, step ordering, and required content retention.

<br>

## under the hood

- **base**: 1.5B parameter instruction-tuned model
- **fine-tuning**: LoRA (8 layers, 2.6M trainable params / 0.17%)
- **training data**: 2,693 pairs (cleaned, constraint-preserving)
- **inference**: MLX native on Apple Silicon
- **latency**: ~2-5s per compression on M-series

<br>

## why "wren"

smallest bird, loudest song. that's the idea.

<br>

<p align="center">
  <sub>made by <a href="https://github.com/Divagation">divagation</a></sub>
</p>
