<p align="center">
  <img src="logo.png" alt="wren" width="200">
</p>

# wren

**say more with less.**

wren compresses prompts and tool output for Claude Code and local agent workflows. it makes verbose context smaller without dropping the details that change behavior: paths, line numbers, flags, negations, errors, and step ordering.

it runs locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). typical verbose text shrinks by 50-80%. short text passes through unchanged. the fluff disappears; the fragile details stay intact.

<br>

## why it exists

coding agents waste context on politeness, repetition, and noisy tool output. Wren is a narrow model built to compress that waste away while keeping the tokens that silently matter.

- system prompts stay directive
- grep and build output stay actionable
- exact values survive verbatim
- local-first workflows stay local

<br>

## quick proof

see Wren work before wiring it into anything:

```bash
wren demo
cat build.log | wren demo --output
```

`wren demo` shows:

- before / after text
- chars saved
- estimated tokens saved
- a quick preservation report for paths, flags, numbers, negations, errors, and step ordering

<br>

## install

requires Python 3.10+ and Apple Silicon.

fastest path:

```bash
pipx install git+https://github.com/Divagation/wren.git
wren doctor
wren demo
```

from source:

```bash
git clone https://github.com/Divagation/wren.git
cd wren
python3 -m venv .venv && source .venv/bin/activate
pip install -e .
wren doctor
wren demo
```

Wren defaults to `Qwen/Qwen2.5-1.5B-Instruct`, which matches the included adapters. if you want to override it, use `config.json` or `WREN_BASE_MODEL`.

```bash
cp config.example.json config.json
# optional: edit config.json if you want a different base model or thresholds

# add to PATH
ln -sf $(pwd)/bin/wren ~/.local/bin/wren
```

<br>

## cli

pipe text in, compressed text out:

```bash
echo "When implementing a REST API, make sure to use proper HTTP status codes..." | wren
wren "Your verbose prompt here"
wren --file ./my-system-prompt.md
wren demo
wren doctor
```

<br>

## MCP server

wren also runs as an MCP server that compresses tool output for Claude Code. the model loads once and stays hot in memory, so compression becomes part of the workflow instead of a separate wait.

**tools exposed:**

| tool | use for | not for |
|------|---------|---------|
| `compressed_read` | exploring unfamiliar files | files you need to edit |
| `compressed_grep` | broad codebase searches | exact line numbers |
| `compressed_exec` | verbose read/build/test/log commands | arbitrary shell or destructive commands |
| `compress_text` | shrinking any large text blob | -- |
| `wren_status` | session compression stats | -- |

**setup:**

```bash
# packaged install
claude mcp add wren wren-mcp -s user

# source checkout
claude mcp add wren /path/to/wren/.venv/bin/python -- /path/to/wren/mcp_server.py -s user

# optional auto-approve for the non-exec tools only
# do not auto-approve compressed_exec
# add: "mcp__wren__compressed_read", "mcp__wren__compressed_grep",
#      "mcp__wren__compress_text", "mcp__wren__wren_status"
```

`compressed_exec` does not open a shell. it only allows a constrained set of inspection/build/test/log commands, and rejects arbitrary or destructive invocations.

<br>

## what it does

| input | output | saved |
|-------|--------|-------|
| "When implementing a REST API, make sure to use proper HTTP status codes for all responses. Use 200 for successful GET requests, 201 for successful POST requests, 204 for DELETE, 400 for bad client requests." | "REST API: 200 GET, 201 POST, 204 DELETE, 400 BAD." | 79% |
| "Before making any changes to the codebase, please read the relevant files first to understand the existing code structure. Do not create new files unless they are absolutely necessary." | "Read existing files, do not create new unless necessary." | 78% |
| "Database migration: 1) pg_dump --format=custom. 2) Maintenance mode. 3) Run db/migrations/0042_add_indexes.sql..." | "1) pg_dump --format=custom. 2) Maintenance mode. 3) db/migrations/0042_add_indexes.sql..." | 40% |

the hard stuff (exact numbers, file paths, flags, negations) stays intact. the fluff disappears.

<br>

## what it preserves

- **negations** -- "NEVER do X unless Y" stays "NEVER X unless Y", not "do X"
- **values** -- numbers, status codes, paths, flags, error codes survive verbatim
- **branches** -- if/else/when/otherwise logic stays complete
- **step ordering** -- numbered procedures keep every step in order
- **constraints** -- limits, thresholds, requirements don't get softened
- **file paths** -- exact paths, line numbers, function signatures (tool output mode)

<br>

## two compression modes

wren uses different system prompts depending on what it's compressing:

- **input mode** (`mode="input"`) -- for user prompts, system instructions, documentation. focuses on preserving meaning and instruction-following behavior.
- **output mode** (`mode="output"`) -- for tool results (code, grep, build logs). focuses on preserving actionable information: paths, line numbers, errors, signatures.

the CLI uses input mode. the MCP server uses output mode.

<br>

## training

wren is LoRA fine-tuned on compression pairs across 20+ categories. training data comes from two pipelines:

**input prompts** (existing):
```bash
python3 generate_data.py mine       # mine prompts from conversation history
python3 generate_data.py compress   # compress via Claude API
python3 generate_data.py merge      # merge into train/valid splits
```

**tool output** (new):
```bash
python3 generate_tool_output.py mine       # mine Read/Grep/Bash results
python3 generate_tool_output.py compress   # compress via Claude API
python3 generate_tool_output.py merge      # merge into train/valid splits
```

both pipelines write to the same `train.jsonl` / `valid.jsonl`. the system prompt in each training example tells the model which mode to use.

**retrain:**
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

31 test cases across 6 dimensions:

```bash
python3 eval.py           # summary with letter grades
python3 eval.py -v        # verbose per test case
python3 eval.py -c values # filter by category
python3 eval.py -j        # JSON output
```

dimensions: compression ratio, value preservation, negation preservation, branch completeness, step ordering, required content retention.

<br>

## under the hood

- **base**: 1.5B parameter instruction-tuned model
- **fine-tuning**: LoRA (8 layers, 2.6M trainable params / 0.17%)
- **inference**: MLX native on Apple Silicon
- **latency**: ~2-5s per compression (CLI), near-instant after first call (MCP server)
- **modes**: input compression (prompts) + output compression (tool results)

<br>

## why "wren"

smallest bird, loudest song.

<br>

<p align="center">
  <sub>made by <a href="https://github.com/Divagation">divagation</a></sub>
</p>
