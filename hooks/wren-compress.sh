#!/bin/bash
# Wren prompt compression hook for Claude Code
# Only compresses prompts longer than 300 chars to avoid overhead on short messages

INPUT=$(cat)
PROMPT=$(echo "$INPUT" | jq -r '.prompt // empty')

# Skip short prompts -- compression overhead isn't worth it
if [ ${#PROMPT} -lt 300 ]; then
  exit 0
fi

# Compress via Wren
COMPRESSED=$(echo "$PROMPT" | /Users/brandon/wren/bin/wren 2>/dev/null)

# If compression failed or produced empty output, pass through
if [ -z "$COMPRESSED" ]; then
  exit 0
fi

# If compressed version isn't meaningfully shorter, skip
ORIG_LEN=${#PROMPT}
COMP_LEN=${#COMPRESSED}
SAVINGS=$(( (ORIG_LEN - COMP_LEN) * 100 / ORIG_LEN ))

if [ "$SAVINGS" -lt 20 ]; then
  exit 0
fi

# Output compressed version as additional context
jq -n \
  --arg compressed "$COMPRESSED" \
  --arg savings "${SAVINGS}% compressed by Wren" \
  '{
    "additionalContext": ("Compressed user prompt: " + $compressed)
  }'
