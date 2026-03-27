#!/usr/bin/env python3
"""Comprehensive tests for Wren training data quality and consistency."""

import json
import os
import re
import sys
import unittest
from collections import Counter

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_PATH = os.path.join(PROJECT_ROOT, "data", "train.jsonl")
VALID_PATH = os.path.join(PROJECT_ROOT, "data", "valid.jsonl")

EXPECTED_SYSTEM_PROMPT = (
    "You are Wren, a prompt compression model. Compress the input to its "
    "shortest form while preserving all meaning and instruction-following "
    "behavior. Output only the compressed text."
)

URL_PATTERN = re.compile(r"https?://[^\s\)\]\}\"'>,]+[^\s\)\]\}\"'>,.:;!?]")
CONSTRAINT_KEYWORDS = ["NEVER", "ALWAYS", "MUST NOT"]


def load_jsonl(path):
    """Load a JSONL file, returning (entries, errors).

    entries: list of (line_number, parsed_object)
    errors:  list of (line_number, error_string)
    """
    entries = []
    errors = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
                entries.append((i, obj))
            except json.JSONDecodeError as e:
                errors.append((i, str(e)))
    return entries, errors


def get_io_pairs(entries):
    """Extract (line_number, input_text, output_text) from parsed entries."""
    pairs = []
    for ln, obj in entries:
        if "messages" not in obj or len(obj["messages"]) != 3:
            continue
        pairs.append((ln, obj["messages"][1]["content"], obj["messages"][2]["content"]))
    return pairs


# ---------------------------------------------------------------------------
# Mixin with all per-file checks
# ---------------------------------------------------------------------------

class TrainingDataMixin:
    """Shared test logic applied to both train.jsonl and valid.jsonl."""

    data_path = None  # override in concrete subclass

    @classmethod
    def setUpClass(cls):
        cls.entries, cls.parse_errors = load_jsonl(cls.data_path)
        cls.io_pairs = get_io_pairs(cls.entries)

    # -- 1. JSONL Format Validation ----------------------------------------

    def test_file_exists(self):
        self.assertTrue(os.path.isfile(self.data_path), f"File not found: {self.data_path}")

    def test_all_lines_valid_json(self):
        """Every line must be valid JSON."""
        self.assertEqual(
            len(self.parse_errors), 0,
            f"Invalid JSON at lines: {self.parse_errors}"
        )

    def test_every_entry_has_messages_key(self):
        missing = [ln for ln, obj in self.entries if "messages" not in obj]
        self.assertEqual(missing, [], f"Missing 'messages' key on lines: {missing}")

    def test_exactly_three_messages(self):
        bad = [
            (ln, len(obj["messages"]))
            for ln, obj in self.entries
            if "messages" in obj and len(obj["messages"]) != 3
        ]
        self.assertEqual(bad, [], f"Wrong message count on lines: {bad}")

    def test_roles_are_system_user_assistant(self):
        expected = ["system", "user", "assistant"]
        bad = []
        for ln, obj in self.entries:
            if "messages" not in obj:
                continue
            roles = [m.get("role") for m in obj["messages"]]
            if roles != expected:
                bad.append((ln, roles))
        self.assertEqual(bad, [], f"Wrong roles: {bad}")

    def test_no_empty_content(self):
        bad = []
        for ln, obj in self.entries:
            if "messages" not in obj:
                continue
            for m in obj["messages"]:
                content = m.get("content", "")
                if not content or not content.strip():
                    bad.append((ln, m.get("role")))
        self.assertEqual(bad, [], f"Empty content: {bad}")

    # -- 2. System Prompt Consistency --------------------------------------

    def test_system_prompt_exact_match(self):
        bad = []
        for ln, obj in self.entries:
            if "messages" not in obj:
                continue
            if obj["messages"][0].get("content") != EXPECTED_SYSTEM_PROMPT:
                bad.append(ln)
        self.assertEqual(bad, [], f"System prompt mismatch on lines: {bad}")

    # -- 3. Compression Ratio Analysis -------------------------------------

    def test_compression_ratio_bounds_long_inputs(self):
        """For inputs >100 chars: ratio must be in [0.15, 0.85]."""
        bad = []
        for ln, inp, out in self.io_pairs:
            if len(inp) <= 100:
                continue
            ratio = len(out) / len(inp)
            if not (0.15 <= ratio <= 0.85):
                bad.append((ln, f"{ratio:.3f}", len(inp), len(out)))
        self.assertEqual(bad, [], f"Ratio out of bounds: {bad}")

    def test_no_expansion_long_inputs(self):
        """For inputs >100 chars: output must not exceed input length."""
        bad = []
        for ln, inp, out in self.io_pairs:
            if len(inp) <= 100:
                continue
            if len(out) > len(inp):
                bad.append((ln, len(inp), len(out)))
        self.assertEqual(bad, [], f"Output longer than input: {bad}")

    def test_short_input_reasonable_length(self):
        """For inputs <50 chars: output length <= input length * 1.5."""
        bad = []
        for ln, inp, out in self.io_pairs:
            if len(inp) >= 50:
                continue
            if len(out) > len(inp) * 1.5:
                bad.append((ln, len(inp), len(out)))
        self.assertEqual(bad, [], f"Short input over-expanded: {bad}")

    def test_average_compression_ratio_in_range(self):
        """Average ratio across >100-char inputs must be in [0.30, 0.65]."""
        ratios = [
            len(out) / len(inp)
            for _, inp, out in self.io_pairs
            if len(inp) > 100
        ]
        if not ratios:
            self.skipTest("No inputs >100 chars")
        avg = sum(ratios) / len(ratios)
        self.assertGreaterEqual(avg, 0.30, f"Average ratio {avg:.3f} below 0.30")
        self.assertLessEqual(avg, 0.65, f"Average ratio {avg:.3f} above 0.65")

    # -- 4. URL Preservation -----------------------------------------------

    def test_urls_preserved_verbatim(self):
        """Any URL (http/https) in the input must appear verbatim in the output."""
        bad = []
        for ln, inp, out in self.io_pairs:
            for url in URL_PATTERN.findall(inp):
                if url not in out:
                    bad.append((ln, url))
        self.assertEqual(bad, [], f"URLs lost in compression: {bad}")

    # -- 5. Constraint Preservation ----------------------------------------

    def test_constraint_keywords_preserved(self):
        """NEVER / ALWAYS / MUST NOT in input should appear in output (case-insensitive)."""
        bad = []
        for ln, inp, out in self.io_pairs:
            out_lower = out.lower()
            for kw in CONSTRAINT_KEYWORDS:
                if kw in inp and kw.lower() not in out_lower:
                    bad.append((ln, kw))
        self.assertEqual(bad, [], f"Constraint keywords lost: {bad}")

    # -- 6. No Exact Duplicates --------------------------------------------

    def test_no_duplicate_user_messages(self):
        seen = {}
        dupes = []
        for ln, inp, _ in self.io_pairs:
            if inp in seen:
                dupes.append((seen[inp], ln))
            else:
                seen[inp] = ln
        self.assertEqual(dupes, [], f"Duplicate user messages (first_line, dupe_line): {dupes}")

    # -- 7. Character / Encoding Validation --------------------------------

    def test_no_null_bytes_or_control_characters(self):
        """No null bytes or control chars (except tab, newline, carriage return)."""
        control_re = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
        bad = []
        for ln, obj in self.entries:
            if "messages" not in obj:
                continue
            for m in obj["messages"]:
                found = control_re.findall(m.get("content", ""))
                if found:
                    bad.append((ln, m["role"], [hex(ord(c)) for c in found]))
        self.assertEqual(bad, [], f"Control characters found: {bad}")

    def test_valid_utf8(self):
        try:
            with open(self.data_path, "r", encoding="utf-8") as f:
                f.read()
        except UnicodeDecodeError as e:
            self.fail(f"Invalid UTF-8 in {self.data_path}: {e}")


# ---------------------------------------------------------------------------
# Concrete test classes (one per file)
# ---------------------------------------------------------------------------

class TestTrainData(TrainingDataMixin, unittest.TestCase):
    """All quality checks applied to data/train.jsonl."""
    data_path = TRAIN_PATH


class TestValidData(TrainingDataMixin, unittest.TestCase):
    """All quality checks applied to data/valid.jsonl."""
    data_path = VALID_PATH


# ---------------------------------------------------------------------------
# 9. Cross-validation
# ---------------------------------------------------------------------------

class TestCrossValidation(unittest.TestCase):
    """No overlap between train and validation user messages."""

    @classmethod
    def setUpClass(cls):
        train_entries, _ = load_jsonl(TRAIN_PATH)
        valid_entries, _ = load_jsonl(VALID_PATH)
        cls.train_user_msgs = {
            obj["messages"][1]["content"]
            for _, obj in train_entries
            if "messages" in obj and len(obj["messages"]) >= 2
        }
        cls.valid_user_msgs = {
            obj["messages"][1]["content"]
            for _, obj in valid_entries
            if "messages" in obj and len(obj["messages"]) >= 2
        }

    def test_no_user_message_overlap(self):
        overlap = self.train_user_msgs & self.valid_user_msgs
        self.assertEqual(
            len(overlap), 0,
            f"Found {len(overlap)} user messages appearing in both train and valid sets. "
            f"First: {list(overlap)[0][:80]}..." if overlap else ""
        )


# ---------------------------------------------------------------------------
# 8. Dataset Statistics (informational, always passes)
# ---------------------------------------------------------------------------

class TestDatasetStatistics(unittest.TestCase):
    """Print dataset statistics. Informational only -- never fails."""

    @classmethod
    def setUpClass(cls):
        cls.train_entries, _ = load_jsonl(TRAIN_PATH)
        cls.valid_entries, _ = load_jsonl(VALID_PATH)

    @staticmethod
    def _stats_block(entries, label):
        pairs = get_io_pairs(entries)
        if not pairs:
            return f"\n  {label}: no valid entries\n"

        input_lens = [len(inp) for _, inp, _ in pairs]
        output_lens = [len(out) for _, _, out in pairs]
        ratios = [len(out) / len(inp) for _, inp, out in pairs if len(inp) > 0]

        # Bucket ratios
        bucket_labels = [
            "<0.20", "0.20-0.30", "0.30-0.40", "0.40-0.50",
            "0.50-0.60", "0.60-0.70", "0.70-0.80", ">=0.80",
        ]
        buckets = Counter()
        for r in ratios:
            if r < 0.2:
                buckets["<0.20"] += 1
            elif r < 0.3:
                buckets["0.20-0.30"] += 1
            elif r < 0.4:
                buckets["0.30-0.40"] += 1
            elif r < 0.5:
                buckets["0.40-0.50"] += 1
            elif r < 0.6:
                buckets["0.50-0.60"] += 1
            elif r < 0.7:
                buckets["0.60-0.70"] += 1
            elif r < 0.8:
                buckets["0.70-0.80"] += 1
            else:
                buckets[">=0.80"] += 1

        # Content-type counts
        url_count = sum(1 for _, inp, _ in pairs if URL_PATTERN.search(inp))
        path_count = sum(
            1 for _, inp, _ in pairs
            if re.search(r"[/\\][\w.-]+[/\\][\w.-]+", inp)
        )
        code_count = sum(
            1 for _, inp, _ in pairs
            if re.search(r"(```|def |function |class |import |#include|=>)", inp)
        )
        constraint_count = sum(
            1 for _, inp, _ in pairs
            if any(kw in inp for kw in CONSTRAINT_KEYWORDS)
        )

        lines = [
            f"\n{'=' * 60}",
            f"  {label}",
            f"{'=' * 60}",
            f"  Total examples:          {len(pairs)}",
            f"  Avg input length:        {sum(input_lens) / len(input_lens):.0f} chars",
            f"  Avg output length:       {sum(output_lens) / len(output_lens):.0f} chars",
            f"  Avg compression ratio:   {sum(ratios) / len(ratios):.3f}",
            f"  Min compression ratio:   {min(ratios):.3f}",
            f"  Max compression ratio:   {max(ratios):.3f}",
            f"",
            f"  Compression ratio distribution:",
        ]
        for b in bucket_labels:
            c = buckets.get(b, 0)
            bar = "#" * c
            lines.append(f"    {b:>10s}: {c:3d}  {bar}")
        lines += [
            f"",
            f"  Content type counts:",
            f"    URLs:            {url_count}",
            f"    File paths:      {path_count}",
            f"    Code snippets:   {code_count}",
            f"    Constraints:     {constraint_count}",
            f"{'=' * 60}",
        ]
        return "\n".join(lines)

    def test_print_statistics(self):
        """Print dataset statistics (always passes)."""
        output = self._stats_block(self.train_entries, f"train.jsonl ({TRAIN_PATH})")
        output += "\n"
        output += self._stats_block(self.valid_entries, f"valid.jsonl ({VALID_PATH})")
        # Use stderr so it shows even when pytest captures stdout
        print(output, file=sys.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=2)
