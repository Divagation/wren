#!/usr/bin/env python3
"""
Wren eval harness. Scores compression quality across dimensions:
  - ratio: compression ratio (output/input length)
  - values: are all numbers, flags, codes, paths preserved?
  - negations: are all NEVER/unless/except/don't preserved?
  - branches: are all conditional branches present?
  - steps: are all numbered steps present?
  - overall: weighted composite score

Usage:
  python3 eval.py                # Run full eval
  python3 eval.py --category X   # Run one category
  python3 eval.py --verbose      # Show each test case
  python3 eval.py --json         # Output as JSON
"""

import json
import re
import subprocess
import sys
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path

WREN_DIR = Path(__file__).parent
EVAL_DATA = WREN_DIR / "data" / "eval.jsonl"
WREN_PY = WREN_DIR / "wren.py"
PYTHON = WREN_DIR / ".venv" / "bin" / "python3"


@dataclass
class TestCase:
    category: str
    input: str
    # What MUST appear in output (substrings)
    required: list[str] = field(default_factory=list)
    # Values that must be preserved (numbers, flags, codes)
    values: list[str] = field(default_factory=list)
    # Negation words that must survive
    negations: list[str] = field(default_factory=list)
    # Number of conditional branches expected
    branch_count: int = 0
    # Number of ordered steps expected
    step_count: int = 0
    # Target ratio range (min, max)
    target_ratio: tuple[float, float] = (0.30, 0.70)


@dataclass
class Result:
    category: str
    input_len: int
    output_len: int
    ratio: float
    ratio_score: float  # 1.0 if in target range, partial otherwise
    value_score: float  # fraction of values preserved
    negation_score: float  # fraction of negations preserved
    branch_score: float  # fraction of branches preserved
    step_score: float  # fraction of steps preserved
    required_score: float  # fraction of required strings found
    overall: float  # weighted composite
    output: str
    missed_values: list[str] = field(default_factory=list)
    missed_negations: list[str] = field(default_factory=list)
    missed_required: list[str] = field(default_factory=list)


def compress(text: str) -> str:
    result = subprocess.run(
        [str(PYTHON), str(WREN_PY)],
        input=text,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def score_ratio(ratio: float, target: tuple[float, float]) -> float:
    lo, hi = target
    if lo <= ratio <= hi:
        return 1.0
    if ratio < lo:
        # Too aggressive -- penalize proportionally
        return max(0, ratio / lo)
    # Too conservative
    return max(0, 1.0 - (ratio - hi) / (1.0 - hi))


def score_values(output: str, values: list[str]) -> tuple[float, list[str]]:
    if not values:
        return 1.0, []
    out_lower = output.lower()
    missed = [v for v in values if v.lower() not in out_lower]
    return (len(values) - len(missed)) / len(values), missed


def score_negations(output: str, negations: list[str]) -> tuple[float, list[str]]:
    if not negations:
        return 1.0, []
    out_lower = output.lower()
    missed = [n for n in negations if n.lower() not in out_lower]
    return (len(negations) - len(missed)) / len(negations), missed


def score_branches(output: str, branch_count: int) -> float:
    if branch_count == 0:
        return 1.0
    # Count distinct conditional indicators
    indicators = re.findall(
        r'(?:if |when |for |else|otherwise|unknown|default|fallback)',
        output.lower()
    )
    # Also count colon-separated entries like "Python: ..., JS: ..., Go: ..."
    colon_entries = re.findall(r'[A-Z][a-z]+(?:/[A-Z]+)*:', output)
    found = max(len(set(indicators)), len(colon_entries))
    return min(1.0, found / branch_count)


def score_steps(output: str, step_count: int) -> float:
    if step_count == 0:
        return 1.0
    # Count numbered steps like "1)", "2.", "step 1", etc.
    numbered = re.findall(r'(?:^|\s)(\d+)[).\s]', output)
    return min(1.0, len(set(numbered)) / step_count)


def score_required(output: str, required: list[str]) -> tuple[float, list[str]]:
    if not required:
        return 1.0, []
    out_lower = output.lower()
    missed = [r for r in required if r.lower() not in out_lower]
    return (len(required) - len(missed)) / len(required), missed


def evaluate(test: TestCase) -> Result:
    output = compress(test.input)
    ratio = len(output) / len(test.input) if len(test.input) > 0 else 1.0

    r_score = score_ratio(ratio, test.target_ratio)
    v_score, v_missed = score_values(output, test.values)
    n_score, n_missed = score_negations(output, test.negations)
    b_score = score_branches(output, test.branch_count)
    s_score = score_steps(output, test.step_count)
    req_score, req_missed = score_required(output, test.required)

    # Weighted composite: preservation matters more than ratio
    overall = (
        r_score * 0.15
        + v_score * 0.25
        + n_score * 0.25
        + b_score * 0.10
        + s_score * 0.10
        + req_score * 0.15
    )

    return Result(
        category=test.category,
        input_len=len(test.input),
        output_len=len(output),
        ratio=ratio,
        ratio_score=r_score,
        value_score=v_score,
        negation_score=n_score,
        branch_score=b_score,
        step_score=s_score,
        required_score=req_score,
        overall=overall,
        output=output,
        missed_values=v_missed,
        missed_negations=n_missed,
        missed_required=req_missed,
    )


# ── Test cases ──────────────────────────────────────────────

TESTS = [
    # === NEGATION PRESERVATION ===
    TestCase(
        category="negation",
        input="NEVER run destructive git commands like push --force, reset --hard, checkout ., restore ., clean -f, or branch -D unless the user explicitly requests these actions. Taking unauthorized destructive actions is unhelpful and can result in lost work. Always create NEW commits rather than amending, unless the user explicitly requests a git amend.",
        negations=["never", "unless"],
        values=["push --force", "reset --hard", "clean -f", "branch -D"],
        required=["unless", "explicitly"],
    ),
    TestCase(
        category="negation",
        input="Do NOT use the Bash tool to run grep, cat, head, tail, sed, or awk commands unless explicitly instructed. Instead, use the appropriate dedicated tool. Using dedicated tools provides a much better experience for the user.",
        negations=["not", "unless"],
        required=["unless", "explicitly"],
        values=["grep", "cat", "head", "tail", "sed", "awk"],
    ),
    TestCase(
        category="negation",
        input="NEVER push to the remote repository unless the user explicitly asks you to do so. Do NOT use interactive flags like -i with git rebase or git add since they require interactive input which is not supported. Do not use --no-edit with git rebase as it is not a valid option.",
        negations=["never", "not", "unless"],
        values=["--no-edit", "-i"],
        required=["unless", "explicitly"],
    ),
    TestCase(
        category="negation",
        input="Never skip hooks with --no-verify or bypass signing with --no-gpg-sign unless the user has explicitly asked for it. If a hook fails, investigate and fix the underlying issue rather than bypassing it.",
        negations=["never", "unless"],
        values=["--no-verify", "--no-gpg-sign"],
        required=["unless", "explicitly"],
    ),
    TestCase(
        category="negation",
        input="Do not create documentation files or README files unless explicitly requested by the user. Never write new files when you can edit existing ones. Avoid adding docstrings, comments, or type annotations to code you did not change.",
        negations=["not", "never", "unless", "avoid"],
        required=["unless", "explicitly"],
    ),
    TestCase(
        category="negation",
        input="IMPORTANT: Do NOT add error handling, fallbacks, or validation for scenarios that cannot happen. Trust internal code and framework guarantees. Only validate at system boundaries like user input and external APIs. Do not use feature flags or backwards-compatibility shims when you can just change the code.",
        negations=["not", "cannot", "do not"],
        required=["system boundaries", "user input"],
    ),
    TestCase(
        category="negation",
        input="Never log the full API key or token value. Only log the last 4 characters for debugging purposes. Do not store credentials in environment variables that are visible to child processes. Avoid passing secrets as command line arguments since they appear in process listings.",
        negations=["never", "not", "avoid"],
        values=["4 characters"],
        required=["last 4"],
    ),
    TestCase(
        category="negation",
        input="Do not retry failed requests more than 3 times. Never retry on 4xx errors since they indicate client mistakes that won't be fixed by retrying. Only retry on 5xx errors and network timeouts. Use exponential backoff with a maximum delay of 30 seconds between retries.",
        negations=["not", "never"],
        values=["3 times", "4xx", "5xx", "30 seconds"],
        required=["exponential backoff"],
    ),

    # === VALUE PRESERVATION ===
    TestCase(
        category="values",
        input="The application uses a rate limiter configured with a maximum of 100 requests per minute per user, with a burst allowance of 20 additional requests. If the limit is exceeded, return HTTP 429 with a Retry-After header set to 60 seconds. The rate limiter uses a sliding window algorithm with Redis as the backing store on port 6379.",
        values=["100", "20", "429", "60", "6379", "sliding window"],
        required=["per user", "Retry-After"],
    ),
    TestCase(
        category="values",
        input="All API endpoints must validate the JWT token in the Authorization header. Tokens expire after 3600 seconds. If the token is expired, return 401 with error code TOKEN_EXPIRED. If the token signature is invalid, return 401 with error code INVALID_TOKEN. Never log the full token value; only log the last 4 characters for debugging.",
        values=["3600", "401", "TOKEN_EXPIRED", "INVALID_TOKEN", "4 characters"],
        negations=["never"],
        required=["Authorization"],
    ),
    TestCase(
        category="values",
        input="Configure the connection pool with a minimum of 5 connections, maximum of 50, and an idle timeout of 300 seconds. Set the connection acquisition timeout to 10 seconds. Enable SSL with certificate validation using the CA bundle at /etc/ssl/certs/ca-certificates.crt. Log slow queries that exceed 500 milliseconds.",
        values=["5", "50", "300", "10", "500", "/etc/ssl/certs/ca-certificates.crt"],
    ),
    TestCase(
        category="values",
        input="This tool allows you to execute shell commands on the local filesystem. When running commands, please be careful to avoid any destructive operations that could damage the user's system. Always quote file paths that contain spaces with double quotes. Try to maintain your current working directory throughout the session by using absolute paths whenever possible. You may specify an optional timeout in milliseconds, up to a maximum of 600000 milliseconds, which is equivalent to 10 minutes.",
        values=["600000", "10 minutes", "double quotes"],
        negations=["avoid"],
        required=["absolute paths", "quote"],
    ),
    TestCase(
        category="values",
        input="Set the Kubernetes resource limits to 512Mi memory and 500m CPU for the application container. The readiness probe should check /health on port 8080 every 15 seconds with a timeout of 5 seconds and a failure threshold of 3. The liveness probe uses the same endpoint but with an initial delay of 30 seconds.",
        values=["512Mi", "500m", "/health", "8080", "15", "5", "3", "30"],
    ),
    TestCase(
        category="values",
        input="The CI pipeline runs on every push to main and on pull requests targeting main. Build timeout is 20 minutes. Tests must achieve at least 80% code coverage. The Docker image is tagged with the git SHA and pushed to registry.example.com/app. Deploy to staging automatically but require manual approval for production.",
        values=["20 minutes", "80%", "registry.example.com/app"],
        required=["manual approval", "production"],
    ),
    TestCase(
        category="values",
        input="Enable gzip compression for responses larger than 1024 bytes. Compress text/html, text/css, application/json, and application/javascript. Set compression level to 6. Cache compressed responses for 3600 seconds. The maximum request body size is 50 megabytes.",
        values=["1024", "6", "3600", "50"],
    ),
    TestCase(
        category="values",
        input="The database backup runs daily at 02:00 UTC via a cron job. Backups are stored in s3://backups/postgres/ with a retention period of 30 days. Point-in-time recovery is enabled with WAL archiving to s3://wal-archive/. The RPO is 5 minutes and the RTO is 1 hour.",
        values=["02:00", "s3://backups/postgres/", "30 days", "s3://wal-archive/", "5 minutes", "1 hour"],
    ),

    # === CONDITIONAL BRANCHES ===
    TestCase(
        category="branches",
        input="If the file is a Python script, use black for formatting with a line length of 88. If it is a JavaScript or TypeScript file, use prettier with single quotes, no semicolons, and a print width of 100. For Go files, always run gofmt. For Rust files, use rustfmt with edition 2021. If the file type is unknown, skip formatting entirely.",
        branch_count=5,
        values=["88", "100", "2021"],
        required=["black", "prettier", "gofmt", "rustfmt", "skip"],
    ),
    TestCase(
        category="branches",
        input="If the error is a 404, return a friendly 'not found' page with a search bar. If it's a 401, redirect to the login page at /auth/login. For 403 errors, show an access denied message with a link to request permissions. If it's a 500 error, log the full stack trace and show a generic error page. For rate limit errors (429), show a countdown timer based on the Retry-After header.",
        branch_count=5,
        values=["404", "401", "/auth/login", "403", "500", "429", "Retry-After"],
    ),
    TestCase(
        category="branches",
        input="In production, enable full request logging with a 30-day retention. In staging, enable debug logging but with only 7-day retention. In development, log everything to stdout with no retention limit. In CI environments, only log errors and warnings to reduce noise.",
        branch_count=4,
        values=["30-day", "7-day"],
        required=["production", "staging", "development"],
    ),
    TestCase(
        category="branches",
        input="If the user is on a free plan, limit file uploads to 10MB and 100 per month. For the pro plan, allow up to 100MB per file and 1000 uploads per month. Enterprise users get unlimited file size and unlimited uploads. If the account is suspended, block all uploads and show a billing page.",
        branch_count=4,
        values=["10MB", "100", "100MB", "1000"],
        required=["free", "pro", "enterprise", "suspended"],
    ),
    TestCase(
        category="branches",
        input="For GET requests, cache the response for 300 seconds with stale-while-revalidate of 60 seconds. For POST and PUT requests, invalidate the cache for the affected resource. For DELETE requests, invalidate the cache and all related resources. OPTIONS requests should always return immediately with CORS headers without hitting the backend.",
        branch_count=4,
        values=["300", "60"],
        required=["GET", "POST", "DELETE", "OPTIONS"],
    ),

    # === STEP PRESERVATION ===
    TestCase(
        category="steps",
        input="Database migration procedure: 1) Take a full backup using pg_dump with --format=custom flag. 2) Put the application in maintenance mode. 3) Run the migration script at db/migrations/0042_add_indexes.sql. 4) Verify the migration by running db/verify_schema.py. 5) If verification fails, restore from backup using pg_restore. 6) Remove maintenance mode only after successful verification.",
        step_count=6,
        values=["pg_dump", "--format=custom", "db/migrations/0042_add_indexes.sql", "db/verify_schema.py", "pg_restore"],
        required=["maintenance"],
    ),
    TestCase(
        category="steps",
        input="Deployment rollback procedure: 1) Identify the last known good version from the deploy log. 2) Run kubectl rollout undo deployment/app-server in the production namespace. 3) Verify pods are healthy with kubectl get pods. 4) Check the /health endpoint returns 200. 5) Monitor error rates in Grafana for 10 minutes. 6) If error rates spike, escalate to the on-call engineer at oncall@company.com. 7) Update the incident channel in Slack with the rollback status.",
        step_count=7,
        values=["kubectl rollout undo", "200", "10 minutes", "oncall@company.com"],
    ),
    TestCase(
        category="steps",
        input="To create a new release: 1) Ensure all tests pass on the main branch. 2) Run npm version patch to bump the version. 3) Generate the changelog with npx conventional-changelog. 4) Commit the version bump and changelog. 5) Create a git tag with the version number. 6) Push the tag to origin. 7) GitHub Actions will automatically build and publish to npm. 8) Verify the package is available at npmjs.com/package/our-lib.",
        step_count=8,
        values=["npm version patch", "conventional-changelog", "npmjs.com/package/our-lib"],
    ),
    TestCase(
        category="steps",
        input="SSL certificate renewal: 1) Generate a new CSR using openssl req -new -key server.key -out server.csr. 2) Submit the CSR to the certificate authority. 3) Download the signed certificate and intermediate chain. 4) Concatenate the certificate and chain into fullchain.pem. 5) Deploy fullchain.pem and server.key to /etc/nginx/ssl/. 6) Test the configuration with nginx -t. 7) Reload nginx with systemctl reload nginx. 8) Verify the new certificate with openssl s_client -connect localhost:443.",
        step_count=8,
        values=["openssl req", "server.csr", "fullchain.pem", "/etc/nginx/ssl/", "nginx -t", "systemctl reload nginx", "localhost:443"],
    ),
    TestCase(
        category="steps",
        input="When the user asks you to create a pull request, follow these steps carefully. 1) Run git status to see all untracked files. IMPORTANT: Never use the -uall flag as it can cause memory issues. 2) Run git diff to see staged and unstaged changes. 3) Check if the current branch tracks a remote branch. 4) Run git log and git diff base...HEAD to understand full commit history. 5) Draft a pull request title under 70 characters and a summary. 6) Create the PR using gh pr create.",
        step_count=6,
        negations=["never"],
        values=["-uall", "70", "gh pr create"],
        required=["git status", "git diff"],
    ),

    # === MIXED / HARD CASES ===
    TestCase(
        category="mixed",
        input="Configure nginx with worker_processes set to auto, worker_connections 4096, keepalive_timeout 65s, client_max_body_size 50m. Enable gzip compression for text/html, text/css, application/json, and application/javascript with gzip_min_length 1024 and gzip_comp_level 6. Set proxy_connect_timeout to 30s and proxy_read_timeout to 120s.",
        values=["auto", "4096", "65s", "50m", "1024", "6", "30s", "120s"],
        target_ratio=(0.40, 0.90),
    ),
    TestCase(
        category="mixed",
        input="Before making any changes to the codebase, please read the relevant files first to understand the existing code structure. Do not create new files unless they are absolutely necessary for achieving your goal. Generally prefer editing an existing file to creating a new one, as this prevents file bloat and builds on existing work more effectively. Avoid giving time estimates or predictions for how long tasks will take.",
        negations=["not", "unless", "avoid"],
        required=["read", "edit"],
    ),
    TestCase(
        category="mixed",
        input="The webhook endpoint at /api/webhooks/stripe accepts POST requests with a JSON body. Verify the signature using the Stripe-Signature header with the webhook secret stored in STRIPE_WEBHOOK_SECRET. If signature verification fails, return 400 immediately. Process payment_intent.succeeded events by updating the order status to 'paid' in the orders table. Process charge.refunded events by setting the status to 'refunded'. Ignore all other event types with a 200 response.",
        values=["Stripe-Signature", "STRIPE_WEBHOOK_SECRET", "400", "200"],
        branch_count=3,
        required=["payment_intent.succeeded", "charge.refunded", "ignore"],
    ),
    TestCase(
        category="mixed",
        input="CRITICAL: Never delete user data without explicit confirmation. All delete operations must be soft deletes by setting deleted_at timestamp. Hard deletes are only permitted after 90 days of soft deletion and must be approved by an admin. The deletion job runs weekly on Sundays at 03:00 UTC. Audit logs must be retained for 7 years regardless of deletion status.",
        negations=["never"],
        values=["90 days", "03:00 UTC", "7 years"],
        required=["soft delete", "deleted_at", "admin"],
    ),
    TestCase(
        category="mixed",
        input="For git commits, follow these rules: use conventional commit format (feat:, fix:, chore:, docs:). Keep the subject line under 72 characters. Do not use past tense in the subject. If the commit includes breaking changes, add a BREAKING CHANGE footer. Never amend published commits as it rewrites shared history. Squash fixup commits before merging to main.",
        negations=["not", "never"],
        values=["72"],
        required=["feat:", "fix:", "BREAKING CHANGE"],
    ),

    # === SHORT PASSTHROUGH ===
    TestCase(
        category="short",
        input="Fix the bug.",
        required=["fix", "bug"],
        target_ratio=(0.90, 1.05),
    ),
    TestCase(
        category="short",
        input="What does this function do?",
        required=["what", "function", "do"],
        target_ratio=(0.90, 1.05),
    ),
    TestCase(
        category="short",
        input="Refactor this to use async/await.",
        required=["refactor", "async/await"],
        target_ratio=(0.90, 1.05),
    ),
    TestCase(
        category="short",
        input="NEVER use dark themes. Always light. No exceptions.",
        negations=["never"],
        required=["never", "light"],
        target_ratio=(0.80, 1.05),
    ),
    TestCase(
        category="short",
        input="Deploy to staging but NOT production.",
        negations=["not"],
        required=["staging", "not", "production"],
        target_ratio=(0.85, 1.10),
    ),
    TestCase(
        category="short",
        input="Check if port 8080 is already in use.",
        values=["8080"],
        required=["port", "8080"],
        target_ratio=(0.80, 1.05),
    ),
    TestCase(
        category="short",
        input="Run the tests and check for failures.",
        required=["tests", "failures"],
        target_ratio=(0.80, 1.05),
    ),
    TestCase(
        category="short",
        input="Show me the error log.",
        required=["error", "log"],
        target_ratio=(0.90, 1.05),
    ),
    TestCase(
        category="short",
        input="Set the timeout to 30 seconds.",
        values=["30"],
        required=["timeout", "30"],
        target_ratio=(0.85, 1.05),
    ),
    TestCase(
        category="short",
        input="Read src/main.rs and explain it.",
        values=["src/main.rs"],
        required=["src/main.rs"],
        target_ratio=(0.80, 1.05),
    ),
    TestCase(
        category="short",
        input="Add a retry with max 3 attempts and 5 second delay.",
        values=["3", "5"],
        required=["retry", "3", "5"],
        target_ratio=(0.80, 1.05),
    ),
    TestCase(
        category="short",
        input="Do NOT merge this PR until CI passes.",
        negations=["not"],
        required=["not", "merge", "CI"],
        target_ratio=(0.85, 1.10),
    ),
]


def run_eval(category=None, verbose=False, as_json=False):
    tests = TESTS if not category else [t for t in TESTS if t.category == category]

    results = []
    for t in tests:
        r = evaluate(t)
        results.append(r)

    if as_json:
        print(json.dumps([asdict(r) for r in results], indent=2))
        return

    # Print results
    categories = {}
    for r in results:
        if r.category not in categories:
            categories[r.category] = []
        categories[r.category].append(r)

    print("WREN EVAL RESULTS")
    print("=" * 70)

    for cat, cat_results in categories.items():
        avg_overall = sum(r.overall for r in cat_results) / len(cat_results)
        avg_ratio = sum(r.ratio for r in cat_results) / len(cat_results)
        avg_values = sum(r.value_score for r in cat_results) / len(cat_results)
        avg_negations = sum(r.negation_score for r in cat_results) / len(cat_results)
        avg_branches = sum(r.branch_score for r in cat_results) / len(cat_results)
        avg_steps = sum(r.step_score for r in cat_results) / len(cat_results)
        avg_required = sum(r.required_score for r in cat_results) / len(cat_results)

        grade = "A" if avg_overall >= 0.9 else "B" if avg_overall >= 0.75 else "C" if avg_overall >= 0.6 else "D" if avg_overall >= 0.4 else "F"

        print(f"\n{cat.upper()} ({len(cat_results)} tests) — {grade} ({avg_overall:.0%})")
        print(f"  Ratio: {avg_ratio:.0%}  Values: {avg_values:.0%}  Negations: {avg_negations:.0%}  Branches: {avg_branches:.0%}  Steps: {avg_steps:.0%}  Required: {avg_required:.0%}")

        if verbose:
            for r in cat_results:
                status = "PASS" if r.overall >= 0.75 else "WARN" if r.overall >= 0.5 else "FAIL"
                print(f"\n  [{status}] {r.input_len}->{r.output_len} ({r.ratio:.0%}) overall={r.overall:.0%}")
                print(f"    OUT: {r.output[:200]}")
                if r.missed_values:
                    print(f"    MISSED VALUES: {r.missed_values}")
                if r.missed_negations:
                    print(f"    MISSED NEGATIONS: {r.missed_negations}")
                if r.missed_required:
                    print(f"    MISSED REQUIRED: {r.missed_required}")

    # Overall summary
    print("\n" + "=" * 70)
    total = len(results)
    avg_overall = sum(r.overall for r in results) / total
    passing = sum(1 for r in results if r.overall >= 0.75)
    warning = sum(1 for r in results if 0.5 <= r.overall < 0.75)
    failing = sum(1 for r in results if r.overall < 0.5)

    grade = "A" if avg_overall >= 0.9 else "B" if avg_overall >= 0.75 else "C" if avg_overall >= 0.6 else "D" if avg_overall >= 0.4 else "F"

    print(f"OVERALL: {grade} ({avg_overall:.0%}) — {total} tests: {passing} pass, {warning} warn, {failing} fail")
    print(f"  Avg ratio: {sum(r.ratio for r in results)/total:.0%}")
    print(f"  Values:    {sum(r.value_score for r in results)/total:.0%}")
    print(f"  Negations: {sum(r.negation_score for r in results)/total:.0%}")
    print(f"  Branches:  {sum(r.branch_score for r in results)/total:.0%}")
    print(f"  Steps:     {sum(r.step_score for r in results)/total:.0%}")
    print(f"  Required:  {sum(r.required_score for r in results)/total:.0%}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wren eval harness")
    parser.add_argument("--category", "-c", help="Filter by category")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show each test case")
    parser.add_argument("--json", "-j", action="store_true", help="Output JSON")
    args = parser.parse_args()
    run_eval(category=args.category, verbose=args.verbose, as_json=args.json)


if __name__ == "__main__":
    main()
