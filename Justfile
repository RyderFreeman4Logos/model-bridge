# model-bridge development recipes

# Build all workspace crates
build:
	cargo build --workspace

# Run all tests
test:
	cargo nextest run --workspace

# Run clippy lint checks
lint:
	cargo clippy --workspace --all-targets -- -D warnings

# Format all code
fmt:
	cargo fmt --all

# Check formatting without modifying
fmt-check:
	cargo fmt --all -- --check

# Alias for lint (used by CSA commit workflow)
clippy: lint

# Detect monolith files by token/line count using tokuin + parallel
# Fails fast on first file exceeding threshold; blocks commit
# Env: MONOLITH_TOKEN_THRESHOLD (default 8000), MONOLITH_LINE_THRESHOLD (default 800), TOKUIN_MODEL (default gpt-4)
find-monolith-files:
	#!/usr/bin/env bash
	set -euo pipefail
	THRESHOLD_TOKENS="${MONOLITH_TOKEN_THRESHOLD:-8000}"
	THRESHOLD_LINES="${MONOLITH_LINE_THRESHOLD:-800}"
	MODEL="${TOKUIN_MODEL:-gpt-4}"

	_monolith_error() {
	    local file="$1" actual="$2" limit="$3"
	    echo ""
	    echo "=========================================="
	    echo "ERROR: Monolith file detected! ($actual, limit: $limit)"
	    echo "  File: $file"
	    echo "=========================================="
	    echo ""
	    echo "REQUIRED ACTION:"
	    echo "1. Stash your current work first:  git stash push -m 'pre-split'"
	    echo "2. Split this file:                /split-monolith-files"
	    echo "3. After splitting, retry your commit."
	    echo ""
	    echo "WHY: Large files cause context window bloat and degrade LLM performance."
	    echo "IMPORTANT: Stash before splitting so you can recover via 'git stash pop' if splitting fails."
	    echo "=========================================="
	}

	check_file() {
	    local file="$1"
	    local threshold_tokens="$2"
	    local threshold_lines="$3"
	    local model="$4"
	    # --- Explicit excludes (customize per project) ---
	    case "$file" in
	        *.lock|*lock.json|*lock.yaml) return 0 ;;  # package manager locks (Cargo.lock)
	        */AGENTS.md) return 0 ;;                     # auto-generated rule aggregation
	    esac
	    [ -f "$file" ] || return 0
	    grep -Iq '' "$file" 2>/dev/null || return 0  # skip binary files

	    # Fast pre-filter: line count (zero-cost, no external tools)
	    local lines
	    lines=$(wc -l < "$file")
	    if [ "$lines" -gt "$threshold_lines" ]; then
	        _monolith_error "$file" "$lines lines" "$threshold_lines lines"
	        return 1
	    fi

	    # Accurate check: token count (requires tokuin)
	    # Use 'command jq' to bypass shell aliases (e.g. jq --color-output)
	    # Fallback: estimate ~4 chars/token if tokuin fails on non-chat files
	    local tokens
	    tokens=$(tokuin estimate --model "$model" --format json "$file" 2>/dev/null \
	        | command jq -r '.tokens // empty' 2>/dev/null)
	    if [ -z "$tokens" ]; then
	        local bytes
	        bytes=$(wc -c < "$file")
	        tokens=$(( bytes / 4 ))
	    fi
	    if [ "$tokens" -gt "$threshold_tokens" ]; then
	        _monolith_error "$file" "$tokens tokens" "$threshold_tokens tokens"
	        return 1
	    fi
	    return 0
	}
	export -f check_file _monolith_error

	git ls-files \
	    | parallel --halt now,fail=1 check_file {} "$THRESHOLD_TOKENS" "$THRESHOLD_LINES" "$MODEL"

# Run full pre-commit checks
pre-commit: find-monolith-files fmt-check lint test
