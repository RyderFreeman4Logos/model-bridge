# model-bridge development recipes

# Build all workspace crates
build:
	cargo build --workspace

# Run all tests
test:
	cargo test --workspace

# Run clippy lint checks
lint:
	cargo clippy --workspace --all-targets

# Format all code
fmt:
	cargo fmt --all

# Check formatting without modifying
fmt-check:
	cargo fmt --all -- --check

# Alias for lint (used by CSA commit workflow)
clippy: lint

# Run full pre-commit checks
pre-commit: fmt-check lint test
