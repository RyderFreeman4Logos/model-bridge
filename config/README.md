# Deployment Config Groups

This directory contains two deployment-specific TOML configs for model-bridge.

## Group A (`group-a.toml`)
- Purpose: internal testing by the researcher.
- Listen: `0.0.0.0:8080`
- Backends (Ollama on localhost):
  - `qwen3-8b-heretic`
  - `qwen3:8b` (official baseline)
- Client key: one internal placeholder API key.

## Group B (`group-b.toml`)
- Purpose: user annotation for mb-feedback users (investors/customers).
- Listen: `0.0.0.0:8081`
- Backends (Ollama on remote Tailscale host):
  - `qwen3-8b-ultimate`
  - `qwen3:8b` (official baseline)
- Client keys: placeholder API keys for annotation users.

## Feedback (Group B)
Feedback logging is controlled by runtime feature + environment variable in current code:
- Build/run with `feedback` feature enabled.
- Set `MB_FEEDBACK_DB_PATH` to your SQLite file path.

Example:
```bash
MB_FEEDBACK_DB_PATH=/var/lib/model-bridge/group-b-feedback.sqlite mb --config config/group-b.toml
```
