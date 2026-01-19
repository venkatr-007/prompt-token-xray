# prompt-token-xray
    A small, reproducible harness to measure **prompt token growth**, **chat-template interaction delta**, and **near-context truncation risk** for OpenAI-compatible chat endpoints (tested with LM Studio local server).
    This project is the implementation of “Project C / Tokenization + Prompt Template X-ray”.

## What it measures
For each prompt multiplier (1x, 2x, 4x, ...):
    - **system_prompt_tokens**
    - **user_prompt_tokens**
    - **combined_prompt_tokens**
    - **interaction delta** (combined − system_only − user_only)
    - **context_pct_used** (requires `--max-context`)
    - Optional: **sentinel truncation check** near saturation (returns `truncation_suspected=true/false`)

## Why this exists
Latency and JSON reliability often degrade as prompts grow. This tool isolates the input-side reasons:
    - tokenization growth
    - chat wrapper/template effects
    - context pressure / overflow boundaries
    - potential tail truncation near saturation

## Requirements
    - Python 3.9+
    - LM Studio running locally (OpenAI-compatible server)
    - `pip install -r requirements.txt`

`requirements.txt` (minimal):
- `requests`

## Project layout
Recommended structure:
    prompt-token-xray/
    scripts/
    bench_token_xray.py
    bench_token_xray_sentinel.py
    prompts/
    system.txt
    user.txt
    results/ # ignored by git
    requirements.txt
    README.md
    .gitignore


## Setup

Create a virtual environment:

```powershell
    python -m venv .venv
    .\.venv\Scripts\activate
    pip install -r requirements.txt

Start LM Studio local server and load your model (example: qwen3-1.7b).

IMPORTANT:
    The script flag --max-context is used for reporting and gating sentinel checks.
    The actual model context window is controlled in LM Studio model settings (n_ctx / context length).

Run: token accounting (no sentinel)
    python .\scripts\bench_token_xray.py `
      --system-file .\prompts\system.txt `
      --user-file .\prompts\user.txt `
      --multipliers 1,2,4,8,16 `
      --max-context 2048 `
      --out-prefix prompt_token_xray

Output:
    results/prompt_token_xray_YYYYMMDD_HHMMSS.csv

Run: sentinel truncation check (near saturation)
    python .\scripts\bench_token_xray_sentinel.py `
      --system-file .\prompts\system.txt `
      --user-file .\prompts\user.txt `
      --multipliers 16,20,22,24,25,26 `
      --max-context 2048 `
      --sentinel "ZXCV-1234-END" `
      --sentinel-threshold-pct 90 `
      --sentinel-max-tokens 16 `
      --out-prefix prompt_token_xray_sentinel


If a request overflows context, the script will write an OVERFLOW row and stop cleanly.
Interpreting results
    interaction_overhead_tokens (interaction delta)
        Can be negative: shared wrapper tokens reduce total vs “system-only + user-only” measured separately.
    truncation_suspected=true
        Sentinel was not reliably visible/echoed (possible truncation or instruction-following degradation under high context pressure).
    OVERFLOW
        Hard limit: prompt did not fit in model context window.
Roadmap (short)
    Compare chat vs raw completions if supported (/v1/completions) to quantify chat wrapper tax.
    Add optional plots / summary report.
    Run cross-server comparisons (LM Studio vs llama.cpp server vs vLLM).

---
## A) Make the sentinel probe shorter (exact change)
    Right now, sentinel checking typically **adds extra instruction text**, which can push you over the edge near saturation. The fix is to make the probe instruction extremely small.
    In `scripts/bench_token_xray_sentinel.py`, locate `run_sentinel_check(...)` and replace the probe construction with this minimal form:
### Replace the `probe_user = (...)` with:
    ```python
    probe_user = user_text + "\n\nEcho the last line exactly:\n" + sentinel

## GitHub quickstart

```bash
python -m venv .venv
# Windows PowerShell:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Start LM Studio server on port 1234 with your model loaded
python .\scripts\bench_token_xray.py --help
```
