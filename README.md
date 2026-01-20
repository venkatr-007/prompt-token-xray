# prompt-token-xray

A small, reproducible harness to measure **prompt token growth**, **chat-template interaction delta**, and **near-context truncation risk** for OpenAI-compatible chat endpoints (tested with LM Studio local server).

## What it measures
For each prompt multiplier (1x, 2x, 4x, ...), it records:

- **system_prompt_tokens**
- **user_prompt_tokens**
- **combined_prompt_tokens**
- **interaction_overhead_tokens** (combined − system_only − user_only)
- **context_pct_used** (requires `--max-context`)
- Optional: **sentinel truncation check** near saturation (`truncation_suspected=true/false`)

## Why this exists
As prompts grow, latency and reliability often degrade. This tool isolates input-side contributors:

- tokenization growth
- chat wrapper/template effects
- context pressure / overflow boundaries
- potential tail truncation near saturation

## Requirements
- Python 3.9+
- LM Studio running locally (OpenAI-compatible server)
- Install deps:
  ```powershell
  pip install -r requirements.txt

requirements.txt is intentionally minimal:

requests

Project layout
prompt-token-xray/
  scripts/
    bench_token_xray.py
    bench_token_xray_sentinel.py
    bench_token_xray_sentinel_v2_renamed.py
    prompt_token_xray_boundary_onecall_v2.py
  prompts/
    system.txt
    user.txt
    system_sentinel.txt
    payload.txt
  results/                 # ignored by git (except results/samples)
  requirements.txt
  README.md
  .gitignore

Setup

Create a virtual environment and install dependencies:

python -m venv .venv
# If PowerShell blocks activation:
#   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt


Start LM Studio local server and load your model (example: qwen3-1.7b).

Note:
--max-context is used only for reporting and sentinel gating.
The actual context window is controlled in LM Studio model settings (context length / n_ctx).

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
  --system-file .\prompts\system_sentinel.txt `
  --user-file .\prompts\user.txt `
  --multipliers 16,20,22,24,25,26 `
  --max-context 2048 `
  --sentinel "ZXCV-1234-END" `
  --sentinel-threshold-pct 90 `
  --sentinel-max-tokens 8 `
  --timeout-s 900 `
  --out-prefix prompt_token_xray_sentinel


Behavior:

* OVERFLOW: prompt did not fit in the model context window; the script writes an OVERFLOW row and stops cleanly.
* truncation_suspected=true: sentinel was not reliably echoed (possible truncation or instruction-following degradation under high context pressure).

A) Make the sentinel probe shorter (exact change)

Sentinel checking can add extra instruction tokens and push you over the edge near saturation. Keep the probe instruction extremely small.
In your sentinel script, build the probe like this:
probe_user = user_text + "\n\nOutput exactly the last line of the user message. No other text.\n" + sentinel
(And ensure the sentinel is the final line.)

Roadmap (short)
Compare chat vs raw completions (if supported) to quantify chat wrapper tax.
Add optional plots / summary report.
Run cross-server comparisons (LM Studio vs llama.cpp server vs vLLM).