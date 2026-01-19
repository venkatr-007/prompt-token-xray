import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime

import requests


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def parse_multipliers(s: str):
    try:
        return [int(x.strip()) for x in s.split(",") if x.strip()]
    except Exception as e:
        raise ValueError(f"Invalid --multipliers '{s}': {e}")


def extract_text(resp_json) -> str:
    try:
        return resp_json["choices"][0]["message"]["content"] or ""
    except Exception:
        return ""


def last_nonempty_line(text: str) -> str:
    for line in reversed(text.splitlines()):
        if line.strip():
            return line.strip()
    return ""


def is_overflow_error(err_json_or_text: str) -> bool:
    t = (err_json_or_text or "").lower()
    # LM Studio often returns strings like "context length" / "not enough" / "overflows"
    return (
        "context" in t and ("overflow" in t or "not enough" in t or "length" in t)
    ) or "overflows" in t


def post_chat(url: str, payload: dict, timeout_s: int):
    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout_s)
    dt = time.time() - t0
    if r.status_code >= 400:
        try:
            err = r.json()
        except Exception:
            err = r.text
        raise RuntimeError(f"HTTP {r.status_code}: {json.dumps(err, ensure_ascii=False)[:2000]}")
    return r.json(), dt


def main():
    ap = argparse.ArgumentParser(description="Prompt-token-XRay boundary runner (ONE CALL per multiplier with sentinel).")
    ap.add_argument("--url", default="http://localhost:1234/v1/chat/completions")
    ap.add_argument("--model", default=None, help="LM Studio model identifier (recommended).")
    ap.add_argument("--system-file", required=True)
    ap.add_argument("--user-file", required=True)
    ap.add_argument("--multipliers", required=True, help="Comma-separated list, e.g. 64,80,84,88")
    ap.add_argument("--max-context", type=int, required=True, help="Model context length you loaded (e.g. 4096/8192)")
    ap.add_argument("--sentinel", required=True, help="Sentinel string that must be echoed as last line")
    ap.add_argument("--sentinel-max-tokens", type=int, default=8, help="Output token budget for echo")
    ap.add_argument("--sentinel-threshold-pct", type=float, default=90.0)
    ap.add_argument("--timeout-s", type=int, default=900)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--outdir", default="results")
    ap.add_argument("--out-prefix", default="ctx_boundary_onecall")

    args = ap.parse_args()

    system_text = read_text(args.system_file)
    user_base = read_text(args.user_file)
    mults = parse_multipliers(args.multipliers)

    os.makedirs(args.outdir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = os.path.join(args.outdir, f"{args.out_prefix}_{ts}.csv")

    fieldnames = [
        "timestamp",
        "model",
        "max_context",
        "multiplier",
        "system_chars",
        "user_chars",
        "combined_prompt_tokens",
        "context_pct",
        "threshold_pct",
        "status",
        "truncation",
        "response_snippet",
        "sentinel_returned",
        "elapsed_s",
        "notes",
    ]

    print(f"Writing CSV: {out_csv}")

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()

        for m in mults:
            user_text = user_base * m
            # minimal probe
            probe = "\n\nEcho the last line exactly.\n" + args.sentinel
            user_final = user_text + probe

            row = {
                "timestamp": ts,
                "model": args.model or "",
                "max_context": args.max_context,
                "multiplier": m,
                "system_chars": len(system_text),
                "user_chars": len(user_text),
                "combined_prompt_tokens": "",
                "context_pct": "",
                "threshold_pct": args.sentinel_threshold_pct,
                "status": "OK",
                "truncation": "",
                "sentinel_returned": "",
                "elapsed_s": "",
                "notes": "",
            }

            payload = {
                "model": args.model,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_final},
                ],
                "max_tokens": args.sentinel_max_tokens,
                "temperature": args.temperature,
                "stream": False,
            }

            try:
                data, dt = post_chat(args.url, payload, args.timeout_s)
                pt = data.get("usage", {}).get("prompt_tokens")
                row["combined_prompt_tokens"] = pt if pt is not None else ""
                if pt is not None:
                    row["context_pct"] = round((pt / args.max_context) * 100.0, 2)
                text = extract_text(data)
                last_line = last_nonempty_line(text)
                clean_out = (text or "").strip()
                clean_out = clean_out.strip("\"\'")
                trunc = (args.sentinel not in clean_out)
                row["truncation"] = "true" if trunc else "false"
                row["response_snippet"] = clean_out.replace("\n", " ")[:120]
                last_line = last_nonempty_line(text)
                row["sentinel_returned"] = last_line[:200]
                row["elapsed_s"] = round(dt, 2)

                # Print summary similar to your other scripts
                pct = row["context_pct"] if row["context_pct"] != "" else "?"
                msg = f"[x{m:>2}] chars(sys={row['system_chars']}, user={row['user_chars']}) tokens(combined={pt}) context%={pct}"
                if trunc:
                    msg += " truncation=true"
                print(msg)

                w.writerow(row)
                f.flush()

                # Optional early-stop if we're past threshold and already truncating
                if pt is not None and (pt / args.max_context * 100.0) >= args.sentinel_threshold_pct and trunc:
                    row2 = {
                        **row,
                        "status": "STOP",
                        "notes": "threshold_reached_and_truncating",
                    }
                    # don't duplicate row in csv; just stop
                    break

            except requests.exceptions.ReadTimeout:
                row["status"] = "TIMEOUT"
                row["notes"] = f"timeout_s={args.timeout_s}"
                print(f"[x{m:>2}] TIMEOUT; stopping further multipliers.")
                w.writerow(row)
                f.flush()
                break
            except Exception as e:
                s = str(e)
                row["notes"] = s[:500]
                if is_overflow_error(s):
                    row["status"] = "OVERFLOW"
                    print(f"[x{m:>2}] OVERFLOW; stopping further multipliers.")
                else:
                    row["status"] = "ERROR"
                    print(f"[x{m:>2}] ERROR: {s[:160]}")
                w.writerow(row)
                f.flush()
                break

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
