import argparse
import csv
import datetime as dt
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import requests


def now_stamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_user_prompt(base: str, multiplier: int) -> str:
    # Separator prevents accidental word-joining that can alter tokenization.
    sep = "\n"
    return sep.join([base] * multiplier)


def post_chat_completion(
    url: str,
    model: str,
    messages: List[Dict[str, str]],
    max_tokens: int,
    temperature: float,
    timeout_s: int,
) -> Tuple[Optional[int], Optional[int], Optional[int], Dict[str, Any], float]:
    """
    Returns:
      prompt_tokens, completion_tokens, total_tokens, raw_json, elapsed_seconds

    If server doesn't return usage, tokens are None.
    """
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,     # keep minimal; we're measuring prompt side
        "temperature": temperature,
        "stream": False,
    }

    t0 = time.time()
    r = requests.post(url, json=payload, timeout=timeout_s)
    elapsed = time.time() - t0

    # Raise with helpful diagnostics
    if r.status_code != 200:
        try:
            err = r.json()
        except Exception:
            err = {"text": r.text}
        raise RuntimeError(f"HTTP {r.status_code} from server: {json.dumps(err, ensure_ascii=False)[:2000]}")

    data = r.json()
    usage = data.get("usage") or {}
    prompt_tokens = usage.get("prompt_tokens")
    completion_tokens = usage.get("completion_tokens")
    total_tokens = usage.get("total_tokens")

    # Normalize to int or None
    prompt_tokens = int(prompt_tokens) if prompt_tokens is not None else None
    completion_tokens = int(completion_tokens) if completion_tokens is not None else None
    total_tokens = int(total_tokens) if total_tokens is not None else None

    return prompt_tokens, completion_tokens, total_tokens, data, elapsed


def try_single_message_variants(
    url: str,
    model: str,
    system_text: str,
    user_text: str,
    timeout_s: int,
    max_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Try the cleanest isolation first:
      A) system-only: [system]
      B) user-only:   [user]
      C) combined:    [system, user]

    Returns dict with token counts and notes.
    """
    notes: List[str] = []
    out: Dict[str, Any] = {}

    # A) system-only
    pt_sys, _, _, _, _ = post_chat_completion(
        url=url,
        model=model,
        messages=[{"role": "system", "content": system_text}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )

    # B) user-only
    pt_user, _, _, _, _ = post_chat_completion(
        url=url,
        model=model,
        messages=[{"role": "user", "content": user_text}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )

    # C) combined
    pt_combined, _, _, _, _ = post_chat_completion(
        url=url,
        model=model,
        messages=[{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )

    out["system_prompt_tokens"] = pt_sys
    out["user_prompt_tokens"] = pt_user
    out["combined_prompt_tokens"] = pt_combined

    # Interaction overhead (may be negative/None if usage not returned)
    if pt_sys is not None and pt_user is not None and pt_combined is not None:
        out["interaction_overhead_tokens"] = pt_combined - pt_sys - pt_user
    else:
        out["interaction_overhead_tokens"] = None
        notes.append("Server did not return usage.prompt_tokens; cannot compute overhead precisely.")

    return out, notes


def try_two_message_empty_fallback(
    url: str,
    model: str,
    system_text: str,
    user_text: str,
    timeout_s: int,
    max_tokens: int,
    temperature: float,
) -> Tuple[Dict[str, Any], List[str]]:
    """
    Fallback when server rejects single-message requests.
    Uses a stable 2-message wrapper and empties one side.

    A) system-only-ish: [system, user:""]
    B) user-only-ish:   [system:"", user]
    C) combined:        [system, user]
    """
    notes: List[str] = ["Used 2-message empty-content fallback; overhead estimate includes wrapper cost with empty messages."]
    out: Dict[str, Any] = {}

    pt_sys, _, _, _, _ = post_chat_completion(
        url=url,
        model=model,
        messages=[{"role": "system", "content": system_text}, {"role": "user", "content": ""}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )

    pt_user, _, _, _, _ = post_chat_completion(
        url=url,
        model=model,
        messages=[{"role": "system", "content": ""}, {"role": "user", "content": user_text}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )

    pt_combined, _, _, _, _ = post_chat_completion(
        url=url,
        model=model,
        messages=[{"role": "system", "content": system_text}, {"role": "user", "content": user_text}],
        max_tokens=max_tokens,
        temperature=temperature,
        timeout_s=timeout_s,
    )

    out["system_prompt_tokens"] = pt_sys
    out["user_prompt_tokens"] = pt_user
    out["combined_prompt_tokens"] = pt_combined

    if pt_sys is not None and pt_user is not None and pt_combined is not None:
        out["interaction_overhead_tokens"] = pt_combined - pt_sys - pt_user
    else:
        out["interaction_overhead_tokens"] = None
        notes.append("Server did not return usage.prompt_tokens; cannot compute overhead precisely.")

    return out, notes


def is_context_overflow_error(err: Exception) -> bool:
    """Best-effort detection of LM Studio / llama.cpp context overflow errors."""
    msg = str(err).lower()
    # Common LM Studio error phrasing observed:
    # "Trying to keep the first ... tokens when context the overflows. However, the model is loaded with context length of only ..."
    if "context" not in msg:
        return False
    overflow_markers = [
        "overflows",
        "overflow",
        "context length",
        "loaded with context length",
        "not enough",
    ]
    return any(m in msg for m in overflow_markers)


def main() -> int:
    ap = argparse.ArgumentParser(description="prompt-token-xray: measure prompt token growth and chat-template overhead (OpenAI-compatible, LM Studio).")
    ap.add_argument("--url", default="http://localhost:1234/v1/chat/completions", help="Chat completions endpoint URL.")
    ap.add_argument("--model", default="qwen3-1.7b", help="Model name as expected by the server.")
    ap.add_argument("--system-file", default="", help="Path to system prompt file (text).")
    ap.add_argument("--user-file", default="", help="Path to base user prompt file (text).")
    ap.add_argument("--system", default="You are a concise assistant.", help="System prompt (used if --system-file not set).")
    ap.add_argument("--user", default="Reply with OK.", help="Base user prompt (used if --user-file not set).")
    ap.add_argument("--multipliers", default="1,4,8,16,32,64", help="Comma-separated prompt multipliers.")
    ap.add_argument("--timeout-s", type=int, default=300, help="HTTP timeout in seconds.")
    ap.add_argument("--max-tokens", type=int, default=1, help="Max output tokens (keep small; we measure prompt tokens).")
    ap.add_argument("--temperature", type=float, default=0.0, help="Temperature (not important here).")
    ap.add_argument("--max-context", type=int, default=0, help="Optional: max context tokens for context_pct_used.")
    ap.add_argument("--outdir", default="results", help="Output directory for CSV.")
    ap.add_argument("--out-prefix", default="prompt_token_xray", help="CSV filename prefix (without timestamp).")
    args = ap.parse_args()
    system_text = read_text(args.system_file) if args.system_file else args.system
    user_base = read_text(args.user_file) if args.user_file else args.user

    # Guardrail: warn early if prompt files are empty/whitespace.
    if len(system_text.strip()) == 0:
        print("WARNING: system prompt is empty (system_chars=0). Check --system-file path/content.", file=sys.stderr)
    if len(user_base.strip()) == 0:
        print("WARNING: user prompt is empty (user_chars~0). Check --user-file path/content.", file=sys.stderr)
    try:
        multipliers = [int(x.strip()) for x in args.multipliers.split(",") if x.strip()]
        if not multipliers:
            raise ValueError("No multipliers provided.")
    except Exception as e:
        print(f"Invalid --multipliers: {e}", file=sys.stderr)
        return 2

    ensure_dir(args.outdir)
    out_csv = os.path.join(args.outdir, f"{args.out_prefix}_{now_stamp()}.csv")

    fieldnames = [
        "timestamp",
        "model",
        "prompt_multiplier",
        "system_chars",
        "user_chars",
        "system_prompt_tokens",
        "user_prompt_tokens",
        "combined_prompt_tokens",
        "interaction_overhead_tokens",
        "max_context",
        "context_pct_used",
        "notes",
    ]

    print(f"Writing CSV: {out_csv}")
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for m in multipliers:
            user_text = build_user_prompt(user_base, m)

            row = {
                "timestamp": dt.datetime.now().isoformat(timespec="seconds"),
                "model": args.model,
                "prompt_multiplier": m,
                "system_chars": len(system_text),
                "user_chars": len(user_text),
                "max_context": args.max_context if args.max_context else "",
                "context_pct_used": "",
                "notes": "",
            }

            # Preferred method
            notes: List[str] = []
            token_info: Dict[str, Any] = {}
            try:
                token_info, local_notes = try_single_message_variants(
                    url=args.url,
                    model=args.model,
                    system_text=system_text,
                    user_text=user_text,
                    timeout_s=args.timeout_s,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
                notes.extend(local_notes)
            except Exception as e:
                # If we hit context overflow, record a row and stop the sweep gracefully.
                if is_context_overflow_error(e):
                    row.update({
                        "system_prompt_tokens": None,
                        "user_prompt_tokens": None,
                        "combined_prompt_tokens": None,
                        "interaction_overhead_tokens": None,
                    })
                    row["notes"] = ("OVERFLOW: " + str(e))[:500]
                    w.writerow(row)
                    print(f"[x{m:>2}] OVERFLOW; stopping further multipliers.")
                    break

                # Otherwise, attempt the fallback (some servers reject single-message isolation).
                notes.append(
                    f"Single-message isolation failed; using fallback. Reason: {type(e).__name__}: {e}"
                )
                try:
                    token_info, local_notes = try_two_message_empty_fallback(
                        url=args.url,
                        model=args.model,
                        system_text=system_text,
                        user_text=user_text,
                        timeout_s=args.timeout_s,
                        max_tokens=args.max_tokens,
                        temperature=args.temperature,
                    )
                    notes.extend(local_notes)
                except Exception as e2:
                    if is_context_overflow_error(e2):
                        row.update({
                            "system_prompt_tokens": None,
                            "user_prompt_tokens": None,
                            "combined_prompt_tokens": None,
                            "interaction_overhead_tokens": None,
                        })
                        row["notes"] = ("OVERFLOW: " + str(e2))[:500]
                        w.writerow(row)
                        print(f"[x{m:>2}] OVERFLOW; stopping further multipliers.")
                        break
                    raise

            row.update(token_info)

            # Context % used (only if we have tokens + max_context)
            if args.max_context and row.get("combined_prompt_tokens") is not None:
                try:
                    pct = (float(row["combined_prompt_tokens"]) / float(args.max_context)) * 100.0
                    row["context_pct_used"] = f"{pct:.2f}"
                    if pct >= 90.0:
                        notes.append("WARNING: Context usage >= 90% (near saturation).")
                except Exception:
                    pass

            row["notes"] = " | ".join(notes).strip()

            # Human-readable console line
            sys_t = row.get("system_prompt_tokens")
            usr_t = row.get("user_prompt_tokens")
            cmb_t = row.get("combined_prompt_tokens")
            ovh = row.get("interaction_overhead_tokens")

            print(
                f"[x{m:>2}] chars(sys={row['system_chars']}, user={row['user_chars']}) "
                f"tokens(sys={sys_t}, user={usr_t}, combined={cmb_t}, overhead={ovh}) "
                + (f"context%={row['context_pct_used']}" if row["context_pct_used"] else "")
            )

            w.writerow(row)

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
