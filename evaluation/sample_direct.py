"""Step 1 of evaluation: sample direct answers for every prompt family.

Each prompt ("Choose a random <topic>.") is sent 30 times with the answer
suffix appended, and randomness comes from temperature alone. Output is
append-only JSONL, one row per (entry_id, replicate_idx), so a crashed run
resumes where it stopped (use --no-resume to start over).

The prompt file can be the released prompts (fields id, prompt) or the
output of dataset/step5_deduplicate.py (fields row_idx, rewrite).

Usage:
    python evaluation/sample_direct.py \
        --pool work/prompts/train.jsonl --out work/raw/my_model/direct.jsonl \
        --target-model <model> --target-base-url http://localhost:8000/v1
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

HERE = Path(__file__).parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

from deepbias.client import ANSWER_SUFFIX, TargetClient  # noqa: E402
from deepbias.client import resolve_api_key as _resolve_api_key  # noqa: E402


DEFAULT_POOL = REPO / "work" / "prompts" / "train.jsonl"
DEFAULT_OUT = None  # required: --out
DEFAULT_CONCURRENCY = 16
DEFAULT_TIMEOUT = 60.0
N_REPLICATES = 30  # matches the framing side's 30 reframes per probe


def load_pool(path: Path) -> list[dict]:
    """Load step4_deduped.jsonl, keep only the fields this stage needs."""
    pool = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            # released prompts use id/prompt, the internal pipeline row_idx/rewrite
            pool.append({"id": str(r.get("id", r.get("row_idx"))),
                         "text": r.get("prompt") or r.get("rewrite")})
    return pool


def load_done(out_path: Path) -> dict[str, set[int]]:
    """Only successfully-completed
    rows (no `error` field) count as done on resume."""
    done: dict[str, set[int]] = {}
    if not out_path.exists():
        return done
    with out_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in row:
                continue
            eid = row.get("entry_id")
            ridx = row.get("replicate_idx")
            if isinstance(eid, str) and isinstance(ridx, int):
                done.setdefault(eid, set()).add(ridx)
    return done


def call_one(
    client: TargetClient,
    entry: dict,
    replicate_idx: int,
    temperature: float,
    max_tokens: int,
    answer_suffix: str | None = None,
    parse_refusal: bool = False,
    system: str | None = None,
    pretrained: bool = False,
) -> dict:
    suffix = answer_suffix if answer_suffix is not None else ANSWER_SUFFIX
    if pretrained:
        # Base/pretrained model: plain completion prompt, no chat template.
        # The answer-format instruction is still included (same text as
        # the instruction-tuned path) since pretraining text often has
        # short-answer instructions like this preceding a short answer, so
        # it may still nudge completion length even without true
        # instruction-following. Stop at the first newline so it can't
        # ramble into a fake "next Q:" continuation. See
        # TargetClient.sample_completion.
        prompt = f"Q: {entry['text']}\n{suffix}\nA:"
    else:
        prompt = f"{entry['text']}\n\n{suffix}"
    err = None
    response = ""
    try:
        if pretrained:
            out = client.sample_completion(
                prompt=prompt, n=1,
                temperature=temperature, max_tokens=max_tokens,
                stop=["\n"],
            )
        else:
            out = client.sample(
                prompt=prompt, n=1,
                temperature=temperature, max_tokens=max_tokens,
                system=system,
            )
        response = out[0] if out else ""
    except Exception as e:
        err = repr(e)
    row = {
        "entry_id":      entry["id"],
        "replicate_idx": replicate_idx,
        "prompt":        prompt,
        "response":      response,
        "ts":            time.time(),
    }
    if err is not None:
        row["error"] = err
    if parse_refusal:
        from deepbias.refusal_filter import is_refusal
        refused, parsed, reason = is_refusal(response)
        row["is_refusal"] = refused
        row["refusal_reason"] = reason
        row["parsed_answer"] = parsed
    return row


def run(
    pool_path: Path,
    out_path: Path,
    limit: int | None,
    ids: set[str] | None,
    concurrency: int,
    temperature: float,
    max_tokens: int,
    resume: bool,
    validate_only: bool,
    target_model: str,
    target_base_url: str,
    target_api_key: str,
    reasoning_effort: str | None = None,
    brace_format: bool = False,
    system: str | None = None,
    pretrained: bool = False,
) -> None:
    pool = load_pool(pool_path)
    if ids is not None:
        pool = [e for e in pool if e["id"] in ids]
    if limit is not None:
        pool = pool[:limit]

    if not pool:
        print("[sample_direct] no entries to process, exiting.", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    done: dict[str, set[int]] = load_done(out_path) if resume else {}
    if not resume and out_path.exists():
        out_path.unlink()

    if validate_only:
        incomplete = []
        for e in pool:
            n = len(done.get(e["id"], ()))
            if n != N_REPLICATES:
                incomplete.append((e["id"], n))
        print(
            f"[sample_direct] VALIDATE: pool={len(pool)} "
            f"complete={len(pool) - len(incomplete)} "
            f"incomplete={len(incomplete)}",
            flush=True,
        )
        for eid, n in incomplete[:20]:
            print(f"  {eid}: {n}/{N_REPLICATES}")
        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more")
        return

    tasks: list[tuple[dict, int]] = []
    skipped = 0
    for e in pool:
        seen = done.get(e["id"], set())
        for r in range(N_REPLICATES):
            if r in seen:
                skipped += 1
                continue
            tasks.append((e, r))

    print(
        f"[sample_direct] pool={len(pool)} N_REP={N_REPLICATES} "
        f"already_done={skipped} todo={len(tasks)} "
        f"concurrency={concurrency} model={target_model}",
        flush=True,
    )

    if not tasks:
        print("[sample_direct] nothing to do, all done.", flush=True)
        return

    client = TargetClient(
        model=target_model,
        base_url=target_base_url,
        api_key=target_api_key,
        timeout=DEFAULT_TIMEOUT,
        reasoning_effort=reasoning_effort,
    )

    write_lock = threading.Lock()
    n_written = 0
    n_errors = 0

    answer_suffix = None
    if brace_format:
        from deepbias.refusal_filter import BRACE_ANSWER_SUFFIX
        answer_suffix = BRACE_ANSWER_SUFFIX
        print("[sample_direct] brace-format ENABLED, refusal flags "
              "will be saved to each output row.", flush=True)

    if system is not None:
        print(f"[sample_direct] system prompt set "
              f"({len(system.split())} words, {len(system)} chars).", flush=True)

    with out_path.open("a") as fout, ThreadPoolExecutor(max_workers=concurrency) as pool_ex:
        futures = {
            pool_ex.submit(
                call_one, client, e, r, temperature, max_tokens,
                answer_suffix, brace_format, system, pretrained,
            ): (e["id"], r)
            for (e, r) in tasks
        }
        bar = tqdm(
            as_completed(futures), total=len(futures),
            desc="baseline", unit="call", smoothing=0.05,
        )
        try:
            for fut in bar:
                try:
                    row = fut.result()
                except Exception as e:
                    n_errors += 1
                    bar.write(f"[sample_direct] task crashed: {e!r}")
                    continue
                if "error" in row:
                    n_errors += 1
                with write_lock:
                    fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                    fout.flush()
                    n_written += 1
                bar.set_postfix(written=n_written, errors=n_errors,
                                 skipped_resume=skipped)
        except KeyboardInterrupt:
            bar.write("[sample_direct] interrupted, draining in-flight tasks...")
            for f in futures:
                f.cancel()
            raise

    print(
        f"\n[sample_direct] done. written={n_written} errors={n_errors} "
        f"skipped_resume={skipped} -> {out_path}",
        flush=True,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool", type=Path, default=DEFAULT_POOL,
                     help=f"Path to step4_deduped.jsonl (default: {DEFAULT_POOL})")
    ap.add_argument("--out", type=Path, required=True,
                     help="Output JSONL of raw direct samples")
    ap.add_argument("--limit", type=int, default=None,
                     help="Process only the first N pool entries (smoke)")
    ap.add_argument("--ids", type=str, default=None,
                     help="Comma-separated entry ids (row_idx as strings) to process (smoke)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                     help=f"Concurrent calls (default {DEFAULT_CONCURRENCY})")
    ap.add_argument("--temperature", type=float, default=0.6,
                     help="Sampling temperature (default 0.6, as in the paper)")
    ap.add_argument("--max-tokens", type=int, default=25,
                     help="Per-call max output tokens")
    ap.add_argument("--no-resume", action="store_true",
                     help="Wipe existing output file and start fresh")
    ap.add_argument("--validate-only", action="store_true",
                     help="Don't sample, just report which entries are incomplete")
    ap.add_argument("--target-model", required=True,
                     help="Target model id")
    ap.add_argument("--target-base-url", required=True,
                     help="OpenAI-compatible endpoint base URL")
    ap.add_argument("--target-api-key", default=None,
                     help="Explicit API key. If unset, read from "
                          "OPENAI_API_KEY / OPENROUTER_API_KEY / "
                          "ANTHROPIC_API_KEY based on the endpoint host.")
    ap.add_argument("--reasoning-effort", default=None,
                     help="OpenAI reasoning_effort param "
                          "(minimal/low/medium/high). Leave unset for "
                          "non-reasoning models.")
    ap.add_argument("--brace-format", action="store_true",
                     help="Append a curly-brace marker to each prompt and "
                          "flag responses without braces (or with "
                          "refusal-shaped contents) as refusals.")
    ap.add_argument("--system-prompt-file", type=Path, default=None,
                     help="Read a system message from this file and send it "
                          "as the chat completions system role for every call.")
    ap.add_argument("--pretrained", action="store_true",
                     help="Target is a base/pretrained (not instruction-tuned) "
                          "model. Uses a plain 'Q: ...\\nA:' completion prompt "
                          "via /v1/completions instead of the chat-formatted "
                          "instruction, since a base model was never trained "
                          "to follow the latter.")
    args = ap.parse_args()

    ids = set(s.strip() for s in args.ids.split(",")) if args.ids else None
    system = args.system_prompt_file.read_text().rstrip("\n") if args.system_prompt_file else None
    if args.system_prompt_file is not None and not system:
        raise SystemExit(f"--system-prompt-file is empty: {args.system_prompt_file}")

    api_key = _resolve_api_key(args.target_base_url, args.target_api_key)

    run(
        pool_path=args.pool,
        out_path=args.out,
        limit=args.limit,
        ids=ids,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        resume=not args.no_resume,
        validate_only=args.validate_only,
        target_model=args.target_model,
        target_base_url=args.target_base_url,
        target_api_key=api_key,
        reasoning_effort=args.reasoning_effort,
        brace_format=args.brace_format,
        system=system,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
