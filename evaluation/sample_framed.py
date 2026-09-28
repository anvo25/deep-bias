"""Step 2 of evaluation: sample answers to every reframing.

Each prompt family has 30 reframings (everyday scenarios that ask for the
same choice). Each reframing is sent once, verbatim, with the answer suffix
appended. Output is append-only JSONL, one row per (entry_id, framing_idx),
so a crashed run resumes where it stopped.

The framing file can be the released framings (fields id, prompt,
framing_idx, framing) or the output of dataset/step6_reframe.py.

Usage:
    python evaluation/sample_framed.py \
        --in work/framings/train.jsonl --out work/raw/my_model/framed.jsonl \
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

from deepbias.client import ANSWER_SUFFIX, TargetClient, resolve_api_key  # noqa: E402


FRAMINGS_PATH = REPO / "work" / "framings" / "train.jsonl"
DEFAULT_CONCURRENCY = 16
DEFAULT_TIMEOUT = 60.0


def load_framings(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            # released framings use id/prompt, dataset/step6_reframe.py
            # writes row_idx/base_question
            if "entry_id" not in r:
                r["entry_id"] = str(r.get("id", r.get("row_idx")))
            r.setdefault("base_question", r.get("prompt"))
            rows.append(r)
    return rows


def load_done(out_path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
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
            fidx = row.get("framing_idx")
            if isinstance(eid, str) and isinstance(fidx, int):
                done.add((eid, fidx))
    return done


def call_one(
    client: TargetClient,
    framing_row: dict,
    temperature: float,
    max_tokens: int,
    answer_suffix: str | None = None,
    parse_refusal: bool = False,
    system: str | None = None,
    pretrained: bool = False,
) -> dict:
    framing = framing_row["framing"]
    suffix = answer_suffix if answer_suffix is not None else ANSWER_SUFFIX
    if pretrained:
        # Base/pretrained model: plain completion prompt, no chat template.
        # Same answer-format instruction as the instruction-tuned path,
        # see sample_direct.py's call_one for the reasoning.
        prompt = f"Q: {framing}\n{suffix}\nA:"
    else:
        prompt = f"{framing}\n\n{suffix}"
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
        "entry_id": framing_row["entry_id"],
        "framing_idx": framing_row["framing_idx"],
        "framing_topic": framing_row.get("framing_topic", ""),
        "option_order": framing_row.get("option_order"),
        "framing": framing,
        "response": response,
        "bucket": framing_row.get("bucket"),
        "tag": framing_row.get("tag"),
        "ts": time.time(),
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
    out_path: Path,
    framings_path: Path,
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
    framings = load_framings(framings_path)
    if ids is not None:
        framings = [f for f in framings if f.get("entry_id") in ids]
    if limit is not None:
        framings = framings[:limit]

    if not framings:
        print("[sample_framed] no framings to process, exiting.", flush=True)
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: set[tuple[str, int]] = load_done(out_path) if resume else set()
    if not resume and out_path.exists():
        out_path.unlink()

    if validate_only:
        from collections import Counter
        per_entry: Counter = Counter()
        for fr in framings:
            if (fr["entry_id"], fr["framing_idx"]) in done:
                per_entry[fr["entry_id"]] += 1
        expected = Counter(fr["entry_id"] for fr in framings)
        incomplete = [(eid, per_entry[eid], expected[eid])
                      for eid in expected
                      if per_entry[eid] != expected[eid]]
        print(
            f"[sample_framed] VALIDATE: entries={len(expected)} "
            f"complete={len(expected) - len(incomplete)} "
            f"incomplete={len(incomplete)}",
            flush=True,
        )
        for eid, n, exp in incomplete[:20]:
            print(f"  {eid}: {n}/{exp}")
        if len(incomplete) > 20:
            print(f"  ... and {len(incomplete) - 20} more")
        return

    tasks: list[dict] = []
    skipped = 0
    for fr in framings:
        key = (fr["entry_id"], fr["framing_idx"])
        if key in done:
            skipped += 1
            continue
        tasks.append(fr)

    print(
        f"[sample_framed] framings={len(framings)} "
        f"already_done={skipped} todo={len(tasks)} "
        f"concurrency={concurrency} model={target_model}",
        flush=True,
    )

    if not tasks:
        print("[sample_framed] nothing to do, all done.", flush=True)
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
        print(f"[sample_framed] brace-format ENABLED, refusal flags "
              f"will be saved to each output row.", flush=True)
    if system is not None:
        print(f"[sample_framed] system prompt set "
              f"({len(system.split())} words, {len(system)} chars).", flush=True)

    with out_path.open("a") as fout, ThreadPoolExecutor(max_workers=concurrency) as pool_ex:
        futures = {
            pool_ex.submit(
                call_one, client, fr, temperature, max_tokens,
                answer_suffix, brace_format, system, pretrained,
            ): (fr["entry_id"], fr["framing_idx"])
            for fr in tasks
        }
        bar = tqdm(
            as_completed(futures), total=len(futures),
            desc="framing-infer", unit="call", smoothing=0.05,
        )
        try:
            for fut in bar:
                try:
                    row = fut.result()
                except Exception as e:
                    n_errors += 1
                    bar.write(f"[sample_framed] task crashed: {e!r}")
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
            bar.write("[sample_framed] interrupted, draining...")
            for f in futures:
                f.cancel()
            raise

    print(
        f"\n[sample_framed] done. written={n_written} errors={n_errors} "
        f"skipped_resume={skipped} -> {out_path}",
        flush=True,
    )


_resolve_api_key = resolve_api_key


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="in_path", type=Path,
                    default=FRAMINGS_PATH,
                    help=f"Input framings JSONL (default: {FRAMINGS_PATH})")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output JSONL of raw framed samples")
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only the first N framing rows (smoke)")
    ap.add_argument("--ids", type=str, default=None,
                    help="Comma-separated entry ids to process (smoke)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY,
                    help=f"Concurrent calls (default {DEFAULT_CONCURRENCY})")
    ap.add_argument("--temperature", type=float, default=0.6,
                    help="Sampling temperature (default 0.6)")
    ap.add_argument("--max-tokens", type=int, default=25,
                    help="Per-call max output tokens")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    ap.add_argument("--target-model", required=True)
    ap.add_argument("--target-base-url", required=True)
    ap.add_argument("--target-api-key", default=None)
    ap.add_argument("--reasoning-effort", default=None)
    ap.add_argument("--brace-format", action="store_true")
    ap.add_argument("--system-prompt-file", type=Path, default=None,
                    help="Read a system message from this file and send it "
                         "as the chat completions system role for every call.")
    ap.add_argument("--pretrained", action="store_true",
                    help="Target is a base/pretrained (not instruction-tuned) "
                         "model. Uses a plain 'Q: ...\\nA:' completion prompt "
                         "via /v1/completions instead of the chat-formatted "
                         "instruction.")
    args = ap.parse_args()

    ids = set(s.strip() for s in args.ids.split(",")) if args.ids else None
    target_api_key = _resolve_api_key(args.target_base_url, args.target_api_key)
    system_prompt = None
    if args.system_prompt_file is not None:
        system_prompt = args.system_prompt_file.read_text().rstrip("\n")
        if not system_prompt:
            raise SystemExit(f"--system-prompt-file is empty: {args.system_prompt_file}")

    run(
        out_path=args.out,
        framings_path=args.in_path,
        limit=args.limit,
        ids=ids,
        concurrency=args.concurrency,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        resume=not args.no_resume,
        validate_only=args.validate_only,
        target_model=args.target_model,
        target_base_url=args.target_base_url,
        target_api_key=target_api_key,
        reasoning_effort=args.reasoning_effort,
        brace_format=args.brace_format,
        system=system_prompt,
        pretrained=args.pretrained,
    )


if __name__ == "__main__":
    main()
