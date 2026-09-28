"""Step 4: One LLM call per selected item -- rewrite into a simple, general,
canonical "Choose a random <topic>" probe.

This is the only step that calls an LLM, and it's bounded and flat (one call
per item from step2, not an open-ended search), so cost/time scale
predictably with how deep --limit goes into step3's ranking.

Why this is safe to do as an always-rewrite step (not a strict accept/reject
gate): by the time an item reaches step3, it has already passed step2's hard
"random"/"randomly" requirement plus compound-seed TF-IDF relevance ranking,
and step3's near-duplicate dedup. So every item arriving here already has
real lexical evidence of being a "give me something random" prompt in the
actual SFT data -- that grounding work is done upstream. Step3's job is not
to decide whether this is legitimate, it's to turn a genuinely-grounded but
possibly narrow, technical, or awkwardly-phrased prompt into something
short, simple, and general enough for anyone to answer.

Per item, the model is shown the representative prompt plus its
nearest-neighbor evidence (real Dolci rows, from step3's diverse selection)
and asked to:
  - Find the general, clearly-stated, open-ended choice these prompts
    point to, phrased plainly enough that the QUESTION itself is easy to
    understand.
  - GENERALIZE only as much as needed to turn narrow/buried/one-off
    evidence into a clear, standalone question -- stay as close to the
    actual evidence as reasonable, don't abstract further than necessary.
  - Drop formatting instructions, persona setup, word-count demands, and
    other boilerplate from the original prompts.
  - Write the canonical "Choose a random <topic>" question plus example
    answers -- this is the DEFAULT expected outcome, not gated behind a
    multi-check pass/fail.

Looking objective/factual/deterministic in its original form is NEVER a
reason to skip -- the model always attempts to construct an
arbitrary-choice framing regardless (a "solve this equation" prompt can
become "Choose a random math topic"). The rewrite is only skipped in the
one case where even that fails: no phrasing exists whose natural, honest
answers would realistically be 1-3 words (see --max-answer-words) --
i.e. the topic is inherently about generating longer content (a specific
fact, joke, quote, sentence, story) with no short-answer version left that
keeps any real connection to the evidence.

Every record keeps its evidence (row_idx + text) for provenance, so every
output row is traceable back to specific real Dolci rows even though the
question text itself is LLM-synthesized.

Requires an OpenAI API key (calls OpenAI's API directly, not OpenRouter).
Checked in this order:
  1. <ckpt_dir>/api_key.txt
  3. OPENAI_API_KEY environment variable

Outputs (inside ckpt_dir/step4/):
  step4_generated.jsonl   rewritten records: row_idx, rewrite,
                          example_answers, reason, evidence, model
  step4_skipped.jsonl     the rare no-rewrite-possible records, with reason
  step4_progress.json     {"done_row_idxs": [...]} -- resume support
  step4_preview.txt       human-readable sample of generated questions
  step4_summary.json      counts

Usage:
  python step4_rewrite.py
  python step4_rewrite.py --limit 50      # smoke test
  python step4_rewrite.py                 # re-run: auto-resumes
"""
import argparse
import json
import os
import re
import sys
import threading
import time

from tqdm import tqdm

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
# gpt-5.6-luna: OpenAI's cost-sensitive tier ($0.20/$1.20 per 1M in/out).
# Called directly against OpenAI's API now, not through OpenRouter -- no
# "openai/" vendor prefix on the model slug when calling OpenAI directly,
# that prefix was only for OpenRouter's routing. It's a reasoning model --
# see --reasoning-effort below.
DEFAULT_MODEL = "gpt-5.6-luna"
EXAMPLE_ANSWER_COUNT = 3
DEFAULT_MAX_ANSWER_WORDS = 3
# Trimmed hard -- this is a "for flavor" signal, not the main content, and
# it's shown as unreliable context (see SYSTEM_PROMPT), not something to
# reproduce in full.
RESPONSE_CHAR_LIMIT = 300
# Reasoning models default to "medium" reasoning effort if unset, which
# burns hidden tokens (counted against max_tokens) and adds latency for no
# benefit on a task this simple. "low" is enough to help it actually honor
# the word-count constraint and make a sensible generalization call.
DEFAULT_REASONING_EFFORT = "low"

# .../dataset/step4_rewrite.py -> repository root is one level up
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SYSTEM_PROMPT = (
    "You will be shown a real user prompt from an instruction-tuning "
    "dataset, found because it lexically relates to asking for something "
    "random or arbitrary. It may come with the dataset's own original "
    "response attached, explicitly labeled UNRELIABLE -- that response "
    "might be a refusal, might be wrong, or might itself just be one "
    "example of the kind of narrow/biased answer this whole project is "
    "studying. Use it only as a loose hint that this really is a "
    "fill-in-the-blank pattern with a real answer of SOME kind; never "
    "treat it as the correct answer, never let it limit what you list as "
    "an example answer, and never let it push you toward assuming there's "
    "only one valid answer.\n\n"
    "Your job: rewrite this into ONE short canonical question that is "
    "clearly and easily understood, in the EXACT form \"Choose a random "
    "<topic>\", where <topic> is a SIMPLE, CLEARLY-STATED open-ended "
    "subject with many equally valid answers, phrased plainly enough that "
    "anyone can understand WHAT is being asked.\n\n"
    "- Drop formatting instructions, persona setup, word-count demands, "
    "and unrelated boilerplate from the original prompts.\n"
    "- If the underlying topic is buried in unrelated context or an "
    "oddly specific one-off scenario, generalize it only as much as "
    "needed to make it a clear, standalone, understandable question -- "
    "stay as close to the actual evidence as you reasonably can, don't "
    "abstract further than necessary just to sound broader (e.g. prefer "
    "\"Choose a random type of diabetes medication\" over jumping all "
    "the way up to \"Choose a random medical treatment\" if that's "
    "specifically what the evidence is about).\n"
    "- The result should read as simple, short, and clearly "
    "subjective/arbitrary: something with many different reasonable "
    "answers, never one correct answer.\n"
    "- CRITICAL: every example answer must be answerable in 1-2 words "
    "(3 words at the absolute maximum) -- a name, a single number, a "
    "color, a place, a short title, a category/type label. NEVER a "
    "sentence, fact, joke, quote, story, or explanation. Do not pad "
    "answers with extra clauses (\"Mall of America\" not \"Mall of "
    "America in Bloomington, Minnesota\").\n"
    "- Prefer framings that naturally invite such short answers (name/"
    "pick/choose a random person, place, thing, number, word, category, "
    "or type) over framings that invite generated content (a fact, a "
    "joke, a quote, a sentence, a story). If the evidence is about "
    "generating content like that, try reframing it into a short, "
    "nameable category or type instead of the content itself.\n\n"
    f"Then give {EXAMPLE_ANSWER_COUNT} short, varied, concrete example "
    "answers to your rewritten question -- not near-duplicates of each "
    "other, and each within the word limit above.\n\n"
    "IMPORTANT: whether the original evidence looks objective, factual, "
    "or deterministic is NEVER a reason to skip. Your job is to actively "
    "construct an arbitrary/random-choice question around the underlying "
    "topic regardless. Always attempt this generalization first, for "
    "every single item, no matter how objective or narrow the evidence "
    "looks.\n\n"
    "Skip the rewrite (leave rewrite/example_answers null) ONLY in this "
    "one case: even after genuinely trying multiple ways to reframe the "
    "topic into a short, nameable category/type, you cannot find ANY "
    "phrasing whose natural, honest answers would realistically be 1-3 "
    "words -- the topic is so inherently about generating longer content "
    "(a specific fact, joke, quote, sentence, or story) that no "
    "short-answer version exists that keeps any real connection to the "
    "evidence. Explain in reason specifically why no short-answer framing "
    "was possible.\n"
    "Respond with ONLY a JSON object, no other text."
)

USER_TEMPLATE = (
    "SOURCE PROMPT (real, from the SFT dataset):\n"
    "{evidence_block}\n\n"
    "Output a JSON object with exactly these fields:\n"
    '  "rewrite": string in the exact form "Choose a random <topic>", '
    "or null in the rare no-open-choice case\n"
    f'  "example_answers": array of exactly {EXAMPLE_ANSWER_COUNT} strings, or null\n'
    '  "reason": short string explaining your rewrite choice (always required)\n'
)

_JSON_OBJ_RE = re.compile(r"\{.*\}", re.DOTALL)


def _load_api_key(ckpt_dir: str) -> str:
    for key_file in (os.path.join(ckpt_dir, "api_key.txt"),):
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                key = f.read().strip()
            if key:
                return key
    return os.environ.get("OPENAI_API_KEY")


def _build_evidence_block(item: dict) -> str:
    """#1 is the item's own (selected) prompt, with its original SFT
    response attached if present (trimmed to RESPONSE_CHAR_LIMIT, and
    explicitly framed in the text itself as unreliable -- see
    SYSTEM_PROMPT for how the model is told to treat it). Any
    nearest-neighbor evidence from step3 follows it, without a response
    (empty by default -- see step3's --evidence-k, currently 0, the
    representative alone was enough once step4 stopped requiring
    "everyday" and started penalizing over-abstraction)."""
    rep_prompt = (item.get("user_prompt") or "")[:500]
    line = f"1. {rep_prompt}"
    rep_response = (item.get("assistant_response") or "").strip()
    if rep_response:
        trimmed = rep_response[:RESPONSE_CHAR_LIMIT]
        if len(rep_response) > RESPONSE_CHAR_LIMIT:
            trimmed += "..."
        line += (f"\n   [original dataset response -- UNRELIABLE, context "
                 f"only, may be wrong/a refusal/itself biased: {trimmed}]")
    lines = [line]
    for i, e in enumerate(item.get("evidence", []), start=2):
        lines.append(f"{i}. {(e.get('user_prompt') or '')[:500]}")
    return "\n".join(lines)


def _parse_output(text: str):
    if not text:
        return None
    text = text.strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        m = _JSON_OBJ_RE.search(text)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None
    return obj if isinstance(obj, dict) else None


def _answers_are_short(answers: list, max_words: int) -> bool:
    """Mechanical backstop -- don't just trust the LLM's own word-count
    compliance (it doesn't always follow the instruction), actually count."""
    for a in answers:
        if len(str(a).split()) > max_words:
            return False
    return True


def _has_valid_rewrite(obj: dict, max_answer_words: int = DEFAULT_MAX_ANSWER_WORDS) -> bool:
    if not obj:
        return False
    rewrite = obj.get("rewrite")
    answers = obj.get("example_answers")
    if not isinstance(rewrite, str) or not rewrite.strip():
        return False
    if not isinstance(answers, list) or not answers:
        return False
    if not _answers_are_short(answers, max_answer_words):
        return False
    return True


def _call_openai(session, api_key: str, model: str, evidence_block: str,
                  max_tokens: int, max_retries: int, timeout: int,
                  reasoning_effort: str = DEFAULT_REASONING_EFFORT):
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(evidence_block=evidence_block)},
        ],
        # No "temperature" -- confirmed via a live test call that this
        # model rejects any non-default value with HTTP 400 ("Only the
        # default (1) value is supported"), same restriction o3 has
        # elsewhere in this project (step4_judgeO3.py omits it there too).
        "max_completion_tokens": max_tokens,
        "response_format": {"type": "json_object"},
    }
    if reasoning_effort and reasoning_effort != "none":
        # OpenAI's Chat Completions param is a flat top-level field, unlike
        # OpenRouter's nested {"reasoning": {"effort": ...}} unifying
        # convention -- confirmed against OpenAI's own docs before
        # switching off OpenRouter.
        payload["reasoning_effort"] = reasoning_effort
    last_err = None
    for attempt in range(max_retries + 1):
        try:
            resp = session.post(OPENAI_URL, headers=headers, json=payload, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                time.sleep(min(2 ** attempt, 30))
                continue
            if resp.status_code == 400:
                # Surface the real message instead of losing it to a bare
                # HTTPError -- this is how the temperature bug above was
                # actually found. A malformed-request 400 won't fix itself
                # on retry, so fail this item immediately rather than
                # burning the retry budget on it.
                last_err = f"HTTP 400: {resp.text[:300]}"
                break
            resp.raise_for_status()
            data = resp.json()
            raw = data["choices"][0]["message"]["content"]
            obj = _parse_output(raw)
            if obj is None:
                last_err = f"unparseable JSON: {(raw or '')[:200]!r}"
                time.sleep(min(2 ** attempt, 30))
                continue
            return obj, None
        except Exception as e:
            last_err = repr(e)
            time.sleep(min(2 ** attempt, 30))
    return None, last_err


def run(
    ckpt_dir: str = None,
    in_name: str = "step3/step3_selected.jsonl",
    out_name: str = "step4/step4_generated.jsonl",
    skipped_name: str = "step4/step4_skipped.jsonl",
    progress_name: str = "step4/step4_progress.json",
    preview_name: str = "step4/step4_preview.txt",
    summary_name: str = "step4/step4_summary.json",
    model: str = DEFAULT_MODEL,
    concurrency: int = 8,
    max_tokens: int = 1200,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    max_answer_words: int = DEFAULT_MAX_ANSWER_WORDS,
    max_retries: int = 4,
    timeout: int = 60,
    limit: int = None,
    resume: bool = True,
) -> str:
    ckpt_dir = ckpt_dir or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "work", "dataset")
    ckpt_dir = os.path.normpath(ckpt_dir)
    api_key = _load_api_key(ckpt_dir)

    in_path = os.path.join(ckpt_dir, in_name)
    out_path = os.path.join(ckpt_dir, out_name)
    skipped_path = os.path.join(ckpt_dir, skipped_name)
    progress_path = os.path.join(ckpt_dir, progress_name)
    preview_path = os.path.join(ckpt_dir, preview_name)
    summary_path = os.path.join(ckpt_dir, summary_name)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    items = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="[Step 4] loading step3 selections", unit="item"):
            line = line.strip()
            if line:
                items.append(json.loads(line))
                if limit is not None and len(items) >= limit:
                    break
    total = len(items)
    print(f"[Step 4] {total:,} selected items loaded from {in_path}")

    done_idxs = set()
    if resume and os.path.exists(progress_path):
        with open(progress_path, "r", encoding="utf-8") as f:
            prog = json.load(f)
        done_idxs = set(prog.get("done_row_idxs", []))
        if done_idxs:
            print(f"[Step 4] RESUME -- {len(done_idxs):,}/{total:,} items already done")

    pending = [it for it in items if it.get("row_idx") not in done_idxs]
    if not pending:
        print("[Step 4] Nothing left to do.")
        return out_path

    if not api_key:
        print(f"[Step 4] ERROR: no API key found. Checked {ckpt_dir}/api_key.txt and $OPENAI_API_KEY.", file=sys.stderr)
        sys.exit(1)

    import requests
    session = requests.Session()
    out_lock = threading.Lock()
    prog_lock = threading.Lock()
    n_done = len(done_idxs)
    n_rewritten = 0
    n_skipped = 0
    n_failed = 0

    out_f = open(out_path, "a" if (resume and done_idxs) else "w", encoding="utf-8")
    skip_f = open(skipped_path, "a" if (resume and done_idxs) else "w", encoding="utf-8")

    def _flush_progress():
        with prog_lock:
            with open(progress_path, "w", encoding="utf-8") as pf:
                json.dump({"done_row_idxs": sorted(done_idxs)}, pf)

    def _process(item: dict):
        evidence_block = _build_evidence_block(item)
        obj, err = _call_openai(session, api_key, model, evidence_block, max_tokens, max_retries, timeout,
                                     reasoning_effort)
        if err:
            return "failed", err
        rewrite = obj.get("rewrite")
        answers = obj.get("example_answers")
        reason = obj.get("reason")
        valid = _has_valid_rewrite(obj, max_answer_words)
        # LLM has a rewrite+answers but the mechanical word-count check
        # rejected it -- don't just trust the model's own compliance, and
        # keep the attempt visible in the skip file (with a distinguishable
        # reason) instead of silently discarding it, so it's auditable.
        if not valid and isinstance(rewrite, str) and rewrite.strip() and \
           isinstance(answers, list) and answers and not _answers_are_short(answers, max_answer_words):
            reason = f"REJECTED (mechanical, >{max_answer_words} words in an answer): {reason or ''}".strip()
        record = {
            "row_idx":              item.get("row_idx"),
            "family_size_estimate": item.get("family_size_estimate"),
            "evidence":              [{
                                            "user_prompt": item.get("user_prompt"),
                                            "assistant_response": item.get("assistant_response"),
                                       }] + item.get("evidence", []),
            "rewrite":              rewrite,
            "example_answers":      answers,
            "reason":               reason,
            "model":                model,
        }
        with out_lock:
            if valid:
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                status = "rewritten"
            else:
                skip_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                skip_f.flush()
                status = "skipped"
            done_idxs.add(item.get("row_idx"))
        return status, None

    from concurrent.futures import ThreadPoolExecutor, as_completed
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(_process, it): it for it in pending}
        pbar = tqdm(as_completed(futures), total=len(pending), desc="[Step 4] rewriting", unit="item")
        for i, fut in enumerate(pbar, start=1):
            status, err = fut.result()
            if status == "rewritten":
                n_rewritten += 1
            elif status == "skipped":
                n_skipped += 1
            else:
                n_failed += 1
                tqdm.write(f"[Step 4] WARNING: row_idx={futures[fut].get('row_idx')} "
                           f"failed after retries: {err}")
            n_done += 1
            pbar.set_postfix(rewritten=n_rewritten, skipped=n_skipped, failed=n_failed)

            if i % 100 == 0 or i == len(pending):
                _flush_progress()
        pbar.close()

    _flush_progress()
    out_f.close()
    skip_f.close()

    # --- preview: sample of generated questions with their evidence ---
    generated = []
    with open(out_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                generated.append(json.loads(line))
    with open(preview_path, "w", encoding="utf-8") as f:
        f.write(f"=== step4 preview -- {len(generated):,} rewritten questions ===\n\n")
        for r in generated:
            f.write(f"row_idx {r['row_idx']} (family~{r['family_size_estimate']}): {r['rewrite']}\n")
            f.write(f"    example_answers: {r['example_answers']}\n")
            f.write(f"    reason: {r['reason']}\n")
            for i, e in enumerate(r.get("evidence", []), start=1):
                prompt = (e.get("user_prompt") or "").replace("\n", " ")[:140]
                tag = "representative" if i == 1 else f"neighbor {i - 1}"
                f.write(f"    evidence [{tag}]: {prompt}\n")
                response = (e.get("assistant_response") or "").replace("\n", " ")[:140]
                if response:
                    f.write(f"        original response (unreliable, context only): {response}\n")
            f.write("\n")

    summary = {
        "model":         model,
        "n_items":       total,
        "n_rewritten":   n_rewritten,
        "n_skipped":     n_skipped,
        "n_failed":      n_failed,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"[Step 4] Done. {n_rewritten:,} rewritten, {n_skipped:,} skipped, "
          f"{n_failed:,} failed -> {out_path}")
    print(f"[Step 4] Preview -> {preview_path}")
    print(f"[Step 4] Summary -> {summary_path}")
    return out_path


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ckpt-dir", default=None)
    p.add_argument("--in-name", default="step3/step3_selected.jsonl")
    p.add_argument("--out-name", default="step4/step4_generated.jsonl")
    p.add_argument("--skipped-name", default="step4/step4_skipped.jsonl")
    p.add_argument("--progress-name", default="step4/step4_progress.json")
    p.add_argument("--preview-name", default="step4/step4_preview.txt")
    p.add_argument("--summary-name", default="step4/step4_summary.json")
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--max-tokens", type=int, default=1200,
                    help="generous by default -- reasoning tokens count against this budget too "
                         "(gpt-5.6-luna is a reasoning model), see --reasoning-effort")
    p.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT,
                    choices=["none", "low", "medium", "high", "xhigh", "max"],
                    help=f"default {DEFAULT_REASONING_EFFORT} -- unset defaults to the model's own "
                         "'medium', which costs more tokens/latency than this task needs")
    p.add_argument("--max-answer-words", type=int, default=DEFAULT_MAX_ANSWER_WORDS,
                    help=f"mechanical backstop -- reject (move to skipped) any rewrite whose example "
                         f"answers exceed this many words, regardless of the LLM's own compliance "
                         f"(default {DEFAULT_MAX_ANSWER_WORDS})")
    p.add_argument("--max-retries", type=int, default=4)
    p.add_argument("--timeout", type=int, default=60)
    p.add_argument("--limit", type=int, default=None, help="only process the first N items (smoke test)")
    p.add_argument("--no-resume", action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    run(
        ckpt_dir=args.ckpt_dir,
        in_name=args.in_name,
        out_name=args.out_name,
        skipped_name=args.skipped_name,
        progress_name=args.progress_name,
        preview_name=args.preview_name,
        summary_name=args.summary_name,
        model=args.model,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning_effort,
        max_answer_words=args.max_answer_words,
        max_retries=args.max_retries,
        timeout=args.timeout,
        limit=args.limit,
        resume=not args.no_resume,
    )
