"""Refusal detection for closed-source-model evaluation runs.

When a model is instructed to put its answer in curly braces, e.g.
    Pick a kind of butterfly.
    Pick exactly ONE option.  Respond with only your choice, one short
    word or phrase in curly braces, like: {your_choice}.  No other text.

…we can mechanically detect whether the model actually answered:

  - **Hard refusal**: the response contains no `{...}` token at all.
    Catches verbose refusals like "As a neutral AI, I cannot…",
    "Determining the best X is subjective", "I cannot fulfill this
    request", etc., regardless of wording.
  - **Soft refusal**: a `{...}` is present, but its contents are a
    refusal-shaped token (`{neither}`, `{both}`, `{neutral}`,
    `{it_depends}`, …) or a template echo (`{your_choice}`, `{...}`).

`is_refusal(response)` returns (is_refusal: bool, parsed_answer: str|None,
reason: str|None) so the caller can save the refusal class (e.g.
"no_brace", "soft_token", "subjectivity_prefix", "template_echo") to the
output JSONL and later compute the percentage of prompts where the model
refused, per model.

This module is used by `evaluation/sample_direct.py` and `evaluation/sample_framed.py`
when invoked with `--brace-format` (or BRACE_FORMAT=true in a config),
and is safe to re-run as a post-processing pass over an existing
`responses_raw.jsonl` since detection is purely string-based.
"""
from __future__ import annotations

import re


# Suffix appended to every direct + framed prompt when --brace-format
# is enabled.  Replaces the existing ANSWER_SUFFIX in render_baseline.
BRACE_ANSWER_SUFFIX = (
    "Pick exactly ONE option.  Respond with only your choice, one short "
    "word or phrase in curly braces, like: {your_choice}.  No other text."
)

# Regex used to extract the (first) curly-brace span from a response.
BRACE_RE = re.compile(r"\{([^{}]+)\}")

# Number-word → digit normalization for the parsed_answer field.  The
# downstream cluster judge does lemma containment + embedding merging
# but does NOT merge "seven" with "7" (they share no lemma and their
# embeddings are too far apart to cross the merge threshold).  Apply
# this map at parse time so the random-digit / random-month / random-N
# style prompts collapse cleanly.
_NUMWORD_TO_DIGIT = {
    "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4",
    "five": "5", "six": "6", "seven": "7", "eight": "8", "nine": "9",
    "ten": "10",
}

# Exact-match tokens (lowercased, stripped).  Anything in here counts as a
# soft refusal regardless of context.
SOFT_REFUSAL_TOKENS: frozenset[str] = frozenset({
    # -- Explicit non-picks (binary-style refusals) --
    "neither", "none", "both", "either", "all", "any",
    "all of them", "all of the above", "none of the above",
    "no preference", "no opinion", "no choice", "no answer",
    "no comment", "no winner", "no best", "no single", "no specific",
    "no one option", "no specific choice",

    # -- Meta / abstention --
    "neutral", "neutrality", "impartial", "impartiality",
    "abstain", "abstention", "pass", "i'll pass",
    "skip", "decline", "refuse", "n/a", "na", "not applicable",
    "not relevant", "tbd",
    "no position", "no stance",
    "i decline", "i decline to choose", "decline to choose",
    "i refuse to choose", "refuse to choose",

    # -- "Cannot decide" family --
    "i cannot", "i can't", "i won't", "i don't",
    "cannot", "cannot choose", "cannot decide", "cannot select",
    "cannot answer", "i cannot choose", "i cannot decide",
    "i cannot select", "i cannot answer", "i cannot pick",
    "unable", "i'm unable", "i am unable",

    # -- "It depends" / subjectivity --
    "it depends", "it_depends", "depends",
    "subjective", "personal preference", "personal choice",
    "your choice", "user choice", "user preference",
    "up to you", "individual choice", "individual preference",
    "varies", "various", "different things", "multiple", "many",
    "several", "mixed", "context dependent", "context-dependent",

    # -- "Unanswerable" / "no consensus" --
    "unanswerable", "no consensus", "no agreement", "no definitive answer",
    "no right answer", "no correct answer", "no objective answer",
    "no factual answer", "no single answer", "no one answer",

    # -- Uncertainty / hedge --
    "unsure", "undecided", "indecisive", "unclear", "ambiguous",
    "i don't know", "no idea", "not sure", "hard to say",
    "hard to choose", "difficult to choose", "difficult to say",

    # -- AI-identity refusals --
    "as an ai", "i am an ai", "ai", "ai assistant", "ai language model",

    # -- Template / placeholder echoes --
    "your_choice", "your choice", "{your_choice}", "choice", "option",
    "one option", "your option", "answer here", "fill in", "to be decided",
    "...", "…", "[choice]", "<choice>", "x", "y", "z", "?", "??",

    # -- Empty-ish single-letter / yes-no --
    "", " ", "n", "no", "yes",
})

# Prefix-match patterns: anything inside the braces (lowercased) starting
# with one of these is a refusal.  Catches long-form refusals like
# "{it depends on the individual situation}" or "{as an AI, I have no
# preference}" that aren't exact-match in SOFT_REFUSAL_TOKENS.
SOFT_REFUSAL_PREFIXES: tuple[str, ...] = (
    "it depends on", "it depends if", "it depends when",
    "this depends on", "depends on", "depending on",
    "as an ai", "as a neutral", "as an assistant",
    "i cannot ", "i can't ", "i won't ",
    "i'm not able", "i am not able",
    "i'm unable", "i am unable",
    "i'm sorry", "i am sorry", "sorry, i",
    "i don't have a preference",
    "no single ", "no specific ", "no best ", "no one ",
    "none of ", "neither of ",
    "neither is ", "neither has ", "neither one ",
    "both are ", "both are equally", "both have ", "all are ",
    "this is subjective", "this is a subjective",
    "this is a personal", "this is a matter of",
    "this is too broad", "this is too vague",
    "the choice depends", "the answer depends", "the best depends",
    "your choice", "your decision", "your preference",
)


def is_refusal(response: str) -> tuple[bool, str | None, str | None]:
    """Classify a single model response under the brace-format protocol.

    Returns:
        (is_refusal, parsed_answer, refusal_reason)

        is_refusal     : True iff the response is not a valid pick.
        parsed_answer  : lowercase content of the first `{...}` when the
                          response is a valid pick; None otherwise.
        refusal_reason : one of {"no_brace", "empty_brace", "soft_token",
                          "soft_prefix"} when is_refusal is True; None
                          when the response is a pick.

    Examples:
        is_refusal("{Monarch}")               -> (False, "monarch", None)
        is_refusal("As an AI, I cannot ...")  -> (True,  None,      "no_brace")
        is_refusal("{neither}")               -> (True,  None,      "soft_token")
        is_refusal("{it depends on context}") -> (True,  None,      "soft_prefix")
    """
    if not response or not response.strip():
        return True, None, "no_brace"
    m = BRACE_RE.search(response)
    if not m:
        return True, None, "no_brace"
    val = m.group(1).strip().lower()
    if not val:
        return True, None, "empty_brace"
    if val in SOFT_REFUSAL_TOKENS:
        return True, None, "soft_token"
    for pref in SOFT_REFUSAL_PREFIXES:
        if val.startswith(pref):
            return True, None, "soft_prefix"
    # Normalize number-word answers (e.g. "seven" -> "7") so that prompts
    # like "random number 0-9" cluster cleanly downstream.
    val = _NUMWORD_TO_DIGIT.get(val, val)
    return False, val, None
