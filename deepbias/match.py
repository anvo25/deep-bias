"""Answer matching: are two free-text answers the same answer?

This is the single primitive behind every "same top answer" comparison in
the paper: the pretrained-vs-SFT agreement curve (Fig. 4), the training-data
evidence search (Table 1), and the case studies. It is deliberately
conservative. Two answers match if any of these hold:

  1. lemma equality          'Monarchs' == 'monarch'
  2. derivational stem       'modernist' == 'modernism' == 'modern'
  3. word-set containment    'Monarch' is contained in 'Monarch butterfly'

Pure functions, no model or index required.
"""
from __future__ import annotations

import re


def _word_lemma(w: str) -> str:
    """Return the best single-word lemma using lemminflect.

    lemminflect's `getLemma` is POS-aware and handles English irregulars
    that WordNet misses (teeth -> tooth, feet -> foot, men -> man, etc.).
    We ask for both NOUN and VERB lemmas and prefer whichever actually
    reduced the surface form (the same heuristic the previous WordNet
    code used).  Falls back to the input if lemminflect returns nothing.
    """
    from lemminflect import getLemma
    n_tup = getLemma(w, upos="NOUN")
    v_tup = getLemma(w, upos="VERB")
    n = n_tup[0] if n_tup else w
    v = v_tup[0] if v_tup else w
    # lemminflect preserves case; lemma_str feeds us already-lowercased
    # tokens but defend anyway.
    n, v = n.lower(), v.lower()
    n_changed = (n != w)
    v_changed = (v != w)
    if n_changed and v_changed:
        return min(n, v, key=lambda x: (len(x), x))
    elif v_changed:
        return v
    elif n_changed:
        return n
    else:
        return n


_NUMBER_TO_WORD = {
    "0": "zero", "1": "one", "2": "two", "3": "three", "4": "four",
    "5": "five", "6": "six", "7": "seven", "8": "eight", "9": "nine",
    "10": "ten", "11": "eleven", "12": "twelve", "13": "thirteen",
    "14": "fourteen", "15": "fifteen", "16": "sixteen", "17": "seventeen",
    "18": "eighteen", "19": "nineteen", "20": "twenty",
}

# Cross-POS noun↔verb pairs that the WordNet lemmatizer can't unify
# (because lemmatize() works within a single POS). Curated; each entry
# is unambiguous (the noun's only common meaning is the action denoted
# by the verb). Words like 'act' / 'action' are intentionally excluded.
_NOUN_TO_VERB = {
    # -nse / -nce → -nd
    "defense": "defend", "offense": "offend",
    "pretense": "pretend", "expense": "expend",
    # -tion → irregular
    "decision": "decide", "incision": "incise",
    "destruction": "destroy", "construction": "construct",
    "production": "produce", "reduction": "reduce",
    "introduction": "introduce", "instruction": "instruct",
    "obstruction": "obstruct", "deduction": "deduce",
    "education": "educate", "investigation": "investigate",
    "preparation": "prepare", "examination": "examine",
    "explanation": "explain", "translation": "translate",
    "evaluation": "evaluate", "violation": "violate",
    "creation": "create", "imagination": "imagine",
    "operation": "operate", "presentation": "present",
    "transportation": "transport", "registration": "register",
    "preservation": "preserve", "concentration": "concentrate",
    "comparison": "compare", "competition": "compete",
    "completion": "complete", "selection": "select",
    "election": "elect", "rejection": "reject",
    "protection": "protect", "detection": "detect",
    "infection": "infect", "correction": "correct",
    "connection": "connect", "construction": "construct",
    "permission": "permit", "admission": "admit",
    "submission": "submit", "transmission": "transmit",
    "emission": "emit", "omission": "omit",
    "expansion": "expand", "extension": "extend",
    "intention": "intend", "ascension": "ascend",
    "discussion": "discuss",
    # -ery / -ory → -er
    "discovery": "discover", "recovery": "recover",
    "delivery": "deliver",
    # -ment → ""
    "argument": "argue", "agreement": "agree",
    "judgment": "judge", "encouragement": "encourage",
    "improvement": "improve", "achievement": "achieve",
    "involvement": "involve", "requirement": "require",
    "treatment": "treat", "enjoyment": "enjoy",
    "settlement": "settle", "development": "develop",
    "movement": "move", "payment": "pay",
    # -al → ""
    "removal": "remove", "approval": "approve",
    "denial": "deny", "trial": "try",
    "refusal": "refuse", "survival": "survive",
    "arrival": "arrive", "betrayal": "betray",
    # -ship → ""
    "censorship": "censor", "membership": "member",
    "sponsorship": "sponsor", "leadership": "lead",
    "ownership": "own", "partnership": "partner",
    # -er / -or → "" (agent → verb)
    "writer": "write", "speaker": "speak",
    "reader": "read", "teacher": "teach",
    "advisor": "advise", "investor": "invest",
    "creator": "create", "performer": "perform",
    "publisher": "publish",
}
_ORDINAL_TO_WORD = {
    "1st": "first", "2nd": "second", "3rd": "third", "4th": "fourth",
    "5th": "fifth", "6th": "sixth", "7th": "seventh", "8th": "eighth",
    "9th": "ninth", "10th": "tenth", "11th": "eleventh", "12th": "twelfth",
    "13th": "thirteenth", "14th": "fourteenth", "15th": "fifteenth",
    "16th": "sixteenth", "17th": "seventeenth", "18th": "eighteenth",
    "19th": "nineteenth", "20th": "twentieth",
}


def lemma_str(s: str) -> str:
    """Lowercase, tokenize on alphanumeric runs, lemmatize each alphabetic
    token (noun+verb; pick shorter form). Also:
      * collapse hyphens/underscores to spaces (so "self-defense" ≡ "self defense")
      * map small digits to words (so "1" ≡ "one", "10" ≡ "ten")
      * map ordinals to words (so "1st" ≡ "first")
    Returns a space-separated lemma string suitable for word-boundary regex.
    """
    if not s:
        return ""
    # Hyphen and underscore → space (handles "self-defense" vs "self defense")
    s = s.replace("-", " ").replace("_", " ")
    s = s.lower()
    # Quick ordinal swap before tokenization, so "1st" → "first"
    s = re.sub(
        r"\b(1st|2nd|3rd|4th|5th|6th|7th|8th|9th|10th|11th|12th|13th|14th|"
        r"15th|16th|17th|18th|19th|20th)\b",
        lambda m: _ORDINAL_TO_WORD[m.group(0)], s,
    )
    out = []
    for w in re.findall(r"[a-zA-Z]+|\d+", s):
        if w.isalpha():
            lemma = _word_lemma(w)
            # Cross-POS noun->verb mapping for cases lemmatization cannot
            # unify (defense <-> defend, decision <-> decide, ...).
            out.append(_NOUN_TO_VERB.get(lemma, lemma))
        elif w.isdigit():
            # Map small digits to their word form (so "7" ≡ "seven").
            # Multi-digit numbers (e.g. "100") fall through unchanged.
            out.append(_NUMBER_TO_WORD.get(w, w))
        else:
            out.append(w)
    return " ".join(out)


# ----------------------------------------------------------------------------
# Derivational stemming and the unified surface_match primitive.
#
# `lemma_str` already handles plurals (monarchs -> monarch), tense
# (ran -> run), digit/word (7 -> seven), hyphens/underscores, and a
# curated cross-POS table (defense -> defend).  It does NOT handle
# derivational morphology between siblings (modern, modernist, modernism,
# modernity all stay distinct).  We add a conservative one-suffix stripper
# and a `surface_match` function that the cluster judge and downstream
# Deep/Shallow comparison should both use as the canonical "are these the
# same answer?" check.
#
# Ordering matters: longer suffixes first so 'modernistic' -> 'modern'
# (strip 'istic'), not 'modernist' -> 'modernist' (strip 'ic' -> wrong).
# 'ative'/'ation' must precede 'ive' for the same reason (innovative ->
# 'innov' via 'ative', not via 'ive' alone, though both land close enough
# here; innovation -> 'innov' via 'ation', so the two now share a stem).
_DERIV_SUFFIXES = ("istic", "ative", "ation", "ical",
                   "ism", "ist", "ity", "ize", "ise", "ive")

# Minimum stem length after stripping; below this we keep the original
# token to avoid wiping out short roots (e.g. don't reduce 'fist' to 'f').
_MIN_STEM_LEN = 4


def _strip_derivational(tok: str) -> str:
    """Strip at most one derivational suffix from a lemma token.

    Conservative: only one suffix per token, and the resulting stem must
    be at least _MIN_STEM_LEN characters long.

    Examples:
      modernist   -> modern
      modernism   -> modern
      modernity   -> modern
      modernize   -> modern
      modernistic -> modern  (strips 'istic')
      innovation  -> innov   (strips 'ation')
      innovative  -> innov   (strips 'ative', same stem as innovation)
      history     -> history (would yield 'histor' if we stripped 'y';
                              we do not strip 'y' so it stays)
      fist        -> fist    (stripping 'ist' would leave 'f', too short)
    """
    for suf in _DERIV_SUFFIXES:
        if tok.endswith(suf) and len(tok) - len(suf) >= _MIN_STEM_LEN:
            return tok[: -len(suf)]
    return tok


def stem_str(s: str) -> str:
    """Like `lemma_str` but additionally strips one derivational suffix per
    token, so `modern`, `modernist`, `modernism`, `modernity`, `modernize`
    all reduce to the same string `modern`.
    """
    base = lemma_str(s)
    if not base:
        return ""
    return " ".join(_strip_derivational(t) for t in base.split())


def surface_match(a: str, b: str) -> bool:
    """The canonical "are these two answers the same answer?" check, used
    by both within-stage cluster merging and the cross-stage Deep/Shallow
    comparison.

    Returns True iff any of the following hold:
      1. lemma equality        ('Monarchs' == 'monarch')
      2. derivational stem eq. ('modernist' == 'modernism' == 'modern')
      3. word-set containment  ('ISO' subset of 'ISO 9001';
                                'Monarch' subset of 'Monarch butterfly')

    Returns False if either side is empty after normalization.
    """
    if not a or not b:
        return False
    la, lb = lemma_str(a), lemma_str(b)
    if la and lb and la == lb:
        return True
    sa, sb = stem_str(a), stem_str(b)
    if sa and sb and sa == sb:
        return True
    wa, wb = set(la.split()), set(lb.split())
    if wa and wb and (wa <= wb or wb <= wa):
        return True
    return False


def cluster_match(response: str, cluster: dict) -> bool:
    """Is `response` a member of `cluster` (as produced by the cluster
    judge in evaluation/cluster.py)?

    `cluster` must have the cluster judge's keys:
      - 'canonical'      (str): representative surface form
      - 'surface_forms'  (dict[str, int]): raw responses -> count
      - 'lemma_keys'     (list[str]): all canon keys merged into this cluster

    Match order (fastest checks first):
      1. exact surface-form match (response in cluster['surface_forms'])
      2. case-insensitive surface-form match
      3. surface_match(response, canonical)
      4. surface_match(response, any lemma_key)

    Returns False on empty response or malformed cluster.
    """
    if not response or not cluster:
        return False
    surface_forms = cluster.get("surface_forms") or {}
    if response in surface_forms:
        return True
    nr = response.strip().lower()
    if any(sf.strip().lower() == nr for sf in surface_forms.keys()):
        return True
    canon = cluster.get("canonical") or ""
    if canon and surface_match(response, canon):
        return True
    for lk in cluster.get("lemma_keys") or []:
        if surface_match(response, lk):
            return True
    return False

