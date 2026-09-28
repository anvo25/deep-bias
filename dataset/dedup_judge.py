"""Shared LLM-judge helper for step5's near-duplicate band.

Embedding-only dedup (step5_deduplicate.py) is precise above 0.95 cosine and
almost never wrong below 0.80, but the 0.80-0.95 band is genuinely mixed:
real duplicates ("choose a random binary array" / "...binary string") sit
right next to legitimately different questions ("choose a random poet" /
"...poem") at similar cosine values. This module asks an LLM to decide each
borderline pair instead of a fixed cosine cutoff.
Used by step5_deduplicate.py.

Results are cached to disk keyed by the (sorted) text pair, so re-running
the same comparison -- across scripts, or across a resumed run -- costs
nothing.
"""
import json
import os
import threading
import time

OPENAI_URL = "https://api.openai.com/v1/chat/completions"
JUDGE_MODEL = "gpt-5.6-luna"
JUDGE_REASONING_EFFORT = "low"

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

JUDGE_SYSTEM_PROMPT = (
    "You compare two short \"Choose a random <topic>\" prompts used to probe "
    "an LLM for answer bias -- each prompt is later shown to a model alone, "
    "which must pick one instance of the topic. Decide whether the two "
    "prompts are effectively duplicates: would a real person choosing to "
    "write one instead of the other be asking a genuinely different "
    "question, with a meaningfully different set of plausible answers? "
    "Judge by the topic's answer space, not by surface wording overlap. "
    "Two prompts that share most words but scope a different, non-overlapping "
    "space of correct answers (different number ranges, disjoint categories, "
    "a general term vs. one narrow instance of it that excludes most of the "
    "general term's other instances) are NOT duplicates. This includes a "
    "generic attribute question ('choose a random color') versus the same "
    "attribute tied to one specific real-world object ('choose a random "
    "ball color', '...sock color', '...ship color') -- naming the object "
    "anchors the answer to that object's own typical real-world values "
    "(balls skew red/orange, socks skew white/black/gray, ships skew "
    "gray/navy/white), which is a genuinely different, differently-biased "
    "answer distribution than the unrestricted generic attribute, and "
    "different named objects are different from EACH OTHER by the same "
    "logic, not just from the generic version. Two prompts that use "
    "different words for what is functionally the same open choice, with "
    "no such object- or range-anchoring difference, ARE duplicates.\n\n"
    "Respond with ONLY a JSON object: "
    '{"duplicate": true or false, "reason": "short string"}.'
)
JUDGE_USER_TEMPLATE = "A: {a}\nB: {b}"


def _load_api_key(ckpt_dir: str = None) -> str:
    candidates = []
    if ckpt_dir:
        candidates.append(os.path.join(ckpt_dir, "api_key.txt"))
    for key_file in candidates:
        if os.path.exists(key_file):
            with open(key_file, "r", encoding="utf-8") as f:
                key = f.read().strip()
            if key:
                return key
    return os.environ.get("OPENAI_API_KEY")


def _pair_key(a: str, b: str) -> str:
    lo, hi = sorted((a, b))
    return f"{lo}␟{hi}"  # unit-separator, won't collide with real text


class DedupJudgeCache:
    """Disk-backed cache of judge decisions, keyed by the unordered text pair.

    Thread-safe; flushes after every new decision so a killed run loses at
    most the one in-flight call.
    """

    def __init__(self, cache_path: str, api_key: str = None, ckpt_dir: str = None,
                 model: str = JUDGE_MODEL, timeout: int = 60, max_retries: int = 4):
        self.cache_path = cache_path
        self.api_key = api_key or _load_api_key(ckpt_dir)
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        self._lock = threading.Lock()
        self._cache: dict = {}
        self.n_calls = 0
        self.n_cache_hits = 0
        self.n_errors = 0
        if os.path.exists(cache_path):
            with open(cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)

    def _save(self):
        tmp = self.cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._cache, f, ensure_ascii=False, indent=1)
        os.replace(tmp, self.cache_path)

    def judge(self, a: str, b: str) -> bool:
        """Returns True iff the LLM judges (a, b) to be duplicates.
        On unrecoverable API failure, falls back to False (i.e. keep both --
        the safe default, matching step5's existing bias toward
        under-merging over wrongly collapsing distinct questions)."""
        key = _pair_key(a, b)
        with self._lock:
            if key in self._cache:
                self.n_cache_hits += 1
                return self._cache[key]["duplicate"]

        result = self._call(a, b)
        with self._lock:
            self._cache[key] = result
            self._save()
        return result["duplicate"]

    def _call(self, a: str, b: str) -> dict:
        import requests
        if not self.api_key:
            self.n_errors += 1
            return {"duplicate": False, "reason": "NO_API_KEY -- skipped judge, kept both"}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
                {"role": "user", "content": JUDGE_USER_TEMPLATE.format(a=a, b=b)},
            ],
            "max_completion_tokens": 200,
            "response_format": {"type": "json_object"},
            "reasoning_effort": JUDGE_REASONING_EFFORT,
        }
        last_err = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(OPENAI_URL, headers=headers, json=payload, timeout=self.timeout)
                self.n_calls += 1
                if resp.status_code == 429 or resp.status_code >= 500:
                    last_err = f"HTTP {resp.status_code}: {resp.text[:300]}"
                    time.sleep(min(2 ** attempt, 30))
                    continue
                if resp.status_code == 400:
                    last_err = f"HTTP 400: {resp.text[:300]}"
                    break
                resp.raise_for_status()
                data = resp.json()
                raw = data["choices"][0]["message"]["content"]
                obj = json.loads(raw)
                dup = bool(obj.get("duplicate"))
                reason = str(obj.get("reason") or "")
                return {"duplicate": dup, "reason": reason}
            except Exception as e:
                last_err = repr(e)
                time.sleep(min(2 ** attempt, 30))
        self.n_errors += 1
        return {"duplicate": False, "reason": f"ERROR after retries ({last_err}) -- kept both"}
