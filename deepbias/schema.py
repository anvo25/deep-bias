"""Convert one clustered row (evaluation/cluster.py output) into the released
per-family record. Used by evaluation/finalize.py for new runs and by
scripts/export_release_data.py for the released data, so both share one
definition."""
from __future__ import annotations

from deepbias.metrics import bias_type


def _dist(d: list[dict]) -> list[dict]:
    # surface_forms is a dict keyed by raw answer text in the cluster output.
    # It becomes a list so the file loads as a fixed Arrow schema.
    return [{"answer": c["canonical"], "count": c["count"], "rate": round(c["rate"], 6),
             "surface_forms": [{"text": t, "count": k} for t, k in c.get("surface_forms", {}).items()],
             "lemma_keys": c.get("lemma_keys", [])}
            for c in d]


def release_record(r: dict, prompt: str | None, model: str) -> dict:
    b, f = r["baseline"], r["framing"]
    dr, fr = float(b.get("mode_rate", 0.0)), float(f.get("framed_rate", 0.0))
    top = (b.get("mode") or "").strip()
    return {
        "id": r["entry_id"], "prompt": prompt, "model": model,
        "direct": {"top_answer": b.get("mode"), "dr": round(dr, 6), "n_samples": r.get("n_baseline", 30),
                   "n_clusters": b.get("n_distinct_clusters"), "distribution": _dist(b.get("distribution", []))},
        "framed": {"top_answer": f.get("mode"), "fr": round(fr, 6), "n_samples": r.get("n_framing", 30),
                   "n_clusters": f.get("n_distinct_clusters"), "distribution": _dist(f.get("distribution", []))},
        "pi": round(dr * fr, 6),
        # a family whose direct answers were all empty or refused has no top
        # answer to be biased toward
        "bias_type": bias_type(dr, dr * fr) if top else "non_bias",
    }


def raw_direct(x: dict) -> dict:
    return {"id": x["entry_id"], "sample_idx": x["replicate_idx"], "prompt": x["prompt"], "response": x["response"]}


def raw_framed(x: dict) -> dict:
    return {"id": x["entry_id"], "framing_idx": x["framing_idx"], "framing": x["framing"], "response": x["response"]}
