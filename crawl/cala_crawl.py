#!/usr/bin/env python3
"""
Cala graph crawl: builds a local, sourced co-investment dataset from Cala's
structured graph (fast entity calls only, no knowledge_query).

For each seed investor:
  1. entity search, keep the entities whose name matches the seed,
  2. retrieve each with all INVESTED_IN and PARTICIPATES_IN_FUNDING_ROUND edges,
     and keep only those that really have investor edges (duplicates get merged),
  3. retrieve every funding round: date, amount, instrument, issuer and all
     participants with their is_lead flag and source links.

Everything is saved under crawl_cache/ as raw JSON, one file per call. A rerun
skips what is already there, so it can be stopped and resumed at any time.

Run:  python crawl/cala_crawl.py            (the 24 built-in seeds, capped at MAX_CALLS)
      python crawl/cala_crawl.py --seeds-file crawl_cache/next_seeds.json
Then: crawl_cache/CRAWL_SUMMARY.md  (CRAWL_SUMMARY_<file>.md for a seeds file)

rounds.json is always the union of every round in the cache, whichever run
fetched it, and is rewritten after each seed so a stopped run loses nothing.
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import cala_probe as cp

ROOT = Path(__file__).resolve().parent.parent     # the repository root
CACHE = ROOT / "crawl_cache"
MAX_CALLS = 4000      # hard cap on new API calls per run (protects credits)
GAP = 1.0             # seconds between new calls
ROUND_PROPS = ["name", "round_label", "announcement_date", "event_date", "close_date",
               "transaction_value", "transaction_value_units", "instrument", "aliases"]

SEEDS = [
    "Lifeline Ventures", "Maki.vc", "Inventure", "Voima Ventures", "Icebreaker.vc",
    "Superhero Capital", "Butterfly Ventures", "OpenOcean", "Tesi", "NordicNinja",
    "Ugly Duckling Ventures", "LUMO Labs", "Eviny Ventures", "Luminar Ventures",
    "Momentum Partners", "byFounders", "EQT Ventures", "Vsquared Ventures",
    "World Fund", "Nordic Science Investments", "Eir Ventures", "Almi Invest",
    "Hoxton Ventures", "European Innovation Council Fund",
]

LEGAL = {"oy", "oyj", "ab", "as", "aps", "ltd", "limited", "gmbh", "llc", "inc",
         "ky", "sr", "sa", "bv", "management", "vc"}
calls = {"new": 0, "cached": 0, "errors": 0}
log = []


def norm(name):
    words = re.sub(r"[^a-z0-9åäöøæü ]", " ", (name or "").lower()).split()
    return " ".join(w for w in words if w not in LEGAL)


def slug(s):
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def fetch(fname, method, path, params=None, body=None):
    """Cached call. Returns the response payload, or None on failure or cap."""
    global MAX_CALLS
    f = CACHE / fname
    if f.exists():
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
            if rec.get("status") == 200:
                calls["cached"] += 1
                return rec["response"]
        except Exception:
            pass
    if calls["new"] >= MAX_CALLS:
        return None
    calls["new"] += 1
    t0 = time.time()
    try:
        status, payload = cp.http(method, path, params, body)
    except Exception as e:
        status, payload = 0, {"_error": repr(e)}
    rec = {"request": {"method": method, "path": path, "params": params, "body": body},
           "status": status, "elapsed_s": round(time.time() - t0, 1), "response": payload}
    f.write_text(json.dumps(rec, ensure_ascii=False), encoding="utf-8")
    print("[%d] %s -> HTTP %s (%ss)" % (calls["new"], fname, status, rec["elapsed_s"]), flush=True)
    time.sleep(GAP)
    if status != 200:
        calls["errors"] += 1
        log.append("FAILED %s: HTTP %s %s" % (fname, status, json.dumps(payload)[:200]))
        if status == 402:          # out of credits: stop the run here (no further calls)
            MAX_CALLS = calls["new"]
            log.append("STOPPED at HTTP 402 (out of credits) after %d new calls" % calls["new"])
            print("HTTP 402: out of credits, stopping", flush=True)
        return None
    return payload


def edges(payload, direction, rel):
    return ((payload or {}).get("relationships") or {}).get(direction, {}).get(rel) or []


def investor_entity(eid):
    """All investment and round edges of one entity, paging in blocks of 200."""
    merged = None
    offset = 0
    while True:
        body = {"relationships": {"outgoing": {
            "INVESTED_IN": {"limit": 200, "offset": offset},
            "PARTICIPATES_IN_FUNDING_ROUND": {"limit": 200, "offset": offset}}, "incoming": {}}}
        p = fetch("investor_%s_o%d.json" % (eid, offset), "POST", "/entities/%s" % eid, body=body)
        if not p:
            break
        inv, rnd = edges(p, "outgoing", "INVESTED_IN"), edges(p, "outgoing", "PARTICIPATES_IN_FUNDING_ROUND")
        if merged is None:
            merged = {"id": p.get("id"), "name": p.get("name"), "entity_type": p.get("entity_type"),
                      "description": p.get("description"), "invested_in": [], "rounds": []}
        merged["invested_in"] += inv
        merged["rounds"] += rnd
        if len(inv) < 200 and len(rnd) < 200:
            break
        offset += 200
    return merged


def round_entity(rid):
    """Round properties plus every participant."""
    # Retrieve Entity is POST only (GET returns 405). A body that names no
    # properties returns none, so list the round fields explicitly.
    body = {"properties": ROUND_PROPS,
            "relationships": {"outgoing": {}, "incoming": {
                "PARTICIPATES_IN_FUNDING_ROUND": {"limit": 200, "offset": 0}}}}
    p = fetch("roundp_%s.json" % rid, "POST", "/entities/%s" % rid, body=body)
    if not p:
        return None
    return parse_round(rid, p)


def parse_round(rid, p):
    parts = edges(p, "incoming", "PARTICIPATES_IN_FUNDING_ROUND")
    props = {k: (v.get("value") if isinstance(v, dict) else v)
             for k, v in (p.get("properties") or {}).items()}
    return {"id": rid, "name": p.get("name"), "properties": props, "participants": parts}


def all_cached_rounds():
    """Every round already in the cache, whichever run fetched it."""
    out = {}
    for f in sorted(CACHE.glob("roundp_*.json")):
        try:
            rec = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if rec.get("status") == 200 and rec.get("response"):
            rid = f.stem[len("roundp_"):]
            out[rid] = parse_round(rid, rec["response"])
    return out


def write_json(path, data):
    """Write through a temp file so a reader never sees half a file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:      # Windows: someone is reading it right now
            time.sleep(0.2)
    os.replace(tmp, path)


def main():
    global MAX_CALLS
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds-file", help="JSON list of investor names to crawl instead of the built-in seeds")
    ap.add_argument("--max-calls", type=int, default=MAX_CALLS, help="cap on new API calls in this run")
    a = ap.parse_args()
    MAX_CALLS = a.max_calls
    seeds, summary_name = SEEDS, "CRAWL_SUMMARY.md"
    if a.seeds_file:
        seeds = json.loads(Path(a.seeds_file).read_text(encoding="utf-8-sig"))
        if not (isinstance(seeds, list) and all(isinstance(x, str) for x in seeds)):
            sys.exit("--seeds-file must hold a JSON list of names")
        seeds = list(dict.fromkeys(x.strip() for x in seeds if x.strip()))
        summary_name = "CRAWL_SUMMARY_%s.md" % slug(Path(a.seeds_file).stem)
    CACHE.mkdir(exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    rounds = all_cached_rounds()          # union: start from everything already fetched
    known_before = len(rounds)
    resolution, out = {}, ["# Cala graph crawl summary", "Started %s, %d seeds" % (started, len(seeds)), ""]
    print("Seeds: %d. Rounds already in cache: %d. Call cap: %d" % (len(seeds), known_before, MAX_CALLS), flush=True)

    for n_seed, seed in enumerate(seeds, 1):
        res = fetch("search_%s.json" % slug(seed), "GET", "/entities",
                    params={"name": seed, "limit": 10})
        hits = (res or {}).get("entities") or []
        key = norm(seed)
        kept = []
        for h in hits:
            n = norm(h.get("name"))
            if h.get("entity_type") not in ("Organization", "Company"):
                continue
            if not (n == key or n.startswith(key + " ")):
                continue
            ent = investor_entity(h["id"])
            if ent and (ent["invested_in"] or ent["rounds"]):
                kept.append(ent)
        resolution[seed] = kept
        round_ids = {r["id"] for e in kept for r in e["rounds"]
                     if (r.get("properties") or {}).get("role") == "INVESTOR"}
        for rid in sorted(round_ids):
            if rid not in rounds:
                r = round_entity(rid)
                if r:
                    rounds[rid] = r
        companies = {c.get("name") for e in kept for c in e["invested_in"]}
        leads = sum(1 for e in kept for r in e["rounds"] if (r.get("properties") or {}).get("is_lead"))
        out.append("## %s" % seed)
        if not kept:
            out.append("- NOT RESOLVED. Search hits: %s" % "; ".join(
                "%s (%s)" % (h.get("name"), h.get("entity_type")) for h in hits[:6]))
        for e in kept:
            out.append("- entity: %s | %s | %s | invested_in %d, rounds %d" % (
                e["name"], e["entity_type"], e["id"], len(e["invested_in"]), len(e["rounds"])))
        out.append("- merged: %d companies, %d rounds, lead on %d" % (len(companies), len(round_ids), leads))
        # co-investors seen in this seed's rounds
        own = {e["id"] for e in kept}
        co = Counter()
        names = {}
        for rid in round_ids:
            for p in (rounds.get(rid) or {}).get("participants", []):
                if (p.get("properties") or {}).get("role") == "INVESTOR" and p.get("id") not in own:
                    co[p["id"]] += 1
                    names[p["id"]] = p.get("name")
        out.append("- top co-investors: %s" % ", ".join(
            "%s (%d)" % (names[i], c) for i, c in co.most_common(12)))
        out.append("")
        write_json(CACHE / "rounds.json", rounds)
        print("== %d/%d %s: %d entities, %d rounds | graph now %d rounds | new calls %d" % (
            n_seed, len(seeds), seed, len(kept), len(round_ids), len(rounds), calls["new"]), flush=True)
        if calls["new"] >= MAX_CALLS:
            out.append("!! Stopped at the call cap (%d). Rerun to continue." % MAX_CALLS)
            break

    dated = sum(1 for r in rounds.values() if any(
        r["properties"].get(k) for k in ("announcement_date", "event_date", "close_date")))
    out.insert(2, "Rounds in cache: %d (%d new in this run; with a date: %d). New calls: %d, from cache: %d, errors: %d."
               % (len(rounds), len(rounds) - known_before, dated, calls["new"], calls["cached"], calls["errors"]))
    out += ["## Problems"] + (log or ["none"])
    (CACHE / summary_name).write_text("\n".join(out), encoding="utf-8")
    write_json(CACHE / "rounds.json", rounds)
    # investors.json keeps every seed ever resolved, not only this run's.
    # A seed that resolved before is not replaced by an empty result from a failed run.
    investors = {}
    try:
        investors = json.loads((CACHE / "investors.json").read_text(encoding="utf-8"))
    except Exception:
        pass
    for seed, kept in resolution.items():
        if kept or seed not in investors:
            investors[seed] = kept
    write_json(CACHE / "investors.json", investors)
    print("\nDone. Rounds in cache: %d. New calls: %d, cached: %d, errors: %d" % (
        len(rounds), calls["new"], calls["cached"], calls["errors"]))
    print("Open crawl_cache/%s" % summary_name)


if __name__ == "__main__":
    main()
