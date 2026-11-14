#!/usr/bin/env python3
"""
Second Degree engine: from a cap table and a round to an intro plan.

Reads the crawl cache (crawl_cache/rounds.json), builds the co-investment graph
and answers one question: for THIS round, which investors outside the cap table
should be approached, and which existing investor should make each introduction?

No network calls. Pure functions over the cached Cala data, so the web app can
call run() directly and the result is identical every time.

CLI examples:
  python xray_core.py --company Photoncycle --type series_a --size 8000000 --lead
  python xray_core.py --cap "Lifeline Ventures; Maki.vc; Tesi" --type seed --size 3000000
  python xray_core.py --company "Onego Bio" --type series_b --size 25000000 --lead --json out.json
"""
import argparse
import datetime
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "crawl_cache"

LEGAL = {"oy", "oyj", "ab", "as", "aps", "ltd", "limited", "gmbh", "llc", "inc", "ky",
         "sr", "sa", "bv", "management", "vc", "s", "p", "a", "llp", "lp", "l", "ag",
         "kg", "co", "i", "ii", "iii", "iv", "fund", "fonds", "partners", "the"}

# Rough conversion to EUR, only for comparing round sizes.
FX = {"EUR": 1.0, "USD": 0.92, "GBP": 1.17, "SEK": 0.088, "NOK": 0.086, "DKK": 0.134, "CHF": 1.05}

STAGE = {  # round type chosen in the form -> round labels that count as the same stage
    "pre_seed": {"PRE_SEED", "ANGEL"},
    "seed": {"SEED", "SEED_PLUS", "PRE_SEED"},
    "series_a": {"SERIES_A", "SEED_PLUS"},
    "series_b": {"SERIES_B", "SERIES_C"},
    "bridge": set(),  # any stage counts
}

# One sector per company, from crawl_cache/company_sectors.json.
SECTORS = {
    "energy": "Energy & climate",
    "food": "Food & agriculture",
    "health": "Health & biotech",
    "computing": "Semiconductors, quantum & photonics",
    "space": "Space & defence",
    "industrial": "Industrial tech, robotics & materials",
    "software": "Software & AI",
    "fintech": "Fintech",
    "consumer": "Consumer & marketplaces",
    "mobility": "Mobility & logistics",
    "media": "Games & media",
    "other": "Other",
}

# Fallback typing for investors that investor_types.json does not cover.
TYPE_HINTS = [
    ("public", ["business finland", "innovation council", "tesi", "almi", "vinnova",
                "enterprise estonia", "smartcap", "finnvera", "eit ", "fraunhofer",
                "region ", "innovation fund", "bayern kapit", "bpifrance", "eifo",
                "investinor", "saminvest", "kfw", "european investment"]),
    ("pension", ["pension", "varma", "ilmarinen", "elo mutual"]),
    ("university", ["university", "chalmers", "kth ", "uu invest", "delft enterprises"]),
]


def norm(name):
    words = re.sub(r"[^a-z0-9åäöøæü ]", " ", (name or "").lower()).split()
    kept = [w for w in words if w not in LEGAL]
    return " ".join(kept) or " ".join(words)


def guess_type(name, entity_type):
    if entity_type == "Person":
        return "angel"
    low = " " + (name or "").lower() + " "
    for label, hints in TYPE_HINTS:
        if any(h in low for h in hints):
            return label
    return "fund"


def to_eur(amount, currency):
    if not amount:
        return None
    return amount * FX.get(currency or "EUR", 1.0)


class Graph:
    """Canonical investors and de-duplicated rounds built from the crawl cache."""

    def __init__(self, rounds_path=None, types_path=None, sectors_path=None):
        raw = json.loads(Path(rounds_path or CACHE / "rounds.json").read_text(encoding="utf-8"))
        overrides = {}
        tp = Path(types_path or CACHE / "investor_types.json")
        if tp.exists():
            overrides = json.loads(tp.read_text(encoding="utf-8"))
        self.display = {}                      # key -> clean display name, when the tagger gave one
        for k, v in list(overrides.items()):
            if isinstance(v, dict):
                if v.get("name"):
                    self.display[k] = v["name"]
                overrides[k] = v.get("type")
        # Keys that were given the same display name are one investor: fold them into one key.
        by_display = defaultdict(list)
        for k, n in self.display.items():
            by_display[n.strip().lower()].append(k)
        self.alias = {}
        for keys in by_display.values():
            if len(keys) > 1:
                main = sorted(keys, key=lambda k: (len(k), k))[0]
                for k in keys:
                    if k != main:
                        self.alias[k] = main
        self.sectors = {}                      # company key -> sector id
        sp = Path(sectors_path or CACHE / "company_sectors.json")
        if sp.exists():
            for k, v in json.loads(sp.read_text(encoding="utf-8")).items():
                sid = v.get("sector") if isinstance(v, dict) else v
                if sid in SECTORS:
                    self.sectors[k] = sid
        self.names = defaultdict(Counter)      # key -> display names seen
        self.entity_ids = defaultdict(set)     # key -> Cala entity ids merged
        self.types = {}
        merged = {}
        for rid, r in raw.items():
            props = r.get("properties") or {}
            date = props.get("announcement_date") or props.get("event_date") or props.get("close_date")
            issuer = next((p for p in r["participants"]
                           if (p.get("properties") or {}).get("role") == "ISSUER"), None)
            company = issuer["name"] if issuer else None
            label = props.get("round_label") or (r.get("name") or "").upper().replace(" ", "_")
            sig = (norm(company) if company else rid, date, label)   # same company, date, label = same round
            rec = merged.setdefault(sig, {
                "id": rid, "company": company, "company_id": issuer["id"] if issuer else None,
                "label": label, "date": date, "amount": props.get("transaction_value"),
                "currency": props.get("transaction_value_units"),
                "instrument": props.get("instrument"), "investors": {}})
            if rec["amount"] is None and props.get("transaction_value"):
                rec["amount"], rec["currency"] = props["transaction_value"], props.get("transaction_value_units")
            for p in r["participants"]:
                pp = p.get("properties") or {}
                if pp.get("role") != "INVESTOR":
                    continue
                key = self.key(p["name"])
                self.names[key][p["name"]] += 1
                self.entity_ids[key].add(p["id"])
                self.types.setdefault(key, overrides.get(key) or guess_type(p["name"], p.get("entity_type")))
                srcs = [s["document"] for s in pp.get("sources") or [] if s.get("document")]
                inv = rec["investors"].setdefault(key, {"lead": False, "sources": []})
                inv["lead"] = inv["lead"] or bool(pp.get("is_lead"))
                for s in srcs:
                    if s not in inv["sources"]:
                        inv["sources"].append(s)
        self.rounds = list(merged.values())
        for r in self.rounds:
            r["amount_eur"] = to_eur(r["amount"], r["currency"])
            r["sector"] = self.sectors.get(norm(r["company"])) if r["company"] else None
        self.by_investor = defaultdict(list)
        self.by_company = defaultdict(list)
        for r in self.rounds:
            for key in r["investors"]:
                self.by_investor[key].append(r)
            if r["company"]:
                self.by_company[norm(r["company"])].append(r)
        self.today = datetime.date.today().isoformat()

    def key(self, name):
        """The graph key for a spelling of an investor's name, after merging duplicates."""
        k = norm(name)
        return self.alias.get(k, k)

    def match(self, name):
        """The graph key for a name as a user wrote it, or None. Also tries the name without a
        fund vehicle suffix: "Lifeline Ventures Fund V Ky" is "Lifeline Ventures"."""
        k = self.key(name)
        if k in self.by_investor:
            return k
        if not hasattr(self, "_by_display"):     # the name as we show it: "Atomico"
            self._by_display = {self.name(key).lower(): key for key in self.by_investor}
        k = self._by_display.get((name or "").strip().lower())
        if k:
            return k
        base = re.sub(r"\s+(fund|fonds|rahasto)?\s*([ivx]+|\d+)?\s*$", "",
                      " ".join(w for w in re.sub(r"[^a-z0-9åäöøæü ]", " ", (name or "").lower()).split()
                               if w not in {"ky", "oy", "ab", "lp", "scsp", "gmbh", "kg", "co"}))
        k = self.key(base)
        if base and k in self.by_investor:
            return k
        low = (name or "").strip().lower()       # a short form: "Plural" for "Plural Strategy Group"
        longer = [key for shown, key in self._by_display.items() if shown.startswith(low + " ")]
        return longer[0] if len(low) >= 5 and len(longer) == 1 else None

    def sector_of(self, company):
        return self.sectors.get(norm(company)) if company else None

    def name(self, key):
        if key in self.display:
            return self.display[key]
        # Prefer the shortest common spelling: "Lifeline Ventures" over "Lifeline Ventures AB".
        options = self.names[key].most_common()
        top = options[0][1]
        return sorted([n for n, c in options if c >= top * 0.3], key=len)[0]

    def cap_table_of(self, company):
        """Investors seen in any round of a company, as (key, first round date)."""
        rounds = self.by_company.get(norm(company), [])
        first = {}
        for r in sorted(rounds, key=lambda x: x["date"] or ""):
            for key in r["investors"]:
                first.setdefault(key, r["date"])
        return first

    def companies(self):
        out = []
        for key, rounds in self.by_company.items():
            funds = {k for r in rounds for k in r["investors"] if self.types.get(k) != "angel"}
            out.append({"company": rounds[0]["company"], "rounds": len(rounds), "investors": len(funds),
                        "latest": max((r["date"] or "") for r in rounds),
                        "sector": self.sectors.get(key)})
        return sorted(out, key=lambda x: (-x["investors"], x["company"]))


def months_between(d1, d2):
    if not d1 or not d2:
        return None
    a, b = datetime.date.fromisoformat(d1[:10]), datetime.date.fromisoformat(d2[:10])
    return (b.year - a.year) * 12 + (b.month - a.month)


def profile(g, key):
    """An investor's own record, from every round we hold for them."""
    rounds = g.by_investor.get(key, [])
    sizes = sorted(r["amount_eur"] for r in rounds if r["amount_eur"])
    dates = [r["date"] for r in rounds if r["date"]]
    latest = max(dates) if dates else None
    return {
        "rounds": len(rounds),
        "led": sum(1 for r in rounds if r["investors"][key]["lead"]),
        "companies": len({norm(r["company"]) for r in rounds if r["company"]}),
        "latest_round": latest,
        "months_since_latest": months_between(latest, g.today),
        "size_min_eur": sizes[0] if sizes else None,
        "size_median_eur": sizes[len(sizes) // 2] if sizes else None,
        "size_max_eur": sizes[-1] if sizes else None,
        # The middle half of the rounds they joined (total round size, not their own ticket).
        # Only with four or more sized rounds; the full range is dominated by outliers.
        "sized_rounds": len(sizes),
        "size_typical_low_eur": sizes[len(sizes) // 4] if len(sizes) >= 4 else None,
        "size_typical_high_eur": sizes[(3 * len(sizes)) // 4] if len(sizes) >= 4 else None,
        "stages": dict(Counter(r["label"] for r in rounds if r["label"])),
        "sectors": dict(Counter(r["sector"] for r in rounds if r.get("sector")).most_common()),
    }


PRIOR_ROUNDS, PRIOR_FIT = 3, 0.35


def blend(reach, fit, balance):
    """Rank score: weighted geometric mean of reach and fit. A warm path to an investor who
    does not fit the round, or a perfect fit nobody knows, scores below a balanced target."""
    return round(max(reach, 1) ** (1 - balance) * max(fit, 1) ** balance)


def fit_score(g, key, rnd):
    """How well an investor's own record matches this round. 0 to 100, with parts."""
    rounds = g.by_investor.get(key, [])
    n = len(rounds)
    size, rtype, lead_needed = rnd.get("size_eur"), rnd.get("type", "bridge"), rnd.get("lead_needed", False)
    parts = {}
    sized = [r["amount_eur"] for r in rounds if r["amount_eur"]]
    if size and sized:
        parts["size"] = sum(1 for a in sized if size / 3 <= a <= size * 3) / len(sized)
    wanted = STAGE.get(rtype, set())
    labelled = [r["label"] for r in rounds if r["label"]]
    if wanted and labelled:
        parts["stage"] = sum(1 for l in labelled if l in wanted) / len(labelled)
    if lead_needed and n:
        parts["lead"] = sum(1 for r in rounds if r["investors"][key]["lead"]) / n
    latest = max((r["date"] for r in rounds if r["date"]), default=None)
    m = months_between(latest, g.today)
    parts["activity"] = 1.0 if m is not None and m <= 12 else 0.5 if m is not None and m <= 24 else 0.0
    # Sector: has this investor backed companies in the same sector? Any such round counts for
    # half; the rest grows with the share of their rounds, and is full at one round in two.
    sector, detail = rnd.get("sector"), {}
    tagged = [r["sector"] for r in rounds if r.get("sector")]
    if sector and tagged:
        same = sum(1 for s in tagged if s == sector)
        parts["sector"] = 0.0 if not same else 0.5 + 0.5 * min(1.0, (same / len(tagged)) / 0.5)
        detail = {"sector_rounds": same, "sector_tagged_rounds": len(tagged)}
    weights = {"size": 0.25, "stage": 0.20, "lead": 0.20, "sector": 0.25, "activity": 0.10}
    used = {k: weights[k] for k in parts}
    raw = sum(parts[k] * w for k, w in used.items()) / sum(used.values())
    # Few known rounds -> pull toward a cautious prior, so that an investor we know one round of
    # does not outrank one with a long, reasonably matching record.
    shrunk = (n * raw + PRIOR_ROUNDS * PRIOR_FIT) / (n + PRIOR_ROUNDS)
    return round(100 * shrunk), {k: round(100 * v) for k, v in parts.items()}, detail


def run(g, cap_names=None, company=None, rnd=None, balance=0.5, include_types=("fund",), top=25,
        viewer=None, path_filter="all"):
    """
    cap_names: investor names as the user typed them. company: optional; when given without
    cap_names the cap table is read from the graph, and that company's own rounds are not
    counted as warm paths. balance: 0 = rank by reach only, 1 = by fit only.

    viewer: the existing investor the user works for, or None for the whole-cap-table view.
    With a viewer, every target has a path of kind "direct" (the viewer has invested with
    them: contact them yourself) or "via" (only a co-investor has: ask for an introduction).
    Without a viewer every path is "via" the best-placed existing investor.
    path_filter: "all", "direct" or "via". With a viewer, "via" keeps only the targets the viewer
    cannot reach alone, which is the part of the list that is new to them.
    """
    rnd = dict(rnd or {"type": "bridge", "size_eur": None, "lead_needed": False})
    # Sector: the one chosen in the form, else the company's own tag, else none (no sector part).
    # "any" switches the sector part off.
    any_sector = rnd.get("sector") == "any"
    chosen = rnd.get("sector") if rnd.get("sector") in SECTORS else None
    own_tag = g.sector_of(company)
    rnd["sector"] = None if any_sector else (chosen or own_tag)
    sector_info = {"id": rnd["sector"], "label": SECTORS.get(rnd["sector"]),
                   "source": "any" if any_sector else "chosen" if chosen else "company" if own_tag else None}
    unmatched, insiders = [], []
    if cap_names:
        for n in cap_names:
            key = g.match(n)
            (insiders if key else unmatched).append(key or n)
    elif company:
        insiders = [k for k in g.cap_table_of(company) if g.types.get(k) != "angel"]
    insiders = list(dict.fromkeys(insiders))
    own_company = norm(company) if company else None
    viewer_key = (g.match(viewer) or g.key(viewer)) if viewer else None
    if viewer_key not in insiders:
        viewer_key = None

    # reach: shared companies between each insider and each outside investor
    paths = defaultdict(lambda: defaultdict(list))   # target -> insider -> [rounds]
    for ins in insiders:
        for r in g.by_investor[ins]:
            if own_company and r["company"] and norm(r["company"]) == own_company:
                continue
            for other in r["investors"]:
                if other not in insiders:
                    paths[other][ins].append(r)

    targets = []
    for key, via in paths.items():
        ttype = g.types.get(key, "fund")
        if include_types and ttype not in include_types:
            continue
        via_out, all_companies, latest_shared = [], set(), None
        for ins, rounds in via.items():
            comps = sorted({r["company"] for r in rounds if r["company"]})
            last = max((r["date"] or "") for r in rounds) or None
            all_companies |= set(comps)
            latest_shared = max(latest_shared or "", last or "") or None
            via_out.append({"insider": g.name(ins), "insider_key": ins, "shared_companies": comps,
                            "shared_rounds": len(rounds), "latest_shared": last,
                            "is_viewer": ins == viewer_key})
        via_out.sort(key=lambda v: (len(v["shared_companies"]), v["latest_shared"] or ""), reverse=True)
        c, m = len(all_companies), len(via)
        age = months_between(latest_shared, g.today)
        recency = 1.0 if age is not None and age <= 18 else 0.8 if age is not None and age <= 36 else 0.6
        reach = round(100 * (1 - math.exp(-(c + 0.5 * (m - 1)) / 2)) * recency)
        fit, fit_parts, fit_detail = fit_score(g, key, rnd)
        prof = profile(g, key)
        mine = next((v for v in via_out if v["is_viewer"]), None)
        best = mine or via_out[0]              # your own path beats a co-investor's
        kind = "direct" if mine else "via"
        evidence = []
        seen = set()
        for ins, rounds in via.items():
            for r in rounds:
                if r["id"] in seen:
                    continue
                seen.add(r["id"])
                evidence.append({
                    "company": r["company"], "round": r["label"], "date": r["date"],
                    "amount": r["amount"], "currency": r["currency"],
                    "with": [g.name(i) for i in via if i in r["investors"]],
                    "target_led": r["investors"][key]["lead"],
                    "sources": (r["investors"][key]["sources"] or
                                [s for i in via if i in r["investors"] for s in r["investors"][i]["sources"]])[:3]})
        evidence.sort(key=lambda e: e["date"] or "", reverse=True)
        ex = next((e for e in evidence if best["insider"] in e["with"]), evidence[0])
        latest_txt = " (latest: %s, %s)" % ((ex["round"] or "round").replace("_", " ").title(),
                                            (ex["date"] or "")[:7]) if ex["date"] else ""
        shared_txt = ", ".join(best["shared_companies"][:3])
        others = [v["insider"] for v in via_out if v is not best]
        if kind == "direct":
            reason = "Contact %s directly: you invested together in %s%s." % (g.name(key), shared_txt, latest_txt)
        else:
            reason = "Ask %s to introduce %s: they invested together in %s%s." % (
                best["insider"], g.name(key), shared_txt, latest_txt)
        targets.append({
            "name": g.name(key), "key": key, "type": ttype, "entity_ids": sorted(g.entity_ids[key]),
            "score": blend(reach, fit, balance),
            "reach": {"score": reach, "shared_companies": c, "insiders_connected": m,
                      "latest_shared": latest_shared, "via": via_out, "best_introducer": best["insider"]},
            "path": {"kind": kind, "introducer": None if kind == "direct" else best["insider"],
                     "shared_companies": best["shared_companies"], "also_known_by": others},
            "fit": {"score": fit, "parts": fit_parts, **fit_detail, **prof},
            "reason": reason, "evidence": evidence})
    targets.sort(key=lambda t: (-t["score"], -t["reach"]["score"], t["name"]))
    total = len(targets)
    counts = Counter(t["path"]["kind"] for t in targets)
    if path_filter in ("direct", "via"):
        targets = [t for t in targets if t["path"]["kind"] == path_filter]
    targets = targets[:top]
    for i, t in enumerate(targets, 1):
        t["rank"] = i

    asks = defaultdict(list)
    for t in targets:
        asks[t["reach"]["best_introducer"]].append(t["name"])

    return {
        "company": company, "round": rnd, "balance": balance,
        "sector": sector_info,
        "sectors": [{"id": k, "label": v} for k, v in SECTORS.items()],
        "viewer": g.name(viewer_key) if viewer_key else None,
        "path_filter": path_filter,
        "summary": {"shown": len(targets), "candidates": total,
                    "direct": counts.get("direct", 0), "via": counts.get("via", 0)},
        "cap_table": [{"name": g.name(k), "key": k, "type": g.types.get(k, "fund"),
                       "is_viewer": k == viewer_key,
                       "entities_merged": len(g.entity_ids[k]), **profile(g, k),
                       "asks": asks.get(g.name(k), [])} for k in insiders],
        "unmatched": unmatched,
        "targets": targets,
        "stats": {"candidates_considered": total, "rounds_in_graph": len(g.rounds),
                  "investors_in_graph": len(g.by_investor)},
        "footer": "Based on publicly announced funding rounds from Cala.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--company")
    ap.add_argument("--cap", help="investor names separated by ;")
    ap.add_argument("--type", default="bridge", choices=sorted(STAGE))
    ap.add_argument("--size", type=float)
    ap.add_argument("--lead", action="store_true")
    ap.add_argument("--sector", choices=sorted(SECTORS), help="default: the company's own tag")
    ap.add_argument("--balance", type=float, default=0.5)
    ap.add_argument("--types", default="fund", help="comma list: fund,public,pension,university,angel")
    ap.add_argument("--top", type=int, default=12)
    ap.add_argument("--viewer", help="the existing investor you work for")
    ap.add_argument("--path-filter", default="all", choices=["all", "direct", "via"])
    ap.add_argument("--json")
    ap.add_argument("--list-companies", action="store_true")
    a = ap.parse_args()
    g = Graph()
    if a.list_companies:
        for c in g.companies()[:40]:
            print("%-32s rounds %d  investors %2d  latest %s" % (c["company"][:32], c["rounds"], c["investors"], c["latest"]))
        return
    res = run(g, [n.strip() for n in a.cap.split(";")] if a.cap else None, a.company,
              {"type": a.type, "size_eur": a.size, "lead_needed": a.lead, "sector": a.sector}, a.balance,
              tuple(a.types.split(",")), a.top, a.viewer, a.path_filter)
    if a.json:
        Path(a.json).write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
    print("Company: %s | round: %s | considered %d candidates" % (res["company"], res["round"], res["stats"]["candidates_considered"]))
    if res["viewer"]:
        print("Viewing as %s: of %d candidates, %d direct and %d through co-investors" % (
            res["viewer"], res["summary"]["candidates"], res["summary"]["direct"], res["summary"]["via"]))
    for c in res["cap_table"]:
        print("  insider %-30s rounds %2d led %2d latest %s  asks: %d" % (
            c["name"][:30], c["rounds"], c["led"], c["latest_round"], len(c["asks"])))
    if res["unmatched"]:
        print("  not matched:", res["unmatched"])
    for t in res["targets"]:
        f = t["fit"]
        size = "%.1f-%.1fM" % (f["size_min_eur"] / 1e6, f["size_max_eur"] / 1e6) if f["size_min_eur"] else "-"
        print("%2d. %-30s score %3d | reach %3d fit %3d %s | led %d/%d | %s | %s | %s" % (
            t["rank"], t["name"][:30], t["score"], t["reach"]["score"], f["score"], f["parts"],
            f["led"], f["rounds"], size, t["type"], t["path"]["kind"]))
        print("      " + t["reason"])


if __name__ == "__main__":
    main()
