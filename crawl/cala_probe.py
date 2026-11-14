#!/usr/bin/env python3
"""
Cala API client and edge inventory probe. The crawl and the app's research
notes use its http() helper; run on its own, it lists which entity types and
relationships Cala has for a set of Nordic investors and companies.
Standard library only.

Run:   python crawl/cala_probe.py
Reads the key from .env (CALA_API_KEY=...). The key is never printed or saved.
Writes one JSON file per call to probe_results/ and a readable SUMMARY.md.
"""
import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# Some networks (for example venue Wi-Fi) advertise IPv6 but cannot route it.
# Python then waits out the full timeout on each IPv6 address before trying
# IPv4, which makes every call take about a minute. Force IPv4 unless told not to.
if os.environ.get("CALA_IPV4", "1") != "0":
    _orig_getaddrinfo = socket.getaddrinfo

    def _ipv4_only(host, port, family=0, *args, **kwargs):
        return _orig_getaddrinfo(host, port, socket.AF_INET, *args, **kwargs)

    socket.getaddrinfo = _ipv4_only

ROOT = Path(__file__).resolve().parent.parent     # the repository root
OUT = Path(os.environ.get("CALA_PROBE_OUT", ROOT / "probe_results"))
BASE = "https://api.cala.ai/v1"
TIMEOUT = 30          # per request for entity lookups
KNOWLEDGE_TIMEOUT = 120  # knowledge_query/search do real work server-side
MAX_ATTEMPTS = 5      # covers 429 backoff of 10+20+40+80s
CALL_GAP = 1.5        # seconds between calls, to stay under the rate limit

# ---- Test set: public Nordic names whose real answers are easy to check ----
COMPANIES = ["IQM Quantum Computers", "ICEYE", "Solar Foods", "Varjo"]
INVESTORS = ["Lifeline Ventures", "Maki.vc", "Voima Ventures", "Inventure",
             "Tesi", "EQT Ventures"]
ROUND_SEARCHES = [("IQM", "FundingRound"), ("IQM", "PrivateCompanyFundingRound"),
                  ("ICEYE", "FundingRound"), ("ICEYE", "PrivateCompanyFundingRound")]
PERSON_SEARCHES = ["Jan Goetz", "Rafal Modrzewski"]
PAPER_SEARCHES = ["superconducting qubit"]

QUERIES = [
    ("q01_company_rounds_nl",
     "List every funding round of IQM Quantum Computers with date, round type, "
     "amount, lead investor and all participating investors"),
    ("q02_company_rounds_ql",
     "funding_rounds.company=IQM Quantum Computers"
     ".return(name, date, series, amount, lead_investor, investors)"),
    ("q03_investor_recent_nl",
     "List all investments made by Lifeline Ventures since January 2024 with "
     "company name, round type, date and amount"),
    ("q04_investor_portfolio_ql",
     "companies.investors=Lifeline Ventures"
     ".return(name, industry, founded_year, funding, location)"),
    ("q05_coinvest_ql",
     "companies.investors=Lifeline Ventures AND Inventure"
     ".return(name, funding, location)"),
    ("q06_coinvest_nl",
     "Which companies have both Voima Ventures and Lifeline Ventures as "
     "investors? Return the company name and the funding rounds where each invested."),
    ("q07_voima_rounds_nl",
     "List all funding rounds in which Voima Ventures participated, with "
     "company, date, round type, amount and co-investors"),
    ("q08_similar_rounds_nl",
     "Series A funding rounds of deep tech companies headquartered in Finland, "
     "Sweden, Denmark or Norway since January 2025, with company, date, amount "
     "and lead investor"),
    ("q09_who_follows_nl",
     "For companies where Lifeline Ventures invested at seed stage, which "
     "investors led the following Series A round? Return company, Series A "
     "date and Series A lead investor."),
    ("q10_lead_role_nl",
     "In which funding rounds since 2024 was EQT Ventures the lead investor? "
     "Return company, date, round type and amount."),
    ("q11_people_nl",
     "Who are the founders, CEO and board members of ICEYE?"),
    ("q12_papers_nl",
     "Research papers authored by the founders of IQM Quantum Computers, with "
     "title, year and co-authors"),
    ("q13_acquisitions_nl",
     "Acquisitions of Finnish startups since 2023 with acquirer, target, date "
     "and price"),
]

SEARCHES = [
    ("s01_latest_round",
     "Who led IQM Quantum Computers' most recent funding round, when was it, "
     "and who else participated?"),
    ("s02_investor_activity",
     "Is Lifeline Ventures still making new investments? List its five most "
     "recent investments with dates."),
]

summary = []
rate_limited = []   # labels that still got 429 after all retries
failed = []         # labels with no HTTP response (timeout, network)


def load_key():
    env = ROOT / ".env"
    if not env.exists():
        sys.exit("No .env file in the repository root.")
    for line in env.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if line.startswith("CALA_API_KEY"):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    sys.exit("CALA_API_KEY not found in .env")


KEY = load_key()


def parse(raw):
    text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    try:
        return json.loads(text)
    except Exception:
        return {"_raw_text": text[:5000]}


def curl_fallback(method, url, body, timeout=TIMEOUT):
    """Used when Python's own HTTPS fails (for example a company TLS proxy)."""
    cmd = ["curl", "-sS", "-m", str(timeout), "-X", method, url,
           "-H", "X-API-KEY: " + KEY, "-H", "Content-Type: application/json",
           "-w", "\n%{http_code}"]
    if body is not None:
        cmd += ["-d", json.dumps(body)]
    p = subprocess.run(cmd, capture_output=True, timeout=timeout + 10)
    out = p.stdout.decode("utf-8", errors="replace")
    if p.returncode != 0:
        return 0, {"_error": "curl failed: " + p.stderr.decode("utf-8", errors="replace")[:500]}
    text, _, code = out.rpartition("\n")
    return int(code) if code.strip().isdigit() else 0, parse(text)


def http(method, path, params=None, body=None):
    url = BASE + path
    timeout = KNOWLEDGE_TIMEOUT if path.startswith("/knowledge") else TIMEOUT
    if params:
        url += "?" + urllib.parse.urlencode(params, doseq=True)
    data = json.dumps(body).encode("utf-8") if body is not None else None
    headers = {"X-API-KEY": KEY, "Content-Type": "application/json",
               "Accept": "application/json", "User-Agent": "cala-probe/0.1"}
    for attempt in range(MAX_ATTEMPTS):
        last = attempt == MAX_ATTEMPTS - 1
        try:
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.status, parse(r.read())
        except urllib.error.HTTPError as e:
            payload = parse(e.read())
            if e.code == 429 and not last:
                wait = retry_after(e.headers.get("Retry-After"), attempt)
                print("[429, waiting %ss]" % wait, end=" ", flush=True)
                time.sleep(wait)
                continue
            return e.code, payload
        except Exception as e:
            if is_timeout(e):
                # The server is slow, not unreachable: a curl retry would just
                # wait out another full timeout. Retry once, then give up.
                if not last and attempt == 0:
                    continue
                return 0, {"_error": "timeout after %ss" % timeout, "_python_error": repr(e)}
            # Python's HTTPS failed outright (e.g. a company TLS proxy): try curl once.
            try:
                status, payload = curl_fallback(method, url, body, timeout)
            except Exception as e2:
                status, payload = 0, {"_error": "curl fallback failed: %r" % (e2,)}
            if status == 0:
                payload["_python_error"] = repr(e)
            if status == 429 and not last:
                time.sleep(retry_after(None, attempt))
                continue
            return status, payload
    return 0, {"_error": "gave up after retries"}


def retry_after(header, attempt):
    """Seconds to wait after a 429: the server's Retry-After if given, else backoff."""
    try:
        if header is not None:
            return min(max(float(header), 1.0), 120.0)
    except ValueError:
        pass
    return min(10 * 2 ** attempt, 80)  # 10, 20, 40, 80


def is_timeout(e):
    if isinstance(e, (TimeoutError, socket.timeout)):
        return True
    reason = getattr(e, "reason", None)
    return isinstance(reason, (TimeoutError, socket.timeout))


def call(label, method, path, params=None, body=None):
    print("-> %s ..." % label, end=" ", flush=True)
    t0 = time.time()
    status, payload = http(method, path, params, body)
    elapsed = round(time.time() - t0, 1)
    record = {"label": label, "request": {"method": method, "path": path,
              "params": params, "body": body}, "status": status,
              "elapsed_s": elapsed, "response": payload}
    (OUT / (label + ".json")).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    print("HTTP %s in %ss" % (status, elapsed), flush=True)
    if status == 429:
        rate_limited.append(label)
    elif status == 0:
        failed.append(label)
    time.sleep(CALL_GAP)
    return status, payload, elapsed


def note(line=""):
    summary.append(line)


def slug(s):
    return "".join(c.lower() if c.isalnum() else "_" for c in s).strip("_")


def entity_search(label, name, etype=None, limit=5):
    params = {"name": name, "limit": limit}
    if etype:
        params["entity_types"] = etype
    status, payload, _ = call(label, "GET", "/entities", params=params)
    ents = payload.get("entities") or [] if isinstance(payload, dict) else []
    flag = "  **RATE-LIMITED, no data**" if status == 429 else (
        "  **NO RESPONSE**" if status == 0 else "")
    note("### Entity search: `%s`%s  (HTTP %s, %d hits)%s" % (
        name, " type=" + etype if etype else "", status, len(ents), flag))
    for e in ents[:5]:
        desc = (e.get("description") or "")[:110]
        note("- %s | %s | %s | %s" % (e.get("name"), e.get("entity_type"), e.get("id"), desc))
    note()
    return ents


def inspect(label, ent):
    """Introspection (properties and relationship types) plus the full profile."""
    eid = ent.get("id")
    if not eid:
        return
    status, payload, _ = call(label + "_introspection", "GET",
                              "/entities/%s/introspection" % eid)
    if isinstance(payload, dict) and status == 200:
        rel = payload.get("relationships") or {}
        obs = payload.get("numerical_observations") or {}
        note("### Introspection: %s (%s)" % (ent.get("name"), ent.get("entity_type")))
        note("- properties: %s" % ", ".join(payload.get("properties") or []))
        note("- outgoing: %s" % ", ".join(rel.get("outgoing") or []))
        note("- incoming: %s" % ", ".join(rel.get("incoming") or []))
        note("- numerical observation groups: %s" % ", ".join(
            "%s (%d)" % (k, len(v or [])) for k, v in obs.items()))
        note()
    else:
        note("### Introspection FAILED for %s: HTTP %s" % (ent.get("name"), status))
        note()
    # Retrieve Entity is POST (GET returns 405). Without a body only the default
    # properties come back, so request everything introspection says exists.
    body = None
    if isinstance(payload, dict) and status == 200:
        rel = payload.get("relationships") or {}
        body = {
            "properties": payload.get("properties") or [],
            "relationships": {
                d: {t: {"limit": 10} for t in (rel.get(d) or [])}
                for d in ("outgoing", "incoming") if rel.get(d)
            },
        }
        if not body["relationships"]:
            del body["relationships"]
    call(label + "_entity", "POST", "/entities/%s" % eid, body=body)


def run_query(label, text, endpoint="query"):
    status, payload, elapsed = call(label, "POST", "/knowledge/" + endpoint,
                                    body={"input": text})
    note("### %s  (HTTP %s, %ss)" % (label, status, elapsed))
    note("Input: `%s`" % text)
    if not isinstance(payload, dict):
        note("- unexpected response shape")
        note()
        return payload
    if status != 200:
        note("- error: %s" % json.dumps(payload, ensure_ascii=False)[:400])
        note()
        return payload
    results = payload.get("results")
    if isinstance(results, list):
        note("- rows: %d" % len(results))
        if results and isinstance(results[0], dict):
            note("- columns: %s" % ", ".join(results[0].keys()))
        for row in results[:5]:
            note("  - %s" % json.dumps(row, ensure_ascii=False)[:400])
    ents = payload.get("entities") or []
    if ents:
        counts = {}
        for e in ents:
            counts[e.get("entity_type")] = counts.get(e.get("entity_type"), 0) + 1
        note("- entities returned: %s" % ", ".join(
            "%s x%d" % (k, v) for k, v in sorted(counts.items(), key=lambda x: -x[1])))
    other = [k for k in payload.keys() if k not in ("results", "entities")]
    if other:
        note("- other top-level keys: %s" % ", ".join(other))
        for k in other:
            v = payload[k]
            text_v = v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)
            note("  - %s: %s" % (k, text_v[:600].replace("\n", " ")))
    note()
    return payload


def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    OUT.mkdir(parents=True, exist_ok=True)
    started = time.strftime("%Y-%m-%d %H:%M:%S")
    note("# Cala probe summary")
    note("Run started %s" % started)
    note()

    note("## 1. Entities: companies and investors")
    note()
    for name in COMPANIES + INVESTORS:
        ents = entity_search("e_" + slug(name), name)
        if ents:
            inspect("e_" + slug(name), ents[0])

    note("## 2. Entities: funding rounds")
    note()
    for name, etype in ROUND_SEARCHES:
        label = "r_%s_%s" % (slug(name), slug(etype))
        ents = entity_search(label, name, etype)
        if ents:
            inspect(label, ents[0])

    note("## 3. Entities: people and papers")
    note()
    for name in PERSON_SEARCHES:
        ents = entity_search("p_" + slug(name), name, "Person")
        if ents:
            inspect("p_" + slug(name), ents[0])
    for name in PAPER_SEARCHES:
        ents = entity_search("paper_" + slug(name), name, "ResearchPaper")
        if ents:
            inspect("paper_" + slug(name), ents[0])

    note("## 4. Structured queries (knowledge_query)")
    note()
    for label, text in QUERIES:
        run_query(label, text, "query")

    note("## 5. Sourced answers (knowledge_search)")
    note()
    for label, text in SEARCHES:
        run_query(label, text, "search")

    note("## 6. Calls that did not get an answer")
    note()
    note("These are NOT evidence that Cala lacks the data; rerun them.")
    note("- rate-limited (429 after retries): %s" % (", ".join(rate_limited) or "none"))
    note("- no response (timeout/network): %s" % (", ".join(failed) or "none"))
    note()
    (OUT / "SUMMARY.md").write_text("\n".join(summary), encoding="utf-8")
    print("\nDone. Results are in: %s" % OUT)
    print("Open probe_results/SUMMARY.md for the readable version.")


if __name__ == "__main__":
    main()
