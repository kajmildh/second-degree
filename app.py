#!/usr/bin/env python3
"""
Second Degree: the web app. One process serves the page and the API.

  python app.py            then open http://127.0.0.1:8000  (hosted: binds 0.0.0.0:$PORT)

Endpoints
  GET  /                   the page (static/index.html)
  GET  /api/companies      companies in the graph, for the picker
  GET  /api/captable       ?company=X  the cap table read from the graph
  POST /api/xray           the intro plan: xray_core.run(...) returned unchanged
  POST /api/match          pasted cap table text -> investor names matched to the graph
                           {"text", "use_model"}: with ANTHROPIC_API_KEY (env or .env) and
                           use_model, Claude reads the names out of text in any format
  POST /api/extract        {"filename", "data": base64}: the text of an uploaded CSV, XLSX or PDF (2 MB)
  GET  /api/config         {"model": bool}: whether the paste box can use the model
  POST /api/research       {"target": key, "sector"}: up to three cited claims from Cala's
                           knowledge search, cached in crawl_cache/research.json
  GET  /api/researched     ?sector=  target keys that already have a note
  python app.py --prefetch-research ["Davis AI"]   fill the cache for the demo cases' top three targets

The graph is loaded once at start. It is loaded again only when the crawl
rewrites crawl_cache/rounds.json (or investor_types.json), so a widening crawl
running next to the app shows up without a restart.

Set XRAY_ROUNDS to another rounds file to pin the graph.
"""
import json
import os
import re
import sys
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

import xray_core as xc

ROOT = Path(__file__).resolve().parent
ROUNDS = Path(os.environ.get("XRAY_ROUNDS") or xc.CACHE / "rounds.json")
TYPES = xc.CACHE / "investor_types.json"
SECTORS = xc.CACHE / "company_sectors.json"

_lock = threading.Lock()
_state = {"graph": None, "stamp": None}


def _stamp():
    return tuple(p.stat().st_mtime_ns if p.exists() else 0 for p in (ROUNDS, TYPES, SECTORS))


def graph():
    """The loaded graph; reloaded when the files under it have changed."""
    stamp = _stamp()
    if _state["graph"] is None or stamp != _state["stamp"]:
        with _lock:
            if _state["graph"] is None or stamp != _state["stamp"]:
                try:
                    _state["graph"] = xc.Graph(rounds_path=ROUNDS, types_path=TYPES)
                    _state["stamp"] = stamp
                except Exception:
                    if _state["graph"] is None:   # nothing to fall back on
                        raise
                    # a reload failed: keep serving the graph we have
    return _state["graph"]


@asynccontextmanager
async def lifespan(_app):
    g = graph()
    print("Graph loaded: %d rounds, %d investors, %d companies" % (
        len(g.rounds), len(g.by_investor), len(g.by_company)), flush=True)
    yield


app = FastAPI(title="Second Degree", lifespan=lifespan)


class Round(BaseModel):
    type: Literal["pre_seed", "seed", "series_a", "series_b", "bridge"] = "bridge"
    size_eur: Optional[float] = Field(default=None, ge=0)
    lead_needed: bool = False
    sector: Optional[str] = None      # a key of xray_core.SECTORS; None = the company's own tag


class XrayRequest(BaseModel):
    company: Optional[str] = None
    cap_names: Optional[List[str]] = None
    round: Round = Round()
    balance: float = Field(default=0.5, ge=0, le=1)
    include_types: List[Literal["fund", "public", "pension", "university", "corporate", "angel"]] = ["fund"]
    top: int = Field(default=25, ge=1, le=200)
    viewer: Optional[str] = None                       # the existing investor the user works for
    path_filter: Literal["all", "direct", "via"] = "all"


class MatchRequest(BaseModel):
    text: str = Field(max_length=20000)
    use_model: bool = False


# ---------------------------------------------------------------- the paste box model (optional)
MODEL = "claude-opus-5-5"
MODEL_PROMPT = """You read a startup's cap table, pasted by a user in any format: a list, a table copied \
from a spreadsheet, a press release, an email, a shareholder register.

Return the names of the institutions and people that invested in the company: venture funds, \
corporates, public funders, pension funds, universities, angels. Use each name as written in the \
text, without share counts, percentages, amounts, round names, roles or notes in brackets. \
Leave out the company itself, its founders and employees (unless the text says they invested), \
option pools, law firms, advisors and anything else that is not an investor. Each investor once, \
in the order they appear. If the text holds no investors, return an empty list."""

_model = {"client": None, "checked": False, "cache": {}, "day": None, "calls": 0}
# Hosted, the model reads at most this many cap tables a day in total; then the rules take over.
# 0 = no limit (local default).
MODEL_CALLS_PER_DAY = int(os.environ.get("MODEL_CALLS_PER_DAY") or (40 if os.environ.get("RENDER") else 0))


def _model_budget():
    """True and counts one call if today's budget allows another model call."""
    if not MODEL_CALLS_PER_DAY:
        return True
    with _lock:
        today = time.strftime("%Y-%m-%d", time.gmtime())
        if _model["day"] != today:
            _model["day"], _model["calls"] = today, 0
        if _model["calls"] >= MODEL_CALLS_PER_DAY:
            return False
        _model["calls"] += 1
        return True


def _anthropic_key():
    """ANTHROPIC_API_KEY from the environment, else from .env next to this file. Never logged."""
    key = os.environ.get("ANTHROPIC_API_KEY")
    if key:
        return key.strip()
    env = ROOT / ".env"
    try:
        for line in env.read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition("=")
            if k.strip() == "ANTHROPIC_API_KEY" and v.strip():
                return v.strip().strip('"').strip("'")
    except OSError:
        pass
    return None


def model_client():
    """An Anthropic client when a key and the SDK are present, else None (the rules-only paste box)."""
    if not _model["checked"]:
        _model["checked"] = True
        key = _anthropic_key()
        if key:
            try:
                import anthropic
                _model["client"] = anthropic.Anthropic(api_key=key, timeout=30.0, max_retries=1)
            except ImportError:
                print("ANTHROPIC_API_KEY is set but the anthropic package is not installed: paste box uses rules only", flush=True)
    return _model["client"]


class InvestorNames(BaseModel):
    investors: List[str]


def model_names(text):
    """Investor names read from text by Claude; None if the model is unavailable or the call fails."""
    client = model_client()
    if client is None:
        return None
    if text in _model["cache"]:
        return _model["cache"][text]
    if not _model_budget():
        return None
    try:
        response = client.beta.messages.parse(
            model=MODEL,
            max_tokens=4000,
            system=MODEL_PROMPT,
            messages=[{"role": "user", "content": text}],
            output_format=InvestorNames,
            output_config={"effort": "low"},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",          # a declined request is rerun on a fallback model
        )
    except Exception as e:                # any API or network failure: the rules take over
        print("paste box model call failed: %s" % type(e).__name__, flush=True)
        return None
    if response.stop_reason in ("refusal", "max_tokens") or response.parsed_output is None:
        return None
    names = []
    for n in response.parsed_output.investors:
        n = " ".join(n.split())
        if len(n) >= 2 and n not in names:
            names.append(n)
    if len(_model["cache"]) > 200:
        _model["cache"].clear()
    _model["cache"][text] = names
    return names


@app.get("/api/companies")
def companies():
    return JSONResponse(graph().companies())


@app.get("/api/captable")
def captable(company: str):
    """Investors seen in any round of the company, with their own record."""
    g = graph()
    first = g.cap_table_of(company)
    if not first:
        raise HTTPException(404, "No rounds for '%s' in the graph" % company)
    rounds = g.by_company[xc.norm(company)]
    investors = []
    for key, first_round in first.items():
        p = xc.profile(g, key)
        investors.append({"name": g.name(key), "key": key, "type": g.types.get(key, "fund"),
                          "rounds": p["rounds"], "led": p["led"], "latest_round": p["latest_round"],
                          "first_round_in_company": first_round})
    return JSONResponse({
        "company": rounds[0]["company"],
        "rounds": sorted(({"round": r["label"], "date": r["date"], "amount": r["amount"],
                           "currency": r["currency"]} for r in rounds),
                         key=lambda r: r["date"] or "", reverse=True),
        "investors": investors,
    })


@app.post("/api/xray")
def xray(req: XrayRequest):
    if not req.company and not req.cap_names:
        raise HTTPException(422, "Give a company or a list of investor names")
    result = xc.run(graph(), req.cap_names or None, req.company, req.round.model_dump(),
                    req.balance, tuple(req.include_types), req.top, req.viewer, req.path_filter)
    return JSONResponse(result)


def aliases(g):
    """Every spelling we hold for an investor -> its graph key. Built once per loaded graph."""
    if not hasattr(g, "_aliases"):
        index = {}
        for key in g.by_investor:                      # the key itself always wins
            index[key] = key
        for old, main in g.alias.items():              # merged duplicates point at the kept key
            index.setdefault(old, main)
        for key in g.by_investor:
            for spelling in [g.name(key)] + list(g.names[key]):
                index.setdefault(xc.norm(spelling), key)
        g._aliases = index
    return g._aliases


def split_names(text):
    """Pasted cap table text -> candidate investor names: one per line, comma or semicolon."""
    out = []
    for part in re.split(r"[\n\r;,\t|]+", text):
        name = re.sub(r"\([^)]*\)", " ", part)                           # (lead), (2024)
        name = re.sub(r"^\s*(?:[-*•–]+|\d+[.)])\s*", "", name)  # bullets and numbering
        name = re.sub(r"\s+\d[\d.,]*\s*(?:%|[kKmM]?\s*(?:eur|EUR|€))?\s*$", "", name)   # a trailing share or amount
        name = " ".join(name.split()).strip(" :")
        if len(name) >= 2 and not re.fullmatch(r"[\d.,% €]+", name) and name not in out:
            out.append(name)
    return out


# ---------------------------------------------------------------- research notes (Cala knowledge search)
RESEARCH = xc.CACHE / "research.json"     # {"<target key>|<sector or any>": note}; committed, so the host can show them
_research = {"lock": threading.Lock(), "busy": set(), "cala": None, "out_of_credits": False}


def research_cache():
    try:
        return json.loads(RESEARCH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def cala():
    """cala_probe when a Cala key is on this machine, else None (the hosted copy shows cached notes only)."""
    if _research["cala"] is None:
        _research["cala"] = False
        try:
            env = (ROOT / ".env").read_text(encoding="utf-8-sig")
        except OSError:
            env = ""
        if any(l.strip().startswith("CALA_API_KEY") for l in env.splitlines()):
            try:
                sys.path.insert(0, str(ROOT / "crawl"))
                import cala_probe            # forces IPv4 for Cala
                _research["cala"] = cala_probe
            except (ImportError, SystemExit):
                pass
    return _research["cala"] or None


def research_question(name, sector_label):
    what = "%s companies" % sector_label if sector_label else "companies"
    return "Has %s invested in %s since 2023? Name the companies and rounds. Cite sources." % (name, what)


META = re.compile(r"\bquery\b|\bentity record\b|INVESTED_IN|PARTICIPATES_IN|relationships?\b", re.I)


def useful(claims):
    """Drop claims that talk about Cala's own records or the question instead of the investor."""
    return [c for c in claims if not META.search(c["text"])][:3]


def claims_from(payload):
    """Up to three claims from explainability, each with the URL of the first context it cites."""
    urls = {}
    for c in payload.get("context") or []:
        for o in c.get("origins") or []:
            u = ((o.get("document") or {}).get("url")) or ((o.get("source") or {}).get("url"))
            if u and c.get("id") not in urls:
                urls[c.get("id")] = {"url": u, "site": (o.get("source") or {}).get("name") or ""}
    out = []
    for e in payload.get("explainability") or []:
        src = next((urls[r] for r in e.get("references") or [] if r in urls), None)
        text = " ".join((e.get("content") or "").split())
        if text and src:
            out.append({"text": text, "url": src["url"], "site": src["site"]})
    return useful(out)


def research(key, sector):
    """The note for one target, from the cache or one knowledge/search call. None when not available."""
    g = graph()
    sid = sector if sector in xc.SECTORS else "any"
    ckey = "%s|%s" % (key, sid)
    cached = research_cache().get(ckey)
    if cached:
        return dict(cached, claims=useful(cached["claims"]))
    cp = cala()
    if cp is None or _research["out_of_credits"] or key not in g.by_investor:
        return None
    with _research["lock"]:
        if ckey in _research["busy"]:
            return None
        _research["busy"].add(ckey)
    try:
        question = research_question(g.name(key), xc.SECTORS.get(sid))
        try:
            status, payload = cp.http("POST", "/knowledge/search", body={"input": question})
        except Exception:
            return None
        if status == 402:
            _research["out_of_credits"] = True
            print("Cala: HTTP 402 (out of credits), research stopped", flush=True)
            return None
        if status != 200 or not isinstance(payload, dict):
            return None
        note = {"target": g.name(key), "sector": sid, "question": question,
                "asked": time.strftime("%Y-%m-%d"), "claims": claims_from(payload)}
        with _research["lock"]:
            data = research_cache()
            data[ckey] = note
            tmp = RESEARCH.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
            os.replace(tmp, RESEARCH)
        return note
    finally:
        _research["busy"].discard(ckey)


class ResearchRequest(BaseModel):
    target: str                       # graph key
    sector: Optional[str] = None


@app.post("/api/research")
def research_one(req: ResearchRequest):
    note = research(req.target, req.sector)
    if note is None:
        return JSONResponse({"status": "unavailable"})
    return JSONResponse(dict(note, status="ok"))


@app.get("/api/researched")
def researched(sector: Optional[str] = None):
    """Target keys with a cached note for this sector: the page marks them."""
    sid = sector if sector in xc.SECTORS else "any"
    return JSONResponse(sorted(k.split("|", 1)[0] for k in research_cache() if k.endswith("|" + sid)))


# Demo cases: the company and the existing investor whose "through co-investors" view is shown.
DEMO = [("Davis AI", "Balderton Capital"), ("Primer", None), ("Aiven", None)]
DEMO_ROUNDS = [{"type": "series_a", "size_eur": 8e6, "lead_needed": True},
               {"type": "seed", "size_eur": 3e6, "lead_needed": True},
               {"type": "seed", "size_eur": 3e6, "lead_needed": False}]


def prefetch_research(companies=None, top=3):
    """The top three targets of each demo case, for all existing investors and for the case's
    viewer through co-investors; one call at a time (about 20 s each)."""
    g = graph()
    for company, viewer in DEMO:
        if companies and company not in companies:
            continue
        for rnd in DEMO_ROUNDS:
            for v, path in [(None, "all")] + ([(viewer, "via")] if viewer else []):
                res = xc.run(g, None, company, dict(rnd), 0.5, ("fund",), top, viewer=v, path_filter=path)
                for t in res["targets"]:
                    t0 = time.time()
                    note = research(t["key"], res["sector"]["id"])
                    print("%-10s %-8s %-8s %-28s %s (%.0f s)" % (company, rnd["type"], "via" if v else "all",
                          t["name"][:28], "%d claims" % len(note["claims"]) if note else "not available",
                          time.time() - t0), flush=True)
                    if _research["out_of_credits"]:
                        return


MAX_UPLOAD = 2 * 1024 * 1024
MAX_TEXT = 20000


class ExtractRequest(BaseModel):
    filename: str = Field(max_length=300)
    data: str = Field(max_length=(MAX_UPLOAD * 4) // 3 + 8)      # base64


def file_text(filename, raw):
    """The text of a CSV, XLSX or PDF file: one line per row or page line."""
    ext = Path(filename).suffix.lower()
    if ext in (".csv", ".txt", ".tsv"):
        for enc in ("utf-8-sig", "cp1252", "latin-1"):
            try:
                return raw.decode(enc)
            except UnicodeDecodeError:
                continue
    if ext in (".xlsx", ".xlsm"):
        import io
        import openpyxl
        wb = openpyxl.load_workbook(io.BytesIO(raw), read_only=True, data_only=True)
        lines = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                cells = [str(c).strip() for c in row if c is not None and str(c).strip()]
                if cells:
                    lines.append("\t".join(cells))
        return "\n".join(lines)
    if ext == ".pdf":
        import io
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(raw))
        return "\n".join((page.extract_text() or "") for page in reader.pages[:30])
    raise HTTPException(415, "Upload a CSV, XLSX or PDF file")


@app.post("/api/extract")
def extract(req: ExtractRequest):
    import base64
    import binascii
    try:
        raw = base64.b64decode(req.data, validate=True)
    except (binascii.Error, ValueError):
        raise HTTPException(400, "The file could not be read")
    if len(raw) > MAX_UPLOAD:
        raise HTTPException(413, "The file is larger than 2 MB")
    try:
        text = file_text(req.filename, raw)
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(422, "No text could be read from this file")
    text = "\n".join(line.rstrip() for line in text.splitlines() if line.strip())
    if not text:
        raise HTTPException(422, "No text could be read from this file")
    return JSONResponse({"text": text[:MAX_TEXT], "truncated": len(text) > MAX_TEXT})


@app.get("/api/config")
def config():
    return JSONResponse({"model": model_client() is not None})


@app.post("/api/match")
def match(req: MatchRequest):
    """Split the text into names (by the model when asked and available, else by rules),
    then match each name on its normalised spelling."""
    g = graph()
    index = aliases(g)
    names = model_names(req.text) if req.use_model and req.text.strip() else None
    reader = "model" if names is not None else "rules"
    if names is None:
        names = split_names(req.text)
    investors = []
    for name in names[:200]:
        key = index.get(xc.norm(name)) or g.match(name)
        if key:
            investors.append({"input": name, "status": "matched", "key": key,
                              "name": g.name(key), "type": g.types.get(key, "fund")})
        else:
            investors.append({"input": name, "status": "no_match", "key": None, "name": name, "type": None})
    return JSONResponse({"investors": investors, "reader": reader})


@app.get("/")
def index():
    return FileResponse(ROOT / "static" / "index.html", headers={"Cache-Control": "no-store"})


if __name__ == "__main__":
    if "--prefetch-research" in sys.argv:          # python app.py --prefetch-research ["Davis AI" ...]
        prefetch_research([a for a in sys.argv[1:] if not a.startswith("--")] or None)
        sys.exit(0)
    import uvicorn
    # Hosted (Render sets RENDER): listen on all interfaces. Locally: loopback only. HOST overrides both.
    host = os.environ.get("HOST") or ("0.0.0.0" if os.environ.get("RENDER") else "127.0.0.1")
    uvicorn.run(app, host=host, port=int(os.environ.get("PORT", "8000")), log_level="warning")
