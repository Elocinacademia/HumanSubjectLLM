"""A short README: pip2_fullpaper_judge.py

LLM-as-a-judge using OpenAI GPT-4o API.
For each paper in papers_all_keyword_filtered.csv:
  1. Find the local PDF in pdfs/
  2. Extract full text
  3. Ask GPT-4o to judge inclusion based on criteria
  4. Save result to save/judged_fullpaper.csv + save/judged_fullpaper_log.jsonl

Progress is tracked in the JSONL log — already-judged papers are skipped.
"""

import hashlib
import io
import json
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, Optional

from openai import OpenAI
import pandas as pd
from pypdf import PdfReader

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

INPUT_CSV  = Path("save") / "papers_all_keyword_filtered.csv"
OUTPUT_CSV = Path("save") / "judged_fullpaper.csv"
LOG_JSONL  = Path("save") / "judged_fullpaper_log.jsonl"
PDF_DIR    = Path("pdfs")

MODEL           = "gpt-4o"
MAX_TOKENS      = 1024
SLEEP_BETWEEN   = 3   # seconds between API calls
MAX_RETRIES     = 3

# ─────────────────────────────────────────────
# Prompt
# ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a research paper screener. You are given the FULL TEXT of a paper.

INCLUDE (Judge_include = YES) if:
  Human participants directly interact with an LLM system during the study
  (e.g. chatting, prompting, iterating, receiving model responses in-session,
  using an LLM-powered interface in a live interaction loop).

EXCLUDE (Judge_include = NO) if:
  - Humans only annotate / rate / rank / review static LLM outputs without direct interaction
  - No human user study involving LLM interaction
  - Paper discusses LLMs theoretically or evaluates them without a user study

UNSURE (Judge_include = UNSURE) if:
  The methodology is present but genuinely ambiguous about direct human–LLM interaction.

Output ONLY valid JSON — nothing else:
{
  "Judge_include": "YES" | "NO" | "UNSURE",
  "Judge_confidence": 0.0,
  "Judge_reason": "2-3 sentences citing specific evidence from the text"
}"""

USER_TEMPLATE = """Title:  {title}
Venue:  {venue}, {year}
URL:    {url}

FULL TEXT:
\"\"\"
{full_text}
\"\"\""""


# ─────────────────────────────────────────────
# Filename helpers (must match download_all_pdfs.py)
# ─────────────────────────────────────────────

def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()

def _safe(s: str, max_len: int = 160) -> str:
    s = _norm(s)
    s = re.sub(r"[^\w\-\.\(\)\[\] ]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "_")
    return (s[:max_len] if len(s) > max_len else s) or "paper"

def _sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:10]

def pdf_path(venue: str, year: str, title: str, url: str) -> Path:
    fname = f"{_safe(venue, 50)}_{_safe(year)}_{_sha1(url)}_{_safe(title, 120)}.pdf"
    return PDF_DIR / fname


# ─────────────────────────────────────────────
# PDF text extraction
# ─────────────────────────────────────────────

def extract_text(path: Path) -> str:
    reader = PdfReader(io.BytesIO(path.read_bytes()))
    pages = []
    for i, page in enumerate(reader.pages, 1):
        try:
            t = (page.extract_text() or "").replace("\x00", " ").strip()
        except Exception:
            t = ""
        if t:
            pages.append(f"\n[PAGE {i}]\n{t}")
    return "\n".join(pages).strip()


# ─────────────────────────────────────────────
# JSON parsing
# ─────────────────────────────────────────────

def parse_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        text = "\n".join(lines[1:-1]).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        s, e = text.rfind("{"), text.rfind("}")
        if s != -1 and e > s:
            return json.loads(text[s:e+1])
    raise ValueError(f"Cannot parse JSON: {text[:200]}")


# ─────────────────────────────────────────────
# judge
# ─────────────────────────────────────────────

def judge(client: OpenAI, full_text: str, title: str,
          venue: str, year: str, url: str) -> Dict[str, Any]:
    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": USER_TEMPLATE.format(
                title=title, venue=venue, year=year, url=url,
                full_text=full_text[:300_000],
            )},
        ],
        response_format={"type": "json_object"},
    )
    text = response.choices[0].message.content or ""
    return parse_json(text)


# ─────────────────────────────────────────────
# Log helpers
# ─────────────────────────────────────────────

def load_log(path: Path) -> Dict[str, Dict]:
    done = {}
    if not path.exists():
        return done
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("URL"):
                    done[rec["URL"]] = rec
            except json.JSONDecodeError:
                pass
    return done

def append_log(path: Path, record: Dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError("OPENAI_API_KEY is not set.")
    client = OpenAI(api_key=api_key)

    df = pd.read_csv(INPUT_CSV)
    for col in ["Venue", "Year", "Title", "Abstract", "URL"]:
        df[col] = df[col].fillna("").astype(str).str.strip()
    df = df[df["URL"] != ""].reset_index(drop=True)
    total = len(df)

    done = load_log(LOG_JSONL)
    print(f"Total papers: {total} | Already done: {len(done)}")

    ok = skipped = no_pdf = errors = 0

    for i, row in df.iterrows():
        url   = row["URL"]
        title = row["Title"]
        venue = row["Venue"]
        year  = row["Year"]

        if url in done and done[url].get("status") == "ok":
            skipped += 1
            continue

        print(f"\n[{i+1}/{total}] {title[:70]}")

        p = pdf_path(venue, year, title, url)
        if not p.exists() or p.stat().st_size < 10_000:
            print(f"  NO PDF: {p.name}")
            rec = {"URL": url, "Title": title, "Venue": venue, "Year": year,
                   "status": "no_pdf", "Judge_include": "", "Judge_confidence": 0.0,
                   "Judge_reason": "PDF not found locally"}
            append_log(LOG_JSONL, rec)
            no_pdf += 1
            continue

        # Extract text
        try:
            full_text = extract_text(p)
        except Exception as e:
            print(f"  TEXT EXTRACT FAILED: {e}")
            rec = {"URL": url, "Title": title, "Venue": venue, "Year": year,
                   "status": "error", "Judge_include": "", "Judge_confidence": 0.0,
                   "Judge_reason": f"Text extraction failed: {e}"}
            append_log(LOG_JSONL, rec)
            errors += 1
            continue

        if not full_text.strip():
            print("  EMPTY TEXT after extraction")
            rec = {"URL": url, "Title": title, "Venue": venue, "Year": year,
                   "status": "no_pdf", "Judge_include": "", "Judge_confidence": 0.0,
                   "Judge_reason": "PDF text could not be extracted"}
            append_log(LOG_JSONL, rec)
            no_pdf += 1
            continue

        last_err = None
        result = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                result = judge(client, full_text, title, venue, year, url)
                break
            except Exception as e:
                last_err = e
                print(f"  Attempt {attempt}/{MAX_RETRIES} failed: {e}")
                if attempt < MAX_RETRIES:
                    time.sleep(SLEEP_BETWEEN)

        if result is None:
            rec = {"URL": url, "Title": title, "Venue": venue, "Year": year,
                   "status": "error", "Judge_include": "", "Judge_confidence": 0.0,
                   "Judge_reason": f"Claude error: {last_err}"}
            append_log(LOG_JSONL, rec)
            errors += 1
            continue

        print(f"  → {result.get('Judge_include')} (conf={result.get('Judge_confidence', 0):.2f})")
        print(f"     {result.get('Judge_reason', '')[:120]}")

        rec = {"URL": url, "Title": title, "Venue": venue, "Year": year,
               "Abstract": row["Abstract"], "status": "ok",
               "Judge_include":    result.get("Judge_include", "UNSURE"),
               "Judge_confidence": result.get("Judge_confidence", 0.0),
               "Judge_reason":     result.get("Judge_reason", "")}
        append_log(LOG_JSONL, rec)
        ok += 1

        time.sleep(SLEEP_BETWEEN)

    # Write final CSV from log
    all_done = load_log(LOG_JSONL)
    rows = []
    for _, row in df.iterrows():
        rec = all_done.get(row["URL"])
        if rec and rec.get("status") == "ok":
            rows.append({
                "Venue": row["Venue"], "Year": row["Year"],
                "Title": row["Title"], "Abstract": row["Abstract"],
                "URL": row["URL"],
                "Judge_include":    rec["Judge_include"],
                "Judge_confidence": rec["Judge_confidence"],
                "Judge_reason":     rec["Judge_reason"],
            })
    pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)

    print("\n" + "="*60)
    print(f"Judged (new):   {ok}")
    print(f"Skipped (done): {skipped}")
    print(f"No PDF:         {no_pdf}")
    print(f"Errors:         {errors}")
    print(f"Output CSV:     {OUTPUT_CSV}")
    print(f"Log JSONL:      {LOG_JSONL}")


if __name__ == "__main__":
    main()
