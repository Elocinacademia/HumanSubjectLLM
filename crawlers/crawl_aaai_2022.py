#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import threading
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed



YEAR = "2022"
VENUE = "AAAI"

TRACK_URLS = [
    f"https://aaai.org/proceeding/{i:02d}-aaai-22-technical-tracks-{i}/"
    for i in range(1, 11)
] + [
    "https://aaai.org/proceeding/11-iaai-22-eaai-22-aaai-22-special-programs-student-papers-demonstrations/"
]

OUT_CSV = "aaai_2022.csv"
CHECKPOINT_JSONL = "aaai_2022_checkpoint.jsonl"

MAX_WORKERS = 8
REQUESTS_PER_SECOND = 2.0
MAX_RETRIES = 6
TIMEOUT = 35
SAVE_EVERY = 20

DO_SECOND_PASS_FOR_EMPTY_ABSTRACT = True
# =======================================================


BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

PAPER_PREFIX = "https://aaai.org/papers/"


# =========================
# Rate Limiter
# =========================
class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(rps, 0.0001)
        self._lock = threading.Lock()
        self._next_allowed = time.monotonic()

    def wait(self):
        with self._lock:
            now = time.monotonic()
            if now < self._next_allowed:
                time.sleep(self._next_allowed - now)
            self._next_allowed = max(self._next_allowed, time.monotonic()) + self.min_interval


rate_limiter = RateLimiter(REQUESTS_PER_SECOND)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(BASE_HEADERS)
    return s


def fetch(session: requests.Session, url: str, allow_redirects: bool = True) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rate_limiter.wait()
            headers = {"Referer": "https://aaai.org/"}
            r = session.get(url, timeout=TIMEOUT, allow_redirects=allow_redirects, headers=headers)

            if r.status_code in (403, 429, 500, 502, 503, 504):
                sleep_s = min(90, (2 ** (attempt - 1)) + random.random() * 2)
                print(f"[retry] HTTP {r.status_code} {url} sleep={sleep_s:.1f}s")
                time.sleep(sleep_s)
                continue

            r.raise_for_status()
            r.encoding = r.apparent_encoding or r.encoding
            return r

        except Exception as e:
            last_exc = e
            sleep_s = min(90, (2 ** (attempt - 1)) + random.random() * 2)
            print(f"[retry] error {e} {url} sleep={sleep_s:.1f}s")
            time.sleep(sleep_s)

    raise last_exc if last_exc else RuntimeError(f"Failed to fetch {url}")


# =========================
# Checkpoint + CSV
# =========================
def load_checkpoint() -> Dict[str, Dict]:
    records: Dict[str, Dict] = {}
    if not os.path.exists(CHECKPOINT_JSONL):
        return records
    with open(CHECKPOINT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                u = obj.get("URL")
                if u:
                    records[u] = obj
            except json.JSONDecodeError:
                continue
    print(f"[info] loaded {len(records)} records from checkpoint")
    return records


_checkpoint_lock = threading.Lock()
def append_checkpoint(record: Dict) -> None:
    with _checkpoint_lock:
        with open(CHECKPOINT_JSONL, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


_csv_lock = threading.Lock()
def save_csv(records_dict: Dict[str, Dict]) -> None:
    df = pd.DataFrame(list(records_dict.values()))
    cols = ["Venue", "Year", "Title", "Authors", "Abstract", "URL"]
    for c in cols:
        if c not in df.columns:
            df[c] = ""
    df = df[cols].drop_duplicates(subset=["URL"])
    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    print(f"[autosave] {OUT_CSV} rows={len(df)}")


# =========================
# Track archive -> paper URLs (pagination)
# =========================
def is_paper_url(u: str) -> bool:
    return u.startswith(PAPER_PREFIX) and u.rstrip("/").split("/")[-1] != "papers"


def collect_paper_urls_from_track(track_url: str, session: requests.Session) -> List[str]:
    out: List[str] = []
    seen: Set[str] = set()
    page = 1

    while True:
        url = track_url if page == 1 else urljoin(track_url, f"page/{page}/")
        try:
            html = fetch(session, url, allow_redirects=True).text
        except Exception as e:
            if page == 1:
                raise
            print(f"[warn] stop pagination at {url}: {e}")
            break

        soup = BeautifulSoup(html, "html.parser")

        links = []
        for a in soup.select("a[href]"):
            href = a.get("href")
            if not href:
                continue
            full = urljoin(url, href.split("#")[0])
            if is_paper_url(full):
                links.append(full)

        added = 0
        for u in links:
            if u not in seen:
                seen.add(u)
                out.append(u)
                added += 1

        if added == 0:
            break

        page += 1

    return out


# =========================
# Paper page parsing helpers
# =========================
def meta_name_all(soup: BeautifulSoup, name: str) -> List[str]:
    target = name.lower()
    vals: List[str] = []
    for m in soup.find_all("meta"):
        n = (m.get("name") or "").lower()
        if n == target:
            c = m.get("content")
            if c:
                vals.append(norm(c))
    return [v for v in vals if v]


def meta_name_first(soup: BeautifulSoup, name: str) -> str:
    vals = meta_name_all(soup, name)
    return vals[0] if vals else ""


def meta_prop(soup: BeautifulSoup, prop: str) -> Optional[str]:
    m = soup.find("meta", attrs={"property": prop})
    if m and m.get("content"):
        return norm(m["content"])
    return None


def extract_labeled_attribute(soup: BeautifulSoup, target_label: str) -> str:
    want = target_label.lower().rstrip(":")
    for sec in soup.select("div.paper-section-wrap"):
        h4 = sec.find("h4")
        if not h4:
            continue
        label = norm(h4.get_text(" ", strip=True)).lower().rstrip(":")
        if label == want:
            box = sec.select_one("div.attribute-output") or sec
            t = norm(box.get_text(" ", strip=True))
            t = re.sub(rf"^{re.escape(target_label)}\s*[:\-]?\s*", "", t, flags=re.I)
            return t
    return ""


def extract_authors_aaai(soup: BeautifulSoup) -> str:
    """
    Priority :
    1) meta DC.creator (best, pure names)
    2) meta citation_author
    3) .author-wrap .author-output p.bold (names only)
    4) labeled Authors block (paper-section-wrap)
    5) other fallbacks
    """
   
    dc_creator = meta_name_first(soup, "DC.creator")
    if dc_creator:
        return dc_creator


    citation_authors = meta_name_all(soup, "citation_author")
    if citation_authors:
        return "; ".join(citation_authors)

    
    names = []
    for p in soup.select(".author-wrap .author-output p.bold"):
        t = norm(p.get_text(" ", strip=True))
        if t:
            names.append(t)
    if names:
        # keep order, dedup
        seen = set()
        uniq = []
        for n in names:
            if n not in seen:
                uniq.append(n)
                seen.add(n)
        return "; ".join(uniq)

    # 4) labeled Authors in paper-section-wrap (if present)
    t = extract_labeled_attribute(soup, "Authors")
    if t:
        t = re.sub(r"^authors?\s*[:\-]?\s*", "", t, flags=re.I).strip()
        if 2 <= len(t) <= 1000:
            return t

    # 5) old fallbacks
    for wrap in soup.select(".papers-author-page, .author-wrap"):
        tt = norm(wrap.get_text(" ", strip=True))
        tt = re.sub(r"^authors?\s*[:\-]?\s*", "", tt, flags=re.I).strip()
        if 2 <= len(tt) <= 1000:
            return tt

    return ""


def extract_abstract_aaai(soup: BeautifulSoup) -> str:
    # 1) citation meta
    abs_meta = meta_name_first(soup, "citation_abstract")
    if abs_meta and len(abs_meta) > 30:
        return abs_meta

    # 2) NEW: paper-section-wrap + attribute-output
    t = extract_labeled_attribute(soup, "Abstract")
    if t:
        t = re.sub(r"^abstract\s*[:\-]?\s*", "", t, flags=re.I).strip()
        if len(t) > 30:
            return t

    # 3) older rules
    for sel in [
        ".abstract-output",
        ".abstract-output p",
        ".paper-section-wrap .abstract-output",
        ".paper-section-wrap .abstract-output p",
    ]:
        node = soup.select_one(sel)
        if node:
            tt = norm(node.get_text(" ", strip=True))
            tt = re.sub(r"^abstract\s*[:\-]?\s*", "", tt, flags=re.I)
            if len(tt) > 30:
                return tt

    # 4) heading "Abstract"
    for h in soup.find_all(["h2", "h3", "h4", "strong"]):
        if norm(h.get_text(" ", strip=True)).lower().rstrip(":") == "abstract":
            p = h.find_next("p")
            if p:
                tt = norm(p.get_text(" ", strip=True))
                tt = re.sub(r"^abstract\s*[:\-]?\s*", "", tt, flags=re.I)
                if len(tt) > 30:
                    return tt

    # 5) og:description
    ogd = meta_prop(soup, "og:description")
    if ogd and len(ogd) > 30:
        return ogd

    return ""


def parse_aaai_paper_html(html: str, url: str, year: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title = ""
    h1 = soup.select_one("h1.entry-title") or soup.find("h1")
    if h1:
        title = norm(h1.get_text(" ", strip=True))

    if not title:
        t = meta_prop(soup, "og:title")
        if t:
            title = t
        else:
            ct = meta_name_first(soup, "DC.title") or meta_name_first(soup, "citation_title")
            if ct:
                title = ct

    authors = extract_authors_aaai(soup)
    abstract = extract_abstract_aaai(soup)

    return {
        "Venue": VENUE,
        "Year": year,
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "URL": url,
    }


def crawl_one(paper_url: str, year: str) -> Dict[str, str]:
    s = build_session()
    r = fetch(s, paper_url, allow_redirects=True)

    ct = (r.headers.get("Content-Type") or "").lower()
    if "application/pdf" in ct:
        return {"Venue": VENUE, "Year": year, "Title": "", "Authors": "", "Abstract": "", "URL": paper_url}

    return parse_aaai_paper_html(r.text, paper_url, year)


# =========================
# Crawl batch with progress + autosave
# =========================
def crawl_batch(urls: List[str], year: str, records: Dict[str, Dict], desc: str) -> None:
    if not urls:
        return

    count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(crawl_one, u, year): u for u in urls}
        with tqdm(total=len(futures), desc=desc, unit="paper") as pbar:
            for fut in as_completed(futures):
                u = futures[fut]
                try:
                    rec = fut.result()
                    records[rec["URL"]] = rec  # dedup by URL
                    append_checkpoint(rec)

                    count += 1
                    if count % SAVE_EVERY == 0:
                        with _csv_lock:
                            save_csv(records)

                except Exception as e:
                    print(f"[error] {u}: {e}")

                pbar.update(1)

    with _csv_lock:
        save_csv(records)



def main():
    records = load_checkpoint()
    sess = build_session()

    # 1) scan all tracks and collect unique paper urls
    all_urls: List[str] = []
    seen: Set[str] = set()

    for track in tqdm(TRACK_URLS, desc="Scanning tracks", unit="track"):
        try:
            urls = collect_paper_urls_from_track(track, sess)
        except Exception as e:
            print(f"[warn] track failed {track}: {e}")
            continue

        for u in urls:
            if u not in seen:
                seen.add(u)
                all_urls.append(u)

    print(f"[info] total unique paper pages collected: {len(all_urls)}")

    # 2) PASS 1
    crawl_batch(all_urls, YEAR, records, desc="PASS 1: Crawling all papers")

    # 3) PASS 2 (empty abstract)
    if DO_SECOND_PASS_FOR_EMPTY_ABSTRACT:
        empty_abs_urls = [u for u in all_urls if not records.get(u, {}).get("Abstract")]
        print(f"[info] PASS 2 targets (empty Abstract after pass 1): {len(empty_abs_urls)}")
        if empty_abs_urls:
            crawl_batch(empty_abs_urls, YEAR, records, desc="PASS 2: Re-crawling empty-abstract")

    # 4) stats
    df = pd.DataFrame(list(records.values())).drop_duplicates(subset=["URL"])
    empty_title = int((df["Title"].fillna("") == "").sum()) if "Title" in df.columns else 0
    empty_auth = int((df["Authors"].fillna("") == "").sum()) if "Authors" in df.columns else 0
    empty_abs = int((df["Abstract"].fillna("") == "").sum()) if "Abstract" in df.columns else 0

    print(f"[done] saved {len(df)} unique rows to {OUT_CSV}")
    print(f"[stats] empty Title={empty_title} | empty Authors={empty_auth} | empty Abstract={empty_abs}")


if __name__ == "__main__":
    main()