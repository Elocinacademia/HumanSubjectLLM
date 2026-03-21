#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import threading
from typing import Dict, List, Optional, Set
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


ISSUE_IDS = list(range(547,560))  
YEAR = "2023"                     


OUT_CSV = "aaai_2023_2025.csv"
CHECKPOINT_JSONL = "aaai_2023_2025_checkpoint.jsonl"

MAX_WORKERS = 8
REQUESTS_PER_SECOND = 2.0
MAX_RETRIES = 6
TIMEOUT = 35
SAVE_EVERY = 20

# Two-pass behavior:
# Pass 1: crawl ALL papers (even if already in checkpoint)
# Pass 2: re-crawl ONLY those whose Abstract is still empty after pass 1
DO_SECOND_PASS_FOR_EMPTY_ABSTRACT = True


BASE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/122.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

OJS_VIEW_PREFIX = "https://ojs.aaai.org/index.php/AAAI/article/view/"
OJS_ISSUE_PREFIX = "https://ojs.aaai.org/index.php/AAAI/issue/view/"


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
            r = session.get(url, timeout=TIMEOUT, allow_redirects=allow_redirects)

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
                    records[u] = obj  # keep latest for each URL
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



def extract_submission_id_from_url(u: str) -> Optional[str]:
    if not u:
        return None
    m = re.search(r"/article/view/(\d+)", u)
    if m:
        return m.group(1)
    m = re.search(r"/article/download/(\d+)/", u)
    if m:
        return m.group(1)
    m = re.search(r"cdn\.aaai\.org/ojs/(\d+)/", u)
    if m:
        return m.group(1)
    m = re.search(r"/ojs/(\d+)/", u)
    if m:
        return m.group(1)
    return None


def to_view_url(u: str) -> Optional[str]:
    sid = extract_submission_id_from_url(u)
    if sid:
        return f"{OJS_VIEW_PREFIX}{sid}"
    if "index.php/AAAI/article/view/" in u:
        return u.split("#")[0]
    return None


# =========================
# Issue page -> view urls
# =========================
def collect_view_urls_from_issue(issue_html: str, issue_url: str) -> List[str]:
    soup = BeautifulSoup(issue_html, "html.parser")
    found: List[str] = []

    for a in soup.select("a[href]"):
        href = a.get("href")
        if not href:
            continue
        full = urljoin(issue_url, href.split("#")[0])
        if ("/AAAI/article/view/" in full) or ("/AAAI/article/download/" in full) or ("cdn.aaai.org/ojs/" in full):
            vu = to_view_url(full)
            if vu:
                found.append(vu)

    seen: Set[str] = set()
    out: List[str] = []
    for u in found:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


# =========================
# Article parsing (Title/Authors/Abstract)
# =========================
def meta_all_case_insensitive(soup: BeautifulSoup, name: str) -> List[str]:
    target = name.lower()
    vals = []
    for m in soup.find_all("meta"):
        n = m.get("name")
        if not n:
            continue
        if n.lower() == target:
            c = m.get("content")
            if c:
                vals.append(norm(c))
    return [v for v in vals if v]


def get_first_meta_any(soup: BeautifulSoup, names: List[str]) -> str:
    for n in names:
        vals = meta_all_case_insensitive(soup, n)
        if vals:
            return vals[0]
    return ""


def extract_jsonld_description(soup: BeautifulSoup) -> str:
    for s in soup.select('script[type="application/ld+json"]'):
        txt = s.get_text(strip=True)
        if not txt:
            continue
        try:
            obj = json.loads(txt)
        except Exception:
            continue

        candidates = []
        if isinstance(obj, dict):
            candidates = [obj]
            if "@graph" in obj and isinstance(obj["@graph"], list):
                candidates += [x for x in obj["@graph"] if isinstance(x, dict)]
        elif isinstance(obj, list):
            candidates = [x for x in obj if isinstance(x, dict)]

        for c in candidates:
            desc = c.get("description")
            if isinstance(desc, str) and norm(desc):
                return norm(desc)
            if isinstance(desc, dict):
                v = desc.get("@value")
                if isinstance(v, str) and norm(v):
                    return norm(v)
    return ""


def extract_abstract_from_html_blocks(soup: BeautifulSoup) -> str:
    selectors = [
        ".obj_article_details .item.abstract .value",
        ".obj_article_details .item.abstract",
        ".item.abstract .value",
        ".item.abstract",
        "section.item.abstract .value",
        "section.item.abstract",
        "#articleAbstract",
        ".article__abstract",
    ]
    for sel in selectors:
        node = soup.select_one(sel)
        if node:
            t = norm(node.get_text(" ", strip=True))
            if t and len(t) > 30 and t.lower() != "abstract":
                t = re.sub(r"^abstract\s*[:\-]?\s*", "", t, flags=re.I)
                return norm(t)
    return ""


def parse_article_html(html: str, url: str, year: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, "html.parser")

    title = get_first_meta_any(soup, ["citation_title"])
    authors_list = meta_all_case_insensitive(soup, "citation_author")
    authors = "; ".join(authors_list) if authors_list else ""

    abstract = get_first_meta_any(soup, [
        "citation_abstract",
        "dcterms.abstract",
        "dcterms:abstract",
        "dc.description",
        "DC.Description",
        "description",
    ])

    if not abstract:
        abstract = extract_jsonld_description(soup)
    if not abstract:
        abstract = extract_abstract_from_html_blocks(soup)

    if not title:
        h1 = soup.select_one("h1.page_title") or soup.find("h1")
        if h1:
            title = norm(h1.get_text(" ", strip=True))

    if not authors:
        auth_nodes = soup.select(".authors .name, .item.authors .name, .pkp_author_name")
        tmp = [norm(n.get_text(" ", strip=True)) for n in auth_nodes if norm(n.get_text(" ", strip=True))]
        if tmp:
            authors = "; ".join(tmp)

    return {
        "Venue": "AAAI",
        "Year": year,
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "URL": url,
    }


def fetch_article_view_html(session: requests.Session, view_url: str) -> Optional[str]:
    """
    If /article/view/<id> hard-redirects to PDF, return None.
    """
    r = fetch(session, view_url, allow_redirects=False)

    if r.status_code in (301, 302, 303, 307, 308):
        root = f"{urlparse(view_url).scheme}://{urlparse(view_url).netloc}/"
        try:
            fetch(session, root, allow_redirects=True)
        except Exception:
            pass
        r2 = fetch(session, view_url, allow_redirects=False)
        if r2.status_code in (301, 302, 303, 307, 308):
            return None

    return r.text


def crawl_one(view_url: str, year: str) -> Dict[str, str]:
    s = build_session()
    html = fetch_article_view_html(s, view_url)
    if html is None:
        return {"Venue": "AAAI", "Year": year, "Title": "", "Authors": "", "Abstract": "", "URL": view_url}
    return parse_article_html(html, view_url, year)


# =========================
# Crawl a batch with progress ++++ autosave
# =========================
def crawl_batch(urls: List[str], year: str, records: Dict[str, Dict], desc: str) -> None:
    if not urls:
        return

    new_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as ex:
        futures = {ex.submit(crawl_one, u, year): u for u in urls}
        with tqdm(total=len(futures), desc=desc, unit="paper") as pbar:
            for fut in as_completed(futures):
                u = futures[fut]
                try:
                    rec = fut.result()
                    records[rec["URL"]] = rec  # overwrite/update => no duplicates in final dict
                    append_checkpoint(rec)

                    new_count += 1
                    if new_count % SAVE_EVERY == 0:
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
    all_view_urls: List[str] = []
    seen_urls: Set[str] = set()

    for issue_id in tqdm(ISSUE_IDS, desc="Scanning issues", unit="issue"):
        issue_url = f"{OJS_ISSUE_PREFIX}{issue_id}"
        try:
            issue_resp = fetch(sess, issue_url, allow_redirects=True)
            view_urls = collect_view_urls_from_issue(issue_resp.text, issue_url)
        except Exception as e:
            print(f"[warn] issue {issue_id} failed: {e}")
            continue

        for u in view_urls:
            if u not in seen_urls:
                all_view_urls.append(u)
                seen_urls.add(u)

    print(f"[info] total unique papers from issues {ISSUE_IDS[0]}-{ISSUE_IDS[-1]}: {len(all_view_urls)}")

    # 2) PASS 1: crawl ALL (regardless checkpoint)
    crawl_batch(
        urls=all_view_urls,
        year=YEAR,
        records=records,
        desc="PASS 1: Crawling all papers"
    )

    # 3) PASS 2: re-crawl those still empty abstract forcely
    if DO_SECOND_PASS_FOR_EMPTY_ABSTRACT:
        empty_abs_urls = [u for u in all_view_urls if not records.get(u, {}).get("Abstract")]
        print(f"[info] PASS 2 targets (empty Abstract after pass 1): {len(empty_abs_urls)}")
        if empty_abs_urls:
            crawl_batch(
                urls=empty_abs_urls,
                year=YEAR,
                records=records,
                desc="PASS 2: Re-crawling empty-abstract"
            )

    # 4) final stats
    df = pd.DataFrame(list(records.values()))
    df = df.drop_duplicates(subset=["URL"])
    empty_title = int((df["Title"].fillna("") == "").sum()) if "Title" in df.columns else 0
    empty_auth = int((df["Authors"].fillna("") == "").sum()) if "Authors" in df.columns else 0
    empty_abs = int((df["Abstract"].fillna("") == "").sum()) if "Abstract" in df.columns else 0

    print(f"[done] saved {len(df)} unique rows to {OUT_CSV}")
    print(f"[stats] empty Title={empty_title} | empty Authors={empty_auth} | empty Abstract={empty_abs}")
    print("[note] If remaining abstracts are empty, those entries likely hard-redirect to PDF and need PDF fallback parsing.")


if __name__ == "__main__":
    main()