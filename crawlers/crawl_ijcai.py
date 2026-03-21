#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import time
import random
import threading
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm



BASE = "https://www.ijcai.org"
YEARS = [2022, 2023, 2024, 2025]

OUT_CSV = "ijcai_2022_2025.csv"
CHECKPOINT_JSONL = "ijcai_2022_2025_checkpoint.jsonl"

MAX_WORKERS = 8
REQUESTS_PER_SECOND = 2.0
MAX_RETRIES = 6
TIMEOUT = 25
SAVE_EVERY = 20

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; IJCAI-Crawler/1.2)"
}


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


def build_session() -> requests.Session:
    s = requests.Session()
    s.headers.update(HEADERS)
    return s


def fetch(session: requests.Session, url: str) -> str:
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rate_limiter.wait()
            resp = session.get(url, timeout=TIMEOUT)

            if resp.status_code in (429, 500, 502, 503, 504):
                sleep_s = min(60, (2 ** (attempt - 1)) + random.random())
                time.sleep(sleep_s)
                continue

            resp.raise_for_status()
            resp.encoding = resp.apparent_encoding
            return resp.text
        except Exception:
            sleep_s = min(60, (2 ** (attempt - 1)) + random.random())
            time.sleep(sleep_s)

    raise RuntimeError(f"Failed to fetch {url}")


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


# =========================
# IJCAI: list page -> details links
# =========================
def get_year_links(session: requests.Session, year: int) -> List[str]:
    url = f"{BASE}/proceedings/{year}"
    html = fetch(session, url)
    soup = BeautifulSoup(html, "html.parser")

    links = []
    for a in soup.find_all("a"):
        if a.get_text(strip=True).lower() == "details":
            href = a.get("href")
            if href:
                links.append(urljoin(BASE, href))

    print(f"[info] {year} found {len(links)} detail links")
    return links


# =========================
# IJCAI: details page parsing (fixed for real HTML)
# =========================
def extract_title_authors(soup: BeautifulSoup) -> Tuple[str, str]:
    """
    Prefer the proceedings-detail area:
      <div class="container-fluid proceedings-detail">
        <div class="row">
          <div class="col-md-8 ...">
            <h1>Title</h1>
            <h2>Authors</h2>
    """
    container = soup.select_one("div.proceedings-detail")
    if container:
        h1 = container.select_one("div.col-md-8 h1")
        h2 = container.select_one("div.col-md-8 h2")
        title = norm(h1.get_text(" ", strip=True)) if h1 else ""
        authors = norm(h2.get_text(" ", strip=True)) if h2 else ""
        if title or authors:
            return title, authors

    # fallback
    page_title = soup.select_one("h1.page-title")
    title = norm(page_title.get_text(" ", strip=True)) if page_title else ""
    authors = ""
    return title, authors


def extract_abstract(soup: BeautifulSoup) -> str:
    
    container = soup.select_one("div.proceedings-detail") or soup

    hr = container.find("hr")
    if not hr:
        return ""

    row = hr.find_next("div", class_="row")
    if not row:
        return ""

    parts = []
    for col in row.find_all("div", class_="col-md-12", recursive=False):
        text = norm(col.get_text(" ", strip=True))
        if not text:
            continue
        if "keywords:" in text.lower():
            break
        parts.append(text)

    # Sometimes col-md-12 is nested (not direct children). If above gives empty, try recursive search:
    if not parts:
        for col in row.select("div.col-md-12"):
            text = norm(col.get_text(" ", strip=True))
            if not text:
                continue
            if "keywords:" in text.lower():
                break
            # skip the "Keywords:" container itself if it formats differently
            if col.select_one(".keywords"):
                break
            parts.append(text)

    return norm(" ".join(parts))


def parse_detail(session: requests.Session, url: str, year: int) -> Dict[str, str]:
    html = fetch(session, url)
    soup = BeautifulSoup(html, "html.parser")

    title, authors = extract_title_authors(soup)
    abstract = extract_abstract(soup)

    return {
        "Venue": "IJCAI",
        "Year": str(year),
        "Title": title,
        "Authors": authors,
        "Abstract": abstract,
        "URL": url
    }


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
            obj = json.loads(line)
            if "URL" in obj:
                records[obj["URL"]] = obj
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



def main():
    records = load_checkpoint()
    session = build_session()

    tasks: List[Tuple[int, str]] = []
    for y in YEARS:
        links = get_year_links(session, y)
        for u in links:
            if u not in records:
                tasks.append((y, u))

    print(f"[info] pending {len(tasks)} papers")

    new_count = 0
    thread_local = threading.local()

    def get_thread_session():
        if not hasattr(thread_local, "session"):
            thread_local.session = build_session()
        return thread_local.session

    def worker(item):
        y, u = item
        s = get_thread_session()
        return parse_detail(s, u, y)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, t): t for t in tasks}
        with tqdm(total=len(futures), desc="Crawling IJCAI", unit="paper") as pbar:
            for fut in as_completed(futures):
                try:
                    rec = fut.result()
                    records[rec["URL"]] = rec
                    append_checkpoint(rec)

                    new_count += 1
                    if new_count % SAVE_EVERY == 0:
                        with _csv_lock:
                            save_csv(records)

                except Exception as e:
                    print("[error]", e)

                pbar.update(1)

    save_csv(records)
    print(f"[done] saved {len(records)} rows to {OUT_CSV}")


if __name__ == "__main__":
    main()