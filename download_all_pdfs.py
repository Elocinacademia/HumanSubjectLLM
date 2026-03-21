"""Short README: download_all_pdfs.py

Download PDFs for all papers in papers_all_keyword_filtered.csv.
Already-downloaded files are skipped automatically.

Strategy priority per paper, the code will follow this logic:
  1. ACM DL PDF (using cookies, please download one from the browser)
  2. Direct URL (works for AAAI, IJCAI, arXiv, etc.)
    - note this doesn't work for some AAAI paper because the URL is incorrect
  3. Semantic Scholar open-access PDF  (backup plan for ACM papers with preprints)
  4. Unpaywall open-access PDF (DOI-based; email used as identifier only)



Put the papers_all_keyword_filtered.csv in the save/ folder, and output is in a pdfs/ folder
Replace the UNPAYWALL_EMAIL with your email address (any valid emaill address)
Download cookies.txt and save in the same folder as this .py
"""

import os
import re
import csv
import io
import time
import math
import json
import hashlib
import random
from http.cookiejar import MozillaCookieJar
from pathlib import Path
from urllib.parse import urljoin, urlparse
from typing import Optional

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

CSV_PATH         = Path("save") / "papers_all_keyword_filtered.csv"
OUT_DIR          = Path("pdfs")
UNPAYWALL_EMAIL  = "xiao.zhan@kcl.ac.uk"
COOKIES_FILE     = Path("cookies.txt")

TIMEOUT             = 60
MAX_RETRIES         = 4
REQUESTS_PER_SECOND = 1.5
CHUNK_SIZE          = 1024 * 256
MIN_VALID_PDF_BYTES = 10_000


class RateLimiter:
    def __init__(self, rps: float):
        self.min_interval = 1.0 / max(rps, 1e-6)
        self.next_allowed = time.monotonic()

    def wait(self):
        now = time.monotonic()
        if now < self.next_allowed:
            time.sleep(self.next_allowed - now)
        self.next_allowed = max(self.next_allowed, time.monotonic()) + self.min_interval


rate_limiter = RateLimiter(REQUESTS_PER_SECOND)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip()


def safe_filename(s: str, max_len: int = 160) -> str:
    s = norm(s)
    s = re.sub(r"[^\w\-\.\(\)\[\] ]+", "_", s)
    s = re.sub(r"\s+", " ", s).strip().replace(" ", "_")
    if len(s) > max_len:
        s = s[:max_len]
    return s or "paper"


def sha1_short(s: str, n: int = 10) -> str:
    return hashlib.sha1(s.encode("utf-8", errors="ignore")).hexdigest()[:n]


def build_output_path(venue: str, year: str, title: str, url: str) -> Path:
    fname = (
        f"{safe_filename(venue, max_len=50)}"
        f"_{safe_filename(year)}"
        f"_{sha1_short(url)}"
        f"_{safe_filename(title, max_len=120)}.pdf"
    )
    return OUT_DIR / fname


def build_session() -> requests.Session:
    sess = requests.Session()
    sess.headers.update({
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    if COOKIES_FILE.exists():
        jar = MozillaCookieJar(str(COOKIES_FILE))
        jar.load(ignore_discard=True, ignore_expires=True)
        sess.cookies.update(jar)
    return sess


def find_acm_pdf_url(doi: Optional[str]) -> Optional[str]:
    """Construct ACM DL direct PDF URL from DOI."""
    if doi and doi.startswith("10.1145/"):
        return f"https://dl.acm.org/doi/pdf/{doi}"
    return None


def fetch(sess: requests.Session, url: str, stream: bool = False) -> requests.Response:
    last_exc = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            rate_limiter.wait()
            r = sess.get(url, timeout=TIMEOUT, allow_redirects=True, stream=stream)
            if r.status_code in (403, 429, 500, 502, 503, 504):
                time.sleep(min(60, (2 ** (attempt - 1)) + random.random() * 2))
                continue
            r.raise_for_status()
            return r
        except Exception as e:
            last_exc = e
            time.sleep(min(60, (2 ** (attempt - 1)) + random.random() * 2))
    raise last_exc or RuntimeError(f"Failed: {url}")


def looks_like_pdf_url(u: str) -> bool:
    u = (u or "").lower().split("?")[0].split("#")[0]
    return u.endswith(".pdf")


def choose_best_pdf(candidates: list) -> Optional[str]:
    if not candidates:
        return None
    bad_kw = ["supp", "supplement", "bibtex", "cite", "poster", "slides",
              "ppt", "video", "code", "dataset", "appendix", "award", "presentation"]
    def score(u):
        return (sum(1 for k in bad_kw if k in u.lower()), len(u))
    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=score)
    return candidates[0]


def extract_pdf_url_generic(html: str, page_url: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
    for m in soup.find_all("meta"):
        name = (m.get("name") or m.get("property") or "").lower()
        if name in ("citation_pdf_url", "dc.identifier") and m.get("content"):
            u = norm(m["content"])
            if looks_like_pdf_url(u):
                return u
    pdfs = []
    for a in soup.select("a[href]"):
        href = a.get("href") or ""
        full = urljoin(page_url, href)
        if looks_like_pdf_url(full):
            pdfs.append(full)
    return choose_best_pdf(pdfs)


def find_direct_pdf_url(sess: requests.Session, url: str) -> Optional[str]:
    """Try to find a PDF URL directly from the paper landing page."""
    url = norm(url)
    if not url:
        return None
    if looks_like_pdf_url(url):
        return url
    try:
        r = fetch(sess, url, stream=False)
        ct = (r.headers.get("Content-Type") or "").lower()
        if "application/pdf" in ct:
            return r.url
        return extract_pdf_url_generic(r.text, r.url)
    except Exception:
        return None


# ─────────────────────────────────────────────
# Open-access fallbacks
# ─────────────────────────────────────────────

def extract_doi(url: str) -> Optional[str]:
    m = re.search(r"(10\.\d{4,}/[^\s\"'<>]+)", url)
    if m:
        return m.group(1).rstrip("./")
    return None


def find_oa_url_semantic_scholar(doi: Optional[str], title: str) -> Optional[str]:
    headers = {"User-Agent": "paper-downloader/1.0 (research)"}
    if doi:
        try:
            r = requests.get(
                f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                params={"fields": "openAccessPdf"},
                headers=headers, timeout=15
            )
            if r.status_code == 200:
                oa = r.json().get("openAccessPdf")
                if oa and oa.get("url"):
                    return oa["url"]
        except Exception:
            pass
    # title search fallback
    try:
        r = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params={"query": title, "fields": "openAccessPdf", "limit": 3},
            headers=headers, timeout=15
        )
        if r.status_code == 200:
            for paper in r.json().get("data", []):
                oa = paper.get("openAccessPdf")
                if oa and oa.get("url"):
                    return oa["url"]
    except Exception:
        pass
    return None


def find_oa_url_unpaywall(doi: str, email: str) -> Optional[str]:
    if not doi:
        return None
    try:
        r = requests.get(
            f"https://api.unpaywall.org/v2/{doi}",
            params={"email": email}, timeout=15,
            headers={"User-Agent": "paper-downloader/1.0"}
        )
        if r.status_code == 200:
            best = r.json().get("best_oa_location")
            if best and best.get("url_for_pdf"):
                return best["url_for_pdf"]
    except Exception:
        pass
    return None


def download_pdf_to_file(sess: requests.Session, pdf_url: str, out_path: Path) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(out_path) + ".part")
    try:
        r = fetch(sess, pdf_url, stream=True)
        ct = (r.headers.get("Content-Type") or "").lower()
        total = 0
        first_chunk = None
        with tmp.open("wb") as f:
            for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                if first_chunk is None:
                    first_chunk = chunk[:16]
                f.write(chunk)
                total += len(chunk)
        if total < MIN_VALID_PDF_BYTES:
            tmp.unlink(missing_ok=True)
            return False
        if first_chunk and b"%PDF" not in first_chunk and "pdf" not in ct:
            tmp.unlink(missing_ok=True)
            return False
        tmp.replace(out_path)
        return True
    except Exception:
        tmp.unlink(missing_ok=True)
        return False


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with CSV_PATH.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Total papers in CSV: {len(rows)}")

    sess = build_session()
    ok = skipped = fail_direct = fail_s2 = fail_all = 0

    for row in tqdm(rows, desc="Downloading", unit="paper"):
        url   = norm(str(row.get("URL", "")))
        title = norm(str(row.get("Title", "")))
        venue = norm(str(row.get("Venue", "")))
        year  = norm(str(row.get("Year", "")))

        if not url:
            fail_all += 1
            continue

        out_path = build_output_path(venue, year, title, url)

        # Skip if already downloaded
        if out_path.exists() and out_path.stat().st_size >= MIN_VALID_PDF_BYTES:
            skipped += 1
            continue

        doi = extract_doi(url)
        pdf_url = None

        # 1. ACM DL (authenticated) 
        acm_pdf_url = find_acm_pdf_url(doi)
        if acm_pdf_url:
            if download_pdf_to_file(sess, acm_pdf_url, out_path):
                ok += 1
                tqdm.write(f"  [acm]    {title[:60]}")
                continue

        # 2. Direct URL 
        try:
            pdf_url = find_direct_pdf_url(sess, url)
        except Exception:
            pass

        if pdf_url:
            if download_pdf_to_file(sess, pdf_url, out_path):
                ok += 1
                tqdm.write(f"  [direct] {title[:60]}")
                continue
        fail_direct += 1

        # 3. Semantic Scholar
        try:
            oa_url = find_oa_url_semantic_scholar(doi, title)
            if oa_url:
                if download_pdf_to_file(sess, oa_url, out_path):
                    ok += 1
                    tqdm.write(f"  [s2]     {title[:60]}")
                    continue
        except Exception:
            pass
        fail_s2 += 1

        # 4. Unpaywall 
        try:
            oa_url = find_oa_url_unpaywall(doi, UNPAYWALL_EMAIL)
            if oa_url:
                if download_pdf_to_file(sess, oa_url, out_path):
                    ok += 1
                    tqdm.write(f"  [unp]    {title[:60]}")
                    continue
        except Exception:
            pass

        fail_all += 1
        tqdm.write(f"  [FAIL]   {title[:60]}")

    print("\n" + "=" * 60)
    print(f"Downloaded (new):   {ok}")
    print(f"Skipped (existing): {skipped}")
    print(f"Failed (all paths): {fail_all}")
    print(f"Output dir:         {OUT_DIR}")


if __name__ == "__main__":
    main()
