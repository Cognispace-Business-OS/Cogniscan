"""
YC Radar — Launch HN Scraper
Fetches YC company launches from Hacker News Algolia API
Extracts: company name, batch, founder info, traction signals
"""

import requests
import re
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict


# ── Config ────────────────────────────────────────────────────────────────────

ALGOLIA_BASE = "https://hn.algolia.com/api/v1"
HN_BASE      = "https://hacker-news.firebaseio.com/v0"
HEADERS      = {"User-Agent": "YC-Radar-Bot/1.0 (research project)"}
RATE_LIMIT   = 0.5   # seconds between requests — be a good citizen


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class YCCompany:
    name: str
    batch: str                  # e.g. "W25", "S24"
    one_liner: str
    hn_url: str
    hn_points: int
    hn_comments: int
    launched_at: str            # ISO date string
    story_id: int
    founder_names: list[str]
    tags: list[str]             # extracted keywords
    github_url: str | None = None
    website_url: str | None = None


# ── Fetch Launch HN Posts ─────────────────────────────────────────────────────

def fetch_launch_hn_posts(pages: int = 5, per_page: int = 50) -> list[dict]:
    """
    Fetch 'Launch HN' stories from Algolia HN API.
    Returns raw hit dicts.
    """
    all_hits = []

    for page in range(pages):
        url = f"{ALGOLIA_BASE}/search"
        params = {
            "query"      : "Launch HN",
            "tags"       : "story",
            "hitsPerPage": per_page,
            "page"       : page,
        }

        resp = requests.get(url, params=params, headers=HEADERS, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        hits = data.get("hits", [])
        if not hits:
            break

        all_hits.extend(hits)
        print(f"  Page {page+1}: fetched {len(hits)} posts (total: {len(all_hits)})")
        time.sleep(RATE_LIMIT)

    return all_hits


# ── Parse Title ───────────────────────────────────────────────────────────────

# Matches: "Launch HN: Acme Corp (YC W25) – one liner description"
LAUNCH_PATTERN = re.compile(
    r"Launch HN:\s*(?P<name>.+?)\s*\(YC\s*(?P<batch>[WSF]\d{2})\)\s*[–—-]\s*(?P<one_liner>.+)",
    re.IGNORECASE
)

def parse_title(title: str) -> tuple[str, str, str] | None:
    """
    Returns (company_name, batch, one_liner) or None if no match.
    """
    m = LAUNCH_PATTERN.match(title.strip())
    if not m:
        return None
    return m.group("name").strip(), m.group("batch").upper(), m.group("one_liner").strip()


# ── Fetch Story Detail (for comment count + metadata) ────────────────────────

def fetch_story_detail(story_id: int) -> dict:
    url = f"{HN_BASE}/item/{story_id}.json"
    resp = requests.get(url, headers=HEADERS, timeout=10)
    resp.raise_for_status()
    return resp.json()


# ── Extract Signals from Post Text ───────────────────────────────────────────

def extract_urls(text: str) -> dict:
    """Pull GitHub and website URLs from post body."""
    github = re.search(r"https?://github\.com/[\w\-]+/[\w\-]+", text or "")
    website = re.search(r"https?://(?!github|ycombinator|news\.yc)[\w\.\-]+\.\w{2,}", text or "")
    return {
        "github_url" : github.group(0) if github else None,
        "website_url": website.group(0) if website else None,
    }

def extract_tags(one_liner: str) -> list[str]:
    """Naive keyword extraction from one-liner."""
    KEYWORDS = [
        "AI", "ML", "LLM", "B2B", "SaaS", "API", "developer", "fintech",
        "healthtech", "automation", "agent", "RAG", "open source", "enterprise",
        "marketplace", "devtools", "infrastructure", "data", "security",
        "robotics", "biotech", "climate", "edtech"
    ]
    return [kw for kw in KEYWORDS if kw.lower() in one_liner.lower()]


# ── Main Pipeline ─────────────────────────────────────────────────────────────

def run(
    pages: int = 5,
    batch_filter: str | None = None,   # e.g. "W25" — None means all
    min_points: int = 0,
    output_file: str = "yc_launches.json"
) -> list[YCCompany]:

    print(f"\n{'='*55}")
    print("  YC RADAR — Launch HN Scraper")
    print(f"{'='*55}\n")

    print("Fetching Launch HN posts from Algolia...")
    hits = fetch_launch_hn_posts(pages=pages)
    print(f"\nTotal raw posts fetched: {len(hits)}\n")

    companies = []

    for hit in hits:
        title    = hit.get("title", "")
        story_id = hit.get("objectID")
        points   = hit.get("points", 0)
        author   = hit.get("author", "")
        created  = hit.get("created_at", "")
        story_text = hit.get("story_text") or ""

        # ── Filter by points ──
        if points < min_points:
            continue

        # ── Parse title ──
        parsed = parse_title(title)
        if not parsed:
            continue  # not a valid Launch HN post

        name, batch, one_liner = parsed

        # ── Filter by batch ──
        if batch_filter and batch != batch_filter.upper():
            continue

        # ── Extract URLs and tags ──
        urls = extract_urls(story_text)
        tags = extract_tags(one_liner)

        # ── Fetch comment count (optional — adds latency) ──
        # Uncomment if you want live comment counts:
        # detail = fetch_story_detail(int(story_id))
        # num_comments = detail.get("descendants", 0)
        num_comments = hit.get("num_comments", 0)

        company = YCCompany(
            name          = name,
            batch         = batch,
            one_liner     = one_liner,
            hn_url        = f"https://news.ycombinator.com/item?id={story_id}",
            hn_points     = points,
            hn_comments   = num_comments,
            launched_at   = created,
            story_id      = int(story_id),
            founder_names = [author],   # post author is usually a founder
            tags          = tags,
            github_url    = urls["github_url"],
            website_url   = urls["website_url"],
        )
        companies.append(company)

    # ── Sort by traction (points + comments weighted) ──
    companies.sort(key=lambda c: c.hn_points + c.hn_comments * 2, reverse=True)

    # ── Print Summary ──
    print(f"{'─'*55}")
    print(f"  Parsed companies : {len(companies)}")
    if batch_filter:
        print(f"  Batch filter     : {batch_filter.upper()}")
    print(f"{'─'*55}\n")

    for i, c in enumerate(companies[:20], 1):
        print(f"{i:>3}. [{c.batch}] {c.name}")
        print(f"       {c.one_liner[:70]}")
        print(f"       ▲ {c.hn_points} pts  💬 {c.hn_comments}  🏷 {', '.join(c.tags) or 'none'}")
        print(f"       🔗 {c.hn_url}")
        if c.github_url:
            print(f"       GitHub: {c.github_url}")
        print()

    # ── Save to JSON ──
    with open(output_file, "w") as f:
        json.dump([asdict(c) for c in companies], f, indent=2)
    print(f"\n✓ Saved {len(companies)} companies to {output_file}")

    return companies


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    run(
        pages        = 10,          # 10 pages × 50 = up to 500 posts
        batch_filter = None,        # set to "W25" to filter one batch
        min_points   = 50,          # ignore low-traction posts
        output_file  = "yc_launches.json"
    )