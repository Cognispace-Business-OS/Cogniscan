"""
Funding Radar — Startup Funding Tracker
Tracks funding announcements from:
  - Hacker News (Launch HN + Show HN + funding stories)
  - Algolia HN search for seed/series terms
Extracts: company, round size, stage, investors, date, traction signals
"""

import requests
import re
import time
import json
from datetime import datetime
from dataclasses import dataclass, asdict, field


# ── Config ────────────────────────────────────────────────────────────────────

ALGOLIA_BASE = "https://hn.algolia.com/api/v1"
HN_BASE      = "https://hacker-news.firebaseio.com/v0"
HEADERS      = {"User-Agent": "FundingRadar/1.0 (research project)"}
RATE_LIMIT   = 0.5


# ── Data Model ────────────────────────────────────────────────────────────────

@dataclass
class FundingEvent:
    # Company
    company_name: str
    one_liner: str
    website_url: str | None

    # Funding
    round_stage: str            # seed, series_a, series_b, pre_seed, etc.
    round_size: str | None      # e.g. "$4M", "$12M" — raw string from text
    round_size_usd: float | None  # parsed numeric value in USD millions
    investors: list[str]        # extracted investor names

    # Source
    source: str                 # "launch_hn", "hn_funding", "show_hn"
    hn_url: str
    hn_points: int
    hn_comments: int
    announced_at: str
    story_id: int

    # Signals
    founder_hn_handle: str
    tags: list[str]
    github_url: str | None = None

    # Enrichment (filled later)
    github_stars: int | None = None
    github_repo: str | None = None


# ── Regex Patterns ────────────────────────────────────────────────────────────

# Launch HN: Acme (YC W25) – one liner
LAUNCH_HN_PATTERN = re.compile(
    r"Launch HN:\s*(?P<name>.+?)\s*\(YC\s*(?P<batch>[WSF]\d{2})\)\s*[–—\-]\s*(?P<one_liner>.+)",
    re.IGNORECASE
)

# "$4M seed", "$12M Series A", "raised $2.5M"
ROUND_SIZE_PATTERN = re.compile(
    r"\$(?P<amount>[\d,\.]+)\s*(?P<unit>K|M|B)?",
    re.IGNORECASE
)

STAGE_KEYWORDS = {
    "pre_seed"  : ["pre-seed", "pre seed", "preseed"],
    "seed"      : ["seed round", "seed funding", "seed stage", "(yc "],
    "series_a"  : ["series a", "series-a"],
    "series_b"  : ["series b", "series-b"],
    "series_c"  : ["series c", "series-c"],
    "series_d"  : ["series d", "series-d"],
    "grant"     : ["grant", "non-dilutive"],
    "ipo"       : ["ipo", "went public", "listed on"],
    "acquisition": ["acquired", "acquisition", "bought by"],
}

INVESTOR_PATTERN = re.compile(
    r"(?:led by|backed by|investors include|from|with participation from)\s+([A-Z][^.,\n]{3,50})",
    re.IGNORECASE
)

TECH_TAGS = [
    "AI", "LLM", "ML", "B2B", "SaaS", "API", "developer tools", "fintech",
    "healthtech", "automation", "agent", "RAG", "open source", "enterprise",
    "marketplace", "devtools", "infrastructure", "security", "robotics",
    "biotech", "climate", "edtech", "crypto", "web3", "deeptech",
]


# ── Parsing Helpers ───────────────────────────────────────────────────────────

def parse_round_size(text: str) -> tuple[str | None, float | None]:
    """Returns (raw_string, usd_millions)"""
    m = ROUND_SIZE_PATTERN.search(text or "")
    if not m:
        return None, None

    raw    = m.group(0)
    amount = float(m.group("amount").replace(",", ""))
    unit   = (m.group("unit") or "M").upper()

    usd_millions = {
        "K": amount / 1000,
        "M": amount,
        "B": amount * 1000,
    }.get(unit, amount)

    return raw, usd_millions


def detect_stage(text: str) -> str:
    text_lower = text.lower()
    for stage, keywords in STAGE_KEYWORDS.items():
        if any(kw in text_lower for kw in keywords):
            return stage
    return "unknown"


def extract_investors(text: str) -> list[str]:
    matches = INVESTOR_PATTERN.findall(text or "")
    # Clean up — remove trailing punctuation and very long matches
    cleaned = []
    for m in matches:
        m = m.strip().rstrip(".,;:")
        if len(m) < 60:
            cleaned.append(m)
    return cleaned[:5]  # cap at 5


def extract_urls(text: str) -> dict:
    github  = re.search(r"https?://github\.com/[\w\-]+(?:/[\w\-]+)?", text or "")
    website = re.search(
        r"https?://(?!github|ycombinator|news\.yc|twitter|linkedin)[\w\.\-]+\.\w{2,6}(?:/[\w\-]*)?",
        text or ""
    )
    return {
        "github_url" : github.group(0) if github else None,
        "website_url": website.group(0) if website else None,
    }


def extract_tags(text: str) -> list[str]:
    return [t for t in TECH_TAGS if t.lower() in (text or "").lower()]


# ── Fetch Functions ───────────────────────────────────────────────────────────

def algolia_search(query: str, pages: int = 5, per_page: int = 50) -> list[dict]:
    hits = []
    for page in range(pages):
        resp = requests.get(
            f"{ALGOLIA_BASE}/search",
            params={"query": query, "tags": "story", "hitsPerPage": per_page, "page": page},
            headers=HEADERS, timeout=10
        )
        resp.raise_for_status()
        data  = resp.json()
        batch = data.get("hits", [])
        if not batch:
            break
        hits.extend(batch)
        print(f"    [{query}] page {page+1}: +{len(batch)} (total {len(hits)})")
        time.sleep(RATE_LIMIT)
    return hits


def fetch_all_hits(pages_per_query: int = 5) -> list[dict]:
    """
    Pull from multiple HN search queries to maximize coverage.
    Deduplicates by objectID.
    """
    QUERIES = [
        "Launch HN",          # YC companies
        "raises seed funding",
        "announces Series A",
        "raises Series B",
        "seed round startup",
        "YC W25",
        "YC S24",
        "funded startup",
    ]

    seen = set()
    all_hits = []

    for q in QUERIES:
        print(f"\n  Searching: '{q}'")
        hits = algolia_search(q, pages=pages_per_query)
        for h in hits:
            oid = h.get("objectID")
            date = h.get("created_at")
            year_now= str(datetime.now().year)

            if oid and oid not in seen and  date.startswith(year_now):
                seen.add(oid)
                all_hits.append(h)

    print(f"\n  Total unique stories: {len(all_hits)}")
    return all_hits


# ── Hit → FundingEvent ────────────────────────────────────────────────────────

def hit_to_funding_event(hit: dict) -> FundingEvent | None:
    title      = hit.get("title", "")
    story_id   = hit.get("objectID", "0")
    points     = hit.get("points") or 0
    comments   = hit.get("num_comments") or 0
    author     = hit.get("author", "")
    created    = hit.get("created_at", "")

    body       = hit.get("story_text") or ""
    full_text  = f"{title} {body}"

    # ── Determine source type ──
    if re.match(r"Launch HN:", title, re.I):
        source = "launch_hn"
        lm = LAUNCH_HN_PATTERN.match(title.strip())
        if lm:
            company_name = lm.group("name").strip()
            one_liner    = lm.group("one_liner").strip()
        else:
            company_name = title.replace("Launch HN:", "").split("–")[0].strip()
            one_liner    = ""
    elif re.match(r"Show HN:", title, re.I):
        source       = "show_hn"
        company_name = title.replace("Show HN:", "").split("–")[0].strip()
        one_liner    = title.split("–")[-1].strip() if "–" in title else ""
    else:
        source       = "hn_funding"
        company_name = title
        one_liner    = ""

    # ── Skip if no funding signal at all ──
    has_funding_signal = any(kw in full_text.lower() for kw in [
        "raised", "raises", "funding", "seed", "series", "invested", "yc", "backed"
    ])
    if not has_funding_signal:
        return None

    # ── Extract funding details ──
    round_size_raw, round_size_usd = parse_round_size(full_text)
    stage     = detect_stage(full_text)
    investors = extract_investors(full_text)
    urls      = extract_urls(body)
    tags      = extract_tags(full_text)

    return FundingEvent(
        company_name    = company_name,
        one_liner       = one_liner,
        website_url     = urls["website_url"],
        round_stage     = stage,
        round_size      = round_size_raw,
        round_size_usd  = round_size_usd,
        investors       = investors,
        source          = source,
        hn_url          = f"https://news.ycombinator.com/item?id={story_id}",
        hn_points       = points,
        hn_comments     = comments,
        announced_at    = created,
        story_id        = int(story_id),
        founder_hn_handle = author,
        tags            = tags,
        github_url      = urls["github_url"],
    )


# ── GitHub Enrichment (optional) ─────────────────────────────────────────────

def enrich_github(event: FundingEvent) -> FundingEvent:
    """
    If a GitHub URL was found, fetch star count.
    Uses unauthenticated API — 60 req/hr limit.
    Pass a token via headers for 5000 req/hr.
    """
    if not event.github_url:
        return event

    # Extract owner/repo from URL
    m = re.search(r"github\.com/([\w\-]+/[\w\-]+)", event.github_url)
    if not m:
        return event

    repo_path = m.group(1)
    try:
        resp = requests.get(
            f"https://api.github.com/repos/{repo_path}",
            headers={**HEADERS, "Accept": "application/vnd.github+json"},
            timeout=8
        )
        if resp.status_code == 200:
            data = resp.json()
            event.github_stars = data.get("stargazers_count")
            event.github_repo  = repo_path
    except Exception:
        pass

    time.sleep(0.3)
    return event


# ── Main ──────────────────────────────────────────────────────────────────────

def yc_funding_run(
    pages_per_query : int  = 3,
    min_points      : int  = 10,
    stage_filter    : str | None = None,   # "seed", "series_a", etc.
    min_round_usd   : float | None = None, # e.g. 1.0 = $1M minimum
    enrich_gh       : bool = False,        # set True to fetch GitHub stars
    output_file     : str  = None
) -> list[FundingEvent]:

    print(f"\n{'='*58}")
    print("  FUNDING RADAR — Startup Funding Tracker")
    print(f"{'='*58}")

    hits   = fetch_all_hits(pages_per_query)
    events = []

    for hit in hits:
        try:
            ev = hit_to_funding_event(hit)
        except Exception as e:
            continue

        if ev is None:
            continue
        if ev.hn_points < min_points:
            continue
        if stage_filter and ev.round_stage != stage_filter:
            continue
        if min_round_usd and (ev.round_size_usd or 0) < min_round_usd:
            continue

        if enrich_gh:
            ev = enrich_github(ev)

        events.append(ev)

    # ── Sort: biggest rounds first, then by HN traction ──
    events.sort(
        key=lambda e: (e.round_size_usd or 0, e.hn_points + e.hn_comments * 2),
        reverse=True
    )

    # ── Deduplicate by company name (keep highest traction) ──
    seen_companies: dict[str, FundingEvent] = {}
    for ev in events:
        key = ev.company_name.lower().strip()
        if key not in seen_companies:
            seen_companies[key] = ev
    events = list(seen_companies.values())

    # ── Print ──
    print(f"\n{'─'*58}")
    print(f"  Funding events found : {len(events)}")
    print(f"{'─'*58}\n")

    STAGE_EMOJI = {
        "pre_seed": "🌱", "seed": "🌿", "series_a": "🚀",
        "series_b": "💰", "series_c": "🏦", "series_d": "🏛",
        "ipo": "📈", "acquisition": "🤝", "grant": "🎓", "unknown": "❓"
    }

    for i, ev in enumerate(events[:25], 1):
        emoji = STAGE_EMOJI.get(ev.round_stage, "❓")
        size  = f" {ev.round_size}" if ev.round_size else ""
        print(f"{i:>3}. {emoji} {ev.company_name}{size} [{ev.round_stage.upper()}]")
        if ev.one_liner:
            print(f"       {ev.one_liner[:75]}")
        if ev.investors:
            print(f"       👥 {', '.join(ev.investors[:3])}")
        print(f"       ▲ {ev.hn_points} pts  💬 {ev.hn_comments}", end="")
        if ev.github_stars is not None:
            print(f"  ⭐ {ev.github_stars:,} stars", end="")
        print(f"\n       🔗 {ev.hn_url}")
        if ev.tags:
            print(f"       🏷  {', '.join(ev.tags[:5])}")
        print()

    # ── Save ──
    if output_file:
        with open(output_file, "w") as f:
            json.dump([asdict(e) for e in events], f, indent=2)
        print(f"\n✓ Saved {len(events)} funding events → {output_file}")

    # ── Stage breakdown ──
    from collections import Counter
    stage_counts = Counter(e.round_stage for e in events)
    print("\n  Stage breakdown:")
    for stage, count in stage_counts.most_common():
        print(f"    {STAGE_EMOJI.get(stage,'❓')} {stage:<12} {count}")

    return events


# ── Entry Point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    yc_funding_run(
        pages_per_query = 3,       # pages per search query (3×8 queries = ~1200 stories)
        min_points      = 10,      # filter noise
        stage_filter    = None,    # None = all stages, or "seed", "series_a" etc.
        min_round_usd   = None,    # None = all sizes, or 1.0 = $1M+ only
        enrich_gh       = False,   # set True to also fetch GitHub star counts
        output_file     = "funding_radar.json"
    )