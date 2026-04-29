#!/usr/bin/env python3
"""
Fetch the 100 most trending GitHub repositories.

Uses the GitHub Search API (no auth required, but rate-limited to 10 req/min unauthenticated).
Set GITHUB_TOKEN env var for higher limits (30 req/min).

Usage:
    python github_trending.py                  # trending today (Python + JavaScript)
    python github_trending.py --language python
    python github_trending.py --since weekly
    python github_trending.py --output repos.json
    python github_trending.py --output repos.csv
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

import requests

GITHUB_API = "https://api.github.com/search/repositories"
PAGE_SIZE = 30  # GitHub max per page


def get_headers():
    token = os.getenv("GITHUB_TOKEN")
    headers = {"Accept": "application/vnd.github+json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    return headers


def date_since(period: str) -> str:
    """Return ISO date string for 'daily', 'weekly', or 'monthly' lookback."""
    delta = {"daily": 1, "weekly": 7, "monthly": 30}.get(period, 1)
    return (datetime.utcnow() - timedelta(days=delta)).strftime("%Y-%m-%d")


def fetch_page(query: str, page: int, retries: int = 3) -> dict:
    params = {
        "q": query,
        "sort": "stars",
        "order": "desc",
        "per_page": PAGE_SIZE,
        "page": page,
    }
    for attempt in range(1, retries + 1):
        resp = requests.get(GITHUB_API, headers=get_headers(), params=params, timeout=15)
        if resp.status_code == 200:
            return resp.json()
        if resp.status_code == 403:
            reset = int(resp.headers.get("X-RateLimit-Reset", time.time() + 60))
            wait = max(reset - time.time(), 1)
            print(f"  Rate limited. Waiting {wait:.0f}s …", file=sys.stderr)
            time.sleep(wait)
        elif resp.status_code == 422:
            print(f"  Bad query: {resp.json().get('message')}", file=sys.stderr)
            sys.exit(1)
        else:
            print(f"  HTTP {resp.status_code} on attempt {attempt}: {resp.text[:200]}", file=sys.stderr)
            time.sleep(2 ** attempt)
    resp.raise_for_status()


def fetch_trending(language: str | None, since: str, limit: int = 100) -> list[dict]:
    """Fetch up to `limit` trending repos from GitHub Search API."""
    date_str = date_since(since)
    query_parts = [f"created:>{date_str}", "stars:>0"]
    if language:
        query_parts.append(f"language:{language}")
    query = " ".join(query_parts)

    print(f"Query : {query}", file=sys.stderr)

    repos, page = [], 1
    while len(repos) < limit:
        print(f"  Fetching page {page} …", file=sys.stderr)
        data = fetch_page(query, page)
        items = data.get("items", [])
        if not items:
            break
        repos.extend(items)
        total_available = data.get("total_count", 0)
        if len(items) < PAGE_SIZE or len(repos) >= total_available:
            break
        page += 1
        time.sleep(0.5)  # be polite

    return repos[:limit]


def format_repo(rank: int, repo: dict) -> dict:
    return {
        "rank": rank,
        "name": repo["full_name"],
        "description": (repo.get("description") or "").strip(),
        "url": repo["html_url"],
        "stars": repo["stargazers_count"],
        "forks": repo["forks_count"],
        "language": repo.get("language") or "N/A",
        "topics": ", ".join(repo.get("topics", [])),
        "created_at": repo["created_at"][:10],
        "updated_at": repo["updated_at"][:10],
    }


def print_table(repos: list[dict]) -> None:
    header = f"{'#':>4}  {'Repository':<40}  {'Stars':>7}  {'Forks':>6}  {'Language':<15}  Description"
    print(header)
    print("-" * len(header))
    for r in repos:
        desc = r["description"][:55] + "…" if len(r["description"]) > 55 else r["description"]
        print(
            f"{r['rank']:>4}  {r['name']:<40}  {r['stars']:>7,}  {r['forks']:>6,}  {r['language']:<15}  {desc}"
        )


def save_json(repos: list[dict], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(repos, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(repos)} repos → {path}")


def save_csv(repos: list[dict], path: str) -> None:
    import csv

    if not repos:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=repos[0].keys())
        writer.writeheader()
        writer.writerows(repos)
    print(f"\nSaved {len(repos)} repos → {path}")




def main():
    parser = argparse.ArgumentParser(description="Fetch 100 trending GitHub repositories.")
    parser.add_argument("--language", "-l", default=None, help="Filter by programming language (e.g. python, rust)")
    parser.add_argument(
        "--since",
        "-s",
        choices=["daily", "weekly", "monthly"],
        default="weekly",
        help="Trending window (default: weekly)",
    )
    parser.add_argument("--limit", "-n", type=int, default=100, help="Number of repos (default: 100)")
    parser.add_argument("--output", "-o", default=None, help="Save to file (.json or .csv)")
    args = parser.parse_args()

    print(f"\n🔍 Fetching top {args.limit} trending repos "
          f"({args.since}, language={args.language or 'any'}) …\n", file=sys.stderr)

    raw_repos = fetch_trending(language=args.language, since=args.since, limit=args.limit)

    if not raw_repos:
        print("No repositories found. Try a wider time range or different language.", file=sys.stderr)
        sys.exit(1)

    formatted = [format_repo(i + 1, r) for i, r in enumerate(raw_repos)]

    print_table(formatted)
    print(f"\nTotal fetched: {len(formatted)}")

    if args.output:
        ext = args.output.rsplit(".", 1)[-1].lower()
        if ext == "json":
            save_json(formatted, args.output)
        elif ext == "csv":
            save_csv(formatted, args.output)
        else:
            print(f"Unknown extension '.{ext}'. Use .json or .csv.", file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()