"""
Google News Scraper
-------------------
Two methods:
  1. RSS Feed (recommended) - reliable, no bot detection issues
  2. Direct HTML scraping - may break if Google changes their layout

Usage:
  pip install requests beautifulsoup4 feedparser

  python scrape_google_news.py
  python scrape_google_news.py --query "artificial intelligence" --method rss
  python scrape_google_news.py --query "python" --method html --limit 10
"""

import argparse
import json
import sys
from datetime import datetime

try:
    import requests
    from bs4 import BeautifulSoup
except ImportError:
    print("Missing dependencies. Run: pip install requests beautifulsoup4")
    sys.exit(1)


# ─────────────────────────────────────────────
# Method 1: RSS Feed (recommended)
# ─────────────────────────────────────────────

def scrape_via_rss(query: str = "", topic: str = "WORLD", limit: int = 20) -> list[dict]:
    """
    Scrape Google News via their public RSS feed.

    Args:
        query:  Search term (e.g. "python programming"). If empty, uses topic.
        topic:  Top-level topic when no query given.
                Options: WORLD, NATION, BUSINESS, TECHNOLOGY, ENTERTAINMENT,
                         SPORTS, SCIENCE, HEALTH
        limit:  Max number of articles to return.

    Returns:
        List of article dicts with keys: title, link, source, published, summary
    """
    if query:
        url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    else:
        url = f"https://news.google.com/rss/headlines/section/topic/{topic}?hl=en-US&gl=US&ceid=US:en"

    headers = {"User-Agent": "Mozilla/5.0 (compatible; NewsScraper/1.0)"}
    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.content, "xml")
    items = soup.find_all("item")[:limit]

    articles = []
    for item in items:
        title_tag = item.find("title")
        link_tag = item.find("link")
        pub_tag = item.find("pubDate")
        desc_tag = item.find("description")
        source_tag = item.find("source")

        # Strip HTML from description/summary
        summary = ""
        if desc_tag and desc_tag.text:
            summary_soup = BeautifulSoup(desc_tag.text, "html.parser")
            summary = summary_soup.get_text(separator=" ").strip()

        articles.append({
            "title": title_tag.text.strip() if title_tag else "",
            "link": link_tag.text.strip() if link_tag and link_tag.text else "",
            "source": source_tag.text.strip() if source_tag else "",
            "published": pub_tag.text.strip() if pub_tag else "",
            "summary": summary[:300] + "..." if len(summary) > 300 else summary,
        })

    return articles


# ─────────────────────────────────────────────
# Method 2: Direct HTML Scraping
# ─────────────────────────────────────────────

def scrape_via_html(query: str = "", limit: int = 20) -> list[dict]:
    """
    Scrape Google News search results directly from HTML.
    Note: Google may block this or change their HTML structure at any time.
    The RSS method is more stable. Use this as a fallback.

    Args:
        query: Search term. If empty, scrapes the Google News homepage.
        limit: Max number of articles to return.

    Returns:
        List of article dicts.
    """
    if query:
        url = f"https://news.google.com/search?q={requests.utils.quote(query)}&hl=en-US&gl=US&ceid=US:en"
    else:
        url = "https://news.google.com/?hl=en-US&gl=US&ceid=US:en"

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }

    response = requests.get(url, headers=headers, timeout=10)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    articles = []

    # Google News uses article tags with specific structure
    article_tags = soup.find_all("article")[:limit]

    for article in article_tags:
        # Title is usually in an <a> with class containing 'title' or inside <h3>/<h4>
        title_el = article.find(["h3", "h4"])
        link_el = article.find("a", href=True)
        time_el = article.find("time")
        source_el = article.find("a", {"data-n-tid": True})

        title = title_el.get_text(strip=True) if title_el else ""
        if not title:
            continue  # Skip empty articles

        href = link_el["href"] if link_el else ""
        # Google News hrefs are relative like ./articles/... — fix them
        if href.startswith("./"):
            href = "https://news.google.com/" + href[2:]
        elif href.startswith("/"):
            href = "https://news.google.com" + href

        articles.append({
            "title": title,
            "link": href,
            "source": source_el.get_text(strip=True) if source_el else "",
            "published": time_el.get("datetime", "") if time_el else "",
            "summary": "",  # HTML scraping doesn't expose summaries easily
        })

    return articles


# ─────────────────────────────────────────────
# Output helpers
# ─────────────────────────────────────────────

def print_articles(articles: list[dict], fmt: str = "text"):
    if fmt == "json":
        print(json.dumps(articles, indent=2, ensure_ascii=False))
        return

    if not articles:
        print("No articles found.")
        return

    print(f"\n{'═' * 60}")
    print(f"  Found {len(articles)} articles  [{datetime.now().strftime('%Y-%m-%d %H:%M')}]")
    print(f"{'═' * 60}\n")

    for i, a in enumerate(articles, 1):
        print(f"[{i}] {a['title']}")
        if a.get("source"):
            print(f"    Source   : {a['source']}")
        if a.get("published"):
            print(f"    Published: {a['published']}")
        if a.get("summary"):
            print(f"    Summary  : {a['summary']}")
        print(f"    URL      : {a['link']}")
        print()


def save_to_file(articles: list[dict], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(articles, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(articles)} articles to {path}")


# ─────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────

# def main():
#     parser = argparse.ArgumentParser(
#         description="Scrape Google News articles",
#         formatter_class=argparse.RawDescriptionHelpFormatter,
#         epilog="""
# Examples:
#   python scrape_google_news.py
#   python scrape_google_news.py --query "climate change"
#   python scrape_google_news.py --topic TECHNOLOGY --limit 15
#   python scrape_google_news.py --query "AI" --format json --output results.json
#   python scrape_google_news.py --query "bitcoin" --method html
#         """
#     )
#     parser.add_argument("--query", "-q", default="", help="Search query (e.g. 'python AI')")
#     parser.add_argument("--topic", "-t", default="WORLD",
#                         choices=["WORLD","NATION","BUSINESS","TECHNOLOGY",
#                                  "ENTERTAINMENT","SPORTS","SCIENCE","HEALTH"],
#                         help="Topic (used when no --query given, RSS only)")
#     parser.add_argument("--method", "-m", default="rss", choices=["rss", "html"],
#                         help="Scraping method: rss (default, stable) or html (fragile)")
#     parser.add_argument("--limit", "-l", type=int, default=20, help="Max articles to fetch")
#     parser.add_argument("--format", "-f", default="text", choices=["text", "json"],
#                         help="Output format")
#     parser.add_argument("--output", "-o", default="", help="Save results to this JSON file")

#     args = parser.parse_args()

#     print(f"Fetching news via {args.method.upper()} "
#           + (f"for query: '{args.query}'" if args.query else f"for topic: {args.topic}") + "...")

#     try:
#         if args.method == "rss":
#             articles = scrape_via_rss(query=args.query, topic=args.topic, limit=args.limit)
#         else:
#             articles = scrape_via_html(query=args.query, limit=args.limit)
#     except requests.RequestException as e:
#         print(f"Network error: {e}", file=sys.stderr)
#         sys.exit(1)

#     print_articles(articles, fmt=args.format)
#     args.output = "Startup_Funding_News.json"

#     save_to_file(articles, args.output)

def news_fetch(query: str = "", method: str = "rss", limit: int = 20):
    topics = ["BUSINESS", "TECHNOLOGY", "SCIENCE", "HEALTH", "WORLD"]
    all_articles = []
    for topic in topics:
        try:
            if method == "rss":
                articles = scrape_via_rss(query=query, topic=topic, limit=limit)
            else:
                articles = scrape_via_html(query=query, limit=limit)
        except requests.RequestException as e:
            print(f"Network error for topic {topic}: {e}", file=sys.stderr)
            continue
        all_articles.extend(articles)
    # Deduplicate by title
    seen = set()
    unique = []
    for a in all_articles:
        if a["title"] not in seen:
            seen.add(a["title"])
            unique.append(a)
    return unique

def main():
    results=news_fetch("Startup Funding")
    save_to_file(results, "Startup_Funding_News.json")
if __name__ == "__main__":
    main()