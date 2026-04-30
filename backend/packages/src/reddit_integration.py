"""
Reddit Scraper — No API Key Required
======================================
Uses Reddit's built-in JSON endpoints (just append .json to any Reddit URL).

Install dependencies:
    pip install requests pandas

Usage:
    python reddit_scraper.py
"""

import requests
import pandas as pd
import time
from datetime import datetime

# ── CONFIG ────────────────────────────────────────────────────────────────────
SUBREDDIT      = "employeesOfOracle"     # subreddit to scrape
CATEGORY       = "new"        # hot | new | top | rising
POST_LIMIT     = 100        # total posts to collect (max ~1000 via pagination)
FETCH_COMMENTS = True         # fetch top comments for each post?
COMMENT_LIMIT  = 5            # how many top-level comments per post
OUTPUT_FILE    = "reddit_posts.json"
DELAY          = 1.5          # seconds between requests (be polite!)
KEYWORD_FILTER = "intern"     # only include posts containing this keyword
# ─────────────────────────────────────────────────────────────────────────────

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; reddit-scraper/1.0)"
}


def fetch_posts(subreddit, category, limit):
    """Fetch posts from a subreddit using Reddit's public JSON API."""
    posts = []
    after = None
    base_url = f"https://www.reddit.com/r/{subreddit}/{category}.json"

    print(f"[*] Scraping r/{subreddit} ({category}) — target: {limit} posts")

    while len(posts) < limit:
        params = {"limit": min(100, limit - len(posts))}
        if after:
            params["after"] = after

        try:
            response = requests.get(base_url, headers=HEADERS, params=params, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            print(f"[!] Request failed: {e}")
            break

        data = response.json()
        children = data.get("data", {}).get("children", [])

        if not children:
            print("[*] No more posts found.")
            break

        for child in children:
            p = child["data"]
            posts.append({
                "id":           p.get("id"),
                "title":        p.get("title"),
                "author":       p.get("author"),
                "score":        p.get("score"),
                "upvote_ratio": p.get("upvote_ratio"),
                "num_comments": p.get("num_comments"),
                "url":          p.get("url"),
                "permalink":    "https://reddit.com" + p.get("permalink", ""),
                "selftext":     p.get("selftext", "").strip(),
                "flair":        p.get("link_flair_text"),
                "is_video":     p.get("is_video"),
                "created_utc":  datetime.utcfromtimestamp(
                                    p.get("created_utc", 0)
                                ).strftime("%Y-%m-%d %H:%M:%S"),
            })

        after = data["data"].get("after")
        print(f"    Collected {len(posts)} posts so far...")

        if not after:
            break

        time.sleep(DELAY)

    return posts


def fetch_comments(permalink, limit):
    """Fetch top-level comments for a single post."""
    url = permalink.rstrip("/") + ".json"
    try:
        response = requests.get(url, headers=HEADERS, params={"limit": limit}, timeout=10)
        response.raise_for_status()
        data = response.json()
        comments_data = data[1]["data"]["children"]
        comments = []
        for c in comments_data[:limit]:
            if c["kind"] == "t1":
                body = c["data"].get("body", "").strip()
                if body and body != "[deleted]":
                    comments.append(body)
        return " | ".join(comments)
    except Exception:
        return ""

def filter_keyword(posts, keyword):
    return [post for post in posts if keyword.lower() in post["title"].lower() or keyword.lower() in post["selftext"].lower()]

def main():
    posts = fetch_posts(SUBREDDIT, CATEGORY, POST_LIMIT)

    posts = filter_keyword(posts, KEYWORD_FILTER)
    print(f"[*] Filtered posts by keyword: {KEYWORD_FILTER}")
    print(f"    Found {len(posts)} posts.")
    if not posts:
        print("[!] No posts collected. Exiting.")
        return

    if FETCH_COMMENTS:
        print(f"\n[*] Fetching top {COMMENT_LIMIT} comments per post...")
        for i, post in enumerate(posts):
            post["top_comments"] = fetch_comments(post["permalink"], COMMENT_LIMIT)
            if (i + 1) % 10 == 0:
                print(f"    Comments fetched for {i + 1}/{len(posts)} posts...")
            time.sleep(DELAY)

    df = pd.DataFrame(posts)
    df.to_json(OUTPUT_FILE, orient="records", indent=4, force_ascii=False)
    print(f"\n[✓] Saved {len(df)} posts to '{OUTPUT_FILE}'")
    print(df[["title", "score", "num_comments", "author"]].head(10).to_string(index=False))


if __name__ == "__main__":
    main()