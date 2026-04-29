import asyncio
import csv
from datetime import datetime
from playwright.async_api import async_playwright

QUERY = "hiring python"
MINIMUM_TWEETS = 20
OUTPUT_FILE = "tweets.csv"

async def scrape_tweets():
    tweet_data = []

    async with async_playwright() as p:
        # ── Launch real browser (headless=False so you can log in) ──
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context()
        page = await context.new_page()

        # ── Step 1: Log in manually ──────────────────────────────────
        await page.goto("https://x.com/login")
        print("🔐 Log in manually in the browser window...")
        # Wait until you're redirected to the home feed
        await page.wait_for_url("https://x.com/home", timeout=120000)
        print("✅ Logged in!")

        # ── Save session so you don't log in every time ──────────────
        await context.storage_state(path="session.json")

        # ── Step 2: Go to search ─────────────────────────────────────
        search_url = f"https://x.com/search?q={QUERY.replace(' ', '%20')}&src=typed_query&f=live"
        await page.goto(search_url)
        await page.wait_for_load_state("networkidle")
        await asyncio.sleep(3)

        print(f"🔍 Searching for: {QUERY}")

        seen_tweets = set()

        while len(tweet_data) < MINIMUM_TWEETS:
            # ── Grab all tweet articles on page ──────────────────────
            articles = await page.query_selector_all('article[data-testid="tweet"]')

            for article in articles:
                try:
                    # Username
                    user_el = await article.query_selector('[data-testid="User-Name"]')
                    username = await user_el.inner_text() if user_el else "Unknown"
                    username = username.split("\n")[0]

                    # Tweet text
                    text_el = await article.query_selector('[data-testid="tweetText"]')
                    text = await text_el.inner_text() if text_el else ""

                    # Timestamp
                    time_el = await article.query_selector("time")
                    created_at = await time_el.get_attribute("datetime") if time_el else ""

                    # Unique key to avoid duplicates
                    key = f"{username}:{created_at}"
                    if key in seen_tweets:
                        continue
                    seen_tweets.add(key)

                    tweet_data.append([len(tweet_data) + 1, username, text, created_at])
                    print(f"👤 {username} | 🕐 {created_at}")
                    print(f"🐦 {text[:80]}...")
                    print("-" * 40)

                except Exception as e:
                    print(f"⚠️ Error parsing tweet: {e}")
                    continue

            print(f"📊 Collected {len(tweet_data)} tweets so far...")

            if len(tweet_data) >= MINIMUM_TWEETS:
                break

            # ── Scroll down to load more ──────────────────────────────
            await page.evaluate("window.scrollBy(0, 1500)")
            await asyncio.sleep(3)

        await browser.close()

    # ── Save to CSV ───────────────────────────────────────────────────
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Tweet_count', 'Username', 'Text', 'Created At'])
        writer.writerows(tweet_data)

    print(f"\n✅ Done! Saved {len(tweet_data)} tweets to {OUTPUT_FILE}")

async def main():
    # ── Reuse saved session if available ─────────────────────────────
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)

        if __import__('os').path.exists("session.json"):
            context = await browser.new_context(storage_state="session.json")
            page = await context.new_page()
            await page.goto("https://x.com/home")
            await asyncio.sleep(2)

            # Check if still logged in
            if "login" in page.url:
                print("⚠️ Session expired, logging in again...")
                await context.close()
                await browser.close()
                await scrape_tweets()
                return
        else:
            await browser.close()
            await scrape_tweets()
            return

        print("✅ Session restored!")
        search_url = f"https://x.com/search?q={QUERY.replace(' ', '%20')}&src=typed_query&f=live"
        await page.goto(search_url)
        await page.wait_for_load_state("networkidle")