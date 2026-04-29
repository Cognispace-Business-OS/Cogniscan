import asyncio
from twikit import Client
from twikit.errors import TwitterException
import os
import json
from dotenv import load_dotenv

load_dotenv()

USERNAME = os.getenv("TWITTER_USERNAME")
EMAIL = os.getenv("TWITTER_EMAIL")
PASSWORD = os.getenv("TWITTER_PASSWORD")
PROXY_URL = os.getenv("PROXY_URL3")  # ✅ Load from .env, not hardcoded

COOKIES_FILE = "cookies.json"

client = Client('en-US', proxy=PROXY_URL)

async def fresh_login():
    print("Logging in fresh...")
    await client.login(
        auth_info_1=USERNAME,
        auth_info_2=EMAIL,
        password=PASSWORD,
    )
    client.save_cookies(COOKIES_FILE)
    print("Logged in and cookies saved.")

async def login():
    if os.path.exists(COOKIES_FILE):
        try:
            with open(COOKIES_FILE, 'r', encoding='utf-8') as f:
                cookies_data = json.load(f)

            if isinstance(cookies_data, list):
                cookies_dict = {c['name']: c['value'] for c in cookies_data}
                client.set_cookies(cookies_dict)
            else:
                client.load_cookies(COOKIES_FILE)

            await client.user()
            print("Session valid, loaded from cookies.")

        except Exception as e:
            print(f"Cookie session invalid ({e}), re-logging in...")
            if os.path.exists(COOKIES_FILE):
                os.remove(COOKIES_FILE)
            await fresh_login()
    else:
        await fresh_login()

async def search_tweets(query: str):
    await login()
    await asyncio.sleep(1)

    try:
        tweets = await client.search_tweet(query, 'Latest', 20)
    except TwitterException as e:
        print(f"Twitter error: {e}")
        return
    except Exception as e:
        print(f"Unexpected error: {e}")
        return

    if not tweets:
        print("No tweets found.")
        return

    for tweet in tweets:
        print(f"👤 {tweet.user.name}")
        print(f"🐦 {tweet.text}")
        print(f"🕐 {tweet.created_at}")
        print("-" * 40)

def main():
    asyncio.run(search_tweets("AI or Hiring"))

if __name__ == "__main__":
    main()