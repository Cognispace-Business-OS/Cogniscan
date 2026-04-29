import asyncio
from twikit import Client, TooManyRequests
from datetime import datetime
import csv
import os
import random
from dotenv import load_dotenv

load_dotenv()

MINIMUM_TWEETS = 10
# ✅ More targeted — tech hiring
QUERY = '(hiring OR "we are hiring") (python OR AI OR developer) lang:en -filter:retweets'

username = os.getenv('TWITTER_USERNAME')
email = os.getenv('TWITTER_EMAIL')
password = os.getenv('TWITTER_PASSWORD')
client = Client(language='en-US')
client.http.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'x-twitter-active-user': 'yes',
    'x-twitter-auth-type': 'OAuth2Session',
    'x-twitter-client-language': 'en',
})

async def get_tweets(tweets):
    if tweets is None:
        print(f'{datetime.now()} - Getting tweets...')
        return await client.search_tweet(QUERY, product='Top')
    else:
        wait_time = random.randint(5, 10)
        print(f'{datetime.now()} - Getting next tweets after {wait_time} seconds...')
        await asyncio.sleep(wait_time)   # ✅ non-blocking sleep
        return await tweets.next()       # ✅ awaited

# --- MONKEY PATCH FOR TWIKIT SEARCH 404 ---
import twikit.client.gql
twikit.client.gql.Endpoint.SEARCH_TIMELINE = 'https://x.com/i/api/graphql/R0u1RWRf748KzyGBXvOYRA/SearchTimeline'

original_search = twikit.client.gql.GQLClient.search_timeline
async def patched_search_timeline(self, query, product, count, cursor):
    variables = {
        'rawQuery': query,
        'count': count,
        'querySource': 'typed_query',
        'product': product,
        'withGrokTranslatedBio': True
    }
    if cursor is not None:
        variables['cursor'] = cursor
    return await self.gql_post(twikit.client.gql.Endpoint.SEARCH_TIMELINE, variables, twikit.client.gql.FEATURES)

twikit.client.gql.GQLClient.search_timeline = patched_search_timeline
# ------------------------------------------

async def main():
    # ── Auth ──────────────────────────────────────────────────────
    if os.path.exists('cookies.json'):
        import json
        with open('cookies.json', 'r', encoding='utf-8') as f:
            cookies_data = json.load(f)
        
        # If it's a list (browser export), convert to dict
        if isinstance(cookies_data, list):
            cookies_dict = {c['name']: c['value'] for c in cookies_data}
            client.set_cookies(cookies_dict)
        else:
            client.load_cookies('cookies.json')
            
        # Explicitly set the CSRF token header (fixes empty 403 errors)
        ct0_token = client.http.cookies.get('ct0')
        if ct0_token:
            client.http.headers['x-csrf-token'] = ct0_token
            
        print("✅ Loaded cookies.")
    else:
        await client.login(auth_info_1=username, auth_info_2=email, password=password)
        client.save_cookies('cookies.json')
        print("✅ Logged in and saved cookies.")

    # ── CSV setup ─────────────────────────────────────────────────
    with open('tweets.csv', 'w', newline='', encoding='utf-8') as f:
        csv.writer(f).writerow(
            ['Tweet_count', 'Username', 'Text', 'Created At', 'Retweets', 'Likes']
        )

    # ── Scrape loop ───────────────────────────────────────────────
    tweet_count = 0
    tweets = None

    while tweet_count < MINIMUM_TWEETS:
        try:
            tweets = await get_tweets(tweets)
        except TooManyRequests as e:
            reset_time = datetime.fromtimestamp(e.rate_limit_reset)
            wait_secs = (reset_time - datetime.now()).total_seconds()
            print(f'{datetime.now()} - Rate limited. Waiting until {reset_time}')
            await asyncio.sleep(wait_secs)  # ✅ non-blocking
            continue
        except Exception as e:
            print(f'{datetime.now()} - Error: {e}')
            break

        if not tweets:
            print(f'{datetime.now()} - No more tweets found.')
            break

        with open('tweets.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            for tweet in tweets:
                tweet_count += 1
                writer.writerow([
                    tweet_count,
                    tweet.user.name,
                    tweet.text,
                    tweet.created_at,
                    tweet.retweet_count,
                    tweet.favorite_count
                ])

        print(f'{datetime.now()} - Got {tweet_count} tweets so far...')

    print(f'{datetime.now()} - Done! Total tweets saved: {tweet_count}')

if __name__ == '__main__':
    asyncio.run(main())