from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langchain_mistralai import ChatMistralAI   # ✅ real Mistral with tool calling
from script3 import news_fetch
from script2 import yc_funding_run
from dotenv import load_dotenv
import os
import logging
import json
from github_trending import fetch_trending, format_repo
from models import Article, Startup, GithubRepo
from database import Session, Base, engine
from utility import extract_startup_names

logging.basicConfig(level=logging.INFO)

load_dotenv()

model = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_KEY"),
    temperature=0.3,
)
@tool
def news_fetch_google(query: str) -> str:
    """Fetches news articles from Google based on the given query."""
    logging.info(f"Fetching news for: {query}")
    articles = news_fetch(query)  # fix: was `aricles`

    session = Session()
    try:
        for article in articles:
            new_article = Article(
                title=article["title"],
                link=article["link"],
                source=article["source"],
                published=article["published"],
                text=article["text"],
            )
            session.add(new_article)
            session.flush()  # fix: flush so new_article.id is assigned before use

            orgs = extract_startup_names(article["text"])
            for org in orgs:
                # fix: check by org name + article_id (now valid after flush)
                startup = session.query(Startup).filter_by(
                    orgs=org, article_id=new_article.id
                ).first()

                if startup is None:
                    # fix: create and link the startup when it doesn't exist
                    startup = Startup(orgs=org, article_id=new_article.id)
                    session.add(startup)

                # if it already exists, no action needed — link is already stored

        session.commit()

    except Exception:
        session.rollback()  # fix: rollback on error to avoid partial writes
        raise

    finally:
        session.close()  # fix: always release the session

    return json.dumps(articles, indent=2)

@tool
def startup_yc_news(stage_filter: str, min_round_usd: float) -> str:
    """Fetches YC startup funding news filtered by stage and minimum round size."""
    logging.info(f"Fetching YC news: stage={stage_filter}, min_round={min_round_usd}")
    return yc_funding_run(stage_filter, min_round_usd)
@tool 
def github_trending_tool(language: str, since: str) -> str:
    """Fetches trending GitHub repositories. 'since' can be 'daily', 'weekly', or 'monthly'."""
    logging.info(f"Fetching trending GitHub repos: language={language}, since={since}")
    # Use a default 'weekly' if since is not provided or invalid
    if since not in ["daily", "weekly", "monthly"]:
        since = "weekly"
    repos = fetch_trending(language, since, limit=10)
    formatted = [format_repo(i + 1, r) for i, r in enumerate(repos)]
    session = Session()
    try:
        for formatted_repo in formatted:
            new_repo = GithubRepo(
                title=formatted_repo["title"],
                url=formatted_repo["url"],
                stars=formatted_repo["stars"],
                language=formatted_repo["language"],
                description=formatted_repo["description"],
            )
            session.add(new_repo)
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
    return json.dumps(formatted, indent=2)


@tool
def fetch_news_from_newsorg(query: str) -> str:
    """
    Public API wrapper for fetching news via NewsAPI (newsapi.org).

    Parameters (from query)
    ----------
    query: str - Keywords to search for.

    Internal Settings
    -------------------
    mode       : 'everything' (searches all articles).
    category   : None (no category filter).
    country    : 'us' (United States).
    source     : None (no specific source).
    language   : 'en' (English).
    sort_by    : 'publishedAt' (newest first).
    page_size  : 10 (number of articles).
    days       : 7  (look back 7 days).
    api_key    : read from environment variable NEWSORG_API_KEY.

    Returns
    -------
    str
        JSON string containing up to 10 recent articles matching the query.
        Each article includes title, link, source, published date and snippet.

    Raises
    ------
    ValueError: If NEWSORG_API_KEY is missing.
    RuntimeError: If the NewsAPI request fails (network or API error).
    """

    try:
        # 1. API Key from environment
        api_key = os.getenv("NEWSORG_API_KEY")
        if not api_key:
            raise ValueError("NEWSORG_API_KEY not set in environment.")

        # 2. Required parameters for the query
        if not query:
            raise ValueError("Query is required for fetching news.")

        logging.info(f"Fetching news for query: {query}")

        # 3. Call the public API wrapper (no sys.exit)
        articles = run_news_fetch(
            api_key=api_key,
            mode="everything",
            query=query,
            category=None,
            country="us",
            source=None,
            language="en",
            sort_by="publishedAt",
            page_size=10,
            days=7,
        )
        session = Session()
        for article in articles:
            new_article = Article(
                title=article["title"],
                link=article["url"],
                source=article["source"]["name"],
                published=article["publishedAt"],
                text=article["description"],
            )
            session.add(new_article)
        session.commit()
        session.close()

        return json.dumps(articles, indent=2)

    except Exception as e:
        logging.error(f"Error fetching news from NewsAPI: {str(e)}")
        return json.dumps({"error": str(e)})



news_agent = create_react_agent(
    model=model,
    tools=[news_fetch_google, startup_yc_news, fetch_news_from_newsorg],
    prompt="You are a helpful assistant who fetches startup funding news.",
)
github_repo_agent = create_react_agent(
    model=model,
    tools=[github_trending_tool],
    prompt="You are a helpful assistant who fetches trending github repos.",
)

def main(): 
    query = "best github repos of this month"
    print("Fetching news for:", query)

    result = news_agent.invoke({
        "messages": [HumanMessage(content=query)]
    })

    print(result["messages"][-1].content)

if __name__ == "__main__":
    main()