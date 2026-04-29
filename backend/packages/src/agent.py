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
    return news_fetch(query)

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
    return json.dumps(formatted, indent=2)

news_agent = create_react_agent(
    model=model,
    tools=[news_fetch_google, startup_yc_news, github_trending_tool],
    prompt="You are a helpful assistant who fetches startup funding news.",
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