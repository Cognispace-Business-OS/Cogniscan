import requests
import datetime

def run_news_fetch(api_key: str, mode: str, query: str, language: str, sort_by: str, page_size: int, days: int) -> list:
    """
    Fetch news from NewsAPI.
    """
    base_url = "https://newsapi.org/v2"
    url = f"{base_url}/{mode}"
    
    # Calculate date range
    from_date = (datetime.datetime.now() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
    
    params = {
        "q": query,
        "language": language,
        "sortBy": sort_by,
        "pageSize": page_size,
        "from": from_date,
        "apiKey": api_key
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception(f"NewsAPI error: {response.status_code} {response.text}")
        
    data = response.json()
    return data.get("articles", [])
