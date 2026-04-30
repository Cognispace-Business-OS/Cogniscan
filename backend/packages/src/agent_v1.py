"""
super_agent.py — LangGraph multi-agent pipeline

Flow:
  resume_node
      │
      ▼
  [news_node ║ github_node]  ← parallel
      │              │
      └──────┬───────┘
             ▼
        skill_match_node
             │
             ▼
          ner_node
             │
             ▼
  [startup_news_node ║ startup_github_node]  ← parallel
             │
             ▼
        composer_node  →  newsletter.pdf
"""

import os
import json
import logging
from typing import TypedDict, Annotated
import operator

from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# local imports
from script3 import news_fetch
from script2 import yc_funding_run
from fetch_news import run_news_fetch
from github_trending import fetch_trending, format_repo
from utility import extract_startup_names as _extract_startup_names
from resume_extractor import resume_extractor
from agent_schema import NEWS_OUTPUT_SCHEMA, GITHUB_OUTPUT_SCHEMA, NewsOutputArgs, GithubOutputArgs

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, HRFlowable, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ── Model ─────────────────────────────────────────────────────────────────────

model = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_KEY"),
    temperature=0.3,
)

# ── Shared State ──────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    resume_path:     str
    skills:          list[str]
    news_list:       Annotated[list[dict], operator.add]   # sorted by relevance_score
    github_repos:    Annotated[list[dict], operator.add]   # sorted by relevance_score
    startup_names:   list[str]
    newsletter_path: str

# ── Tools ─────────────────────────────────────────────────────────────────────

@tool
def extract_info_from_resume(file_path: str) -> str:
    """Reads a resume file and extracts skills as a JSON list."""
    result = resume_extractor(file_path)
    return json.dumps(result.get("skills", []))


@tool
def news_fetch_google(query: str) -> str:
    """Fetches news articles from Google based on the given query."""
    logging.info(f"Fetching Google news for: {query}")
    articles = news_fetch(query)
    return json.dumps(articles, indent=2)


@tool
def fetch_news_from_newsorg(query: str) -> str:
    """Fetches news articles from NewsAPI (newsapi.org) based on the given query."""
    logging.info(f"Fetching NewsAPI news for: {query}")
    try:
        api_key = os.getenv("NEWSORG_API_KEY")
        if not api_key:
            raise ValueError("NEWSORG_API_KEY not set in environment.")
        articles = run_news_fetch(
            api_key=api_key,
            mode="everything",
            query=query,
            language="en",
            sort_by="publishedAt",
            page_size=10,
            days=7,
        )
        return json.dumps(articles, indent=2)
    except Exception as e:
        logging.error(f"NewsAPI error: {e}")
        return json.dumps({"error": str(e)})


@tool
def startup_yc_news(stage_filter: str, min_round_usd: float) -> str:
    """Fetches YC startup funding news filtered by stage and minimum round size."""
    logging.info(f"Fetching YC news: stage={stage_filter}, min_round={min_round_usd}")
    return yc_funding_run(stage_filter, min_round_usd)


@tool
def github_trending_tool(language: str, since: str) -> str:
    """Fetches trending GitHub repositories. 'since': 'daily'|'weekly'|'monthly'."""
    logging.info(f"Fetching GitHub trending: language={language}, since={since}")
    if since not in ["daily", "weekly", "monthly"]:
        since = "weekly"
    repos = fetch_trending(language, since, limit=10)
    formatted = [format_repo(i + 1, r) for i, r in enumerate(repos)]
    return json.dumps(formatted, indent=2)


# ── Output schema tools (LangChain @tool with Pydantic args_schema) ───────────

@tool(args_schema=NewsOutputArgs)
def return_news_data(**kwargs):
    """Must be called to return all collected news articles in structured format."""
    return "SUCCESS"

@tool(args_schema=GithubOutputArgs)
def return_github_data(**kwargs):
    """Must be called to return all trending GitHub repositories in structured format."""
    return "SUCCESS"


# ── Sub-agents ────────────────────────────────────────────────────────────────

news_agent = create_react_agent(
    model=model,
    tools=[news_fetch_google, fetch_news_from_newsorg, startup_yc_news, return_news_data],
    prompt=(
        "You are a startup news aggregator. Your job:\n"
        "1. Call ALL THREE fetch tools:\n"
        "   - news_fetch_google       → general startup/tech news\n"
        "   - fetch_news_from_newsorg → broader news coverage\n"
        "   - startup_yc_news         → YC funding news (stage_filter='all', min_round_usd=0)\n"
        "2. Deduplicate articles by title.\n"
        "3. Call `return_news_data` with ALL collected articles as your FINAL step.\n"
        "Never respond with plain text."
    ),
)

github_agent = create_react_agent(
    model=model,
    tools=[github_trending_tool, return_github_data],
    prompt=(
        "You are a GitHub trends fetcher. "
        "Call github_trending_tool to fetch repositories, "
        "then call `return_github_data` with the structured results as your FINAL step. "
        "Never respond with plain text."
    ),
)


# ── Output extractors ─────────────────────────────────────────────────────────

def extract_tool_output(result: dict, tool_name: str) -> dict:
    """Pull validated tool_call args from an agent result."""
    for message in reversed(result["messages"]):
        for call in getattr(message, "tool_calls", []):
            if call.get("name") == tool_name:
                return call["args"]
        content = getattr(message, "content", "")
        if isinstance(content, str):
            try:
                data = json.loads(content)
                if isinstance(data, dict):
                    return data
            except json.JSONDecodeError:
                pass
    raise ValueError(f"Agent did not call '{tool_name}' and returned no valid JSON.")


def extract_news_output(result: dict) -> dict:
    parsed = extract_tool_output(result, "return_news_data")
    parsed.setdefault("articles", [])
    parsed["total"] = len(parsed["articles"])
    return parsed


def extract_github_output(result: dict) -> dict:
    parsed = extract_tool_output(result, "return_github_data")
    parsed.setdefault("repositories", [])
    return parsed


# ── Embeddings ────────────────────────────────────────────────────────────────

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")


def build_skill_string(skills: list[str]) -> str:
    return ", ".join(skills)


def embed(texts: list[str]) -> np.ndarray:
    return embedding_model.encode(texts, convert_to_numpy=True)


def build_item_string(item: dict, source_type: str) -> str:
    if source_type == "news":
        return " ".join(filter(None, [
            item.get("title", ""),
            item.get("summary", ""),
            item.get("source", ""),
            item.get("funding_stage", ""),
        ]))
    return " ".join(filter(None, [
        item.get("name", ""),
        item.get("description", ""),
        item.get("language", ""),
        " ".join(item.get("topics", [])),
    ]))


def score_and_sort(
    skills: list[str],
    items: list[dict],
    source_type: str,
) -> list[dict]:
    """
    Embed skills string + all items, compute cosine similarity,
    attach relevance_score to each item, and return all sorted
    highest-first. No filtering — caller slices what it needs.
    """
    if not items or not skills:
        return items

    skill_string = build_skill_string(skills)
    item_strings = [build_item_string(i, source_type) for i in items]
    all_vecs     = embed([skill_string] + item_strings)
    skill_vec    = all_vecs[0:1]
    item_vecs    = all_vecs[1:]
    scores       = cosine_similarity(skill_vec, item_vecs)[0]

    scored = [
        {**item, "relevance_score": round(float(score), 4)}
        for item, score in zip(items, scores)
    ]
    scored.sort(key=lambda x: x["relevance_score"], reverse=True)
    return scored


# ── Graph Nodes ───────────────────────────────────────────────────────────────

def resume_node(state: AgentState) -> dict:
    logging.info("▶ resume_node")
    result = resume_extractor(state["resume_path"])
    skills = result.get("skills", [])
    logging.info(f"  Skills extracted: {skills}")
    return {"skills": skills}


def news_node(state: AgentState) -> dict:
    logging.info("▶ news_node")
    messages   = [HumanMessage(content="Fetch the latest startup funding and tech news from all available sources.")]
    last_error = None
    for attempt in range(1, 4):
        try:
            result   = news_agent.invoke({"messages": messages})
            parsed   = extract_news_output(result)
            articles = parsed.get("articles", [])
            logging.info(f"  ✓ {len(articles)} articles fetched")
            return {"news_list": articles}
        except (ValueError, KeyError) as e:
            last_error = e
            logging.warning(f"  Attempt {attempt}/3 failed: {e}")
            messages.append(HumanMessage(
                content=f"Invalid response: {e}. You MUST call `return_news_data`. Try again."
            ))
    logging.error(f"  news_node failed: {last_error}")
    return {"news_list": []}


def github_node(state: AgentState) -> dict:
    logging.info("▶ github_node")
    messages   = [HumanMessage(content="Fetch the top trending GitHub repositories this week across all languages.")]
    last_error = None
    for attempt in range(1, 4):
        try:
            result = github_agent.invoke({"messages": messages})
            parsed = extract_github_output(result)
            repos  = parsed.get("repositories", [])
            logging.info(f"  ✓ {len(repos)} repos fetched")
            return {"github_repos": repos}
        except (ValueError, KeyError) as e:
            last_error = e
            logging.warning(f"  Attempt {attempt}/3 failed: {e}")
            messages.append(HumanMessage(
                content=f"Invalid response: {e}. You MUST call `return_github_data`. Try again."
            ))
    logging.error(f"  github_node failed: {last_error}")
    return {"github_repos": []}


def skill_match_node(state: AgentState) -> dict:
    """Score and sort news + repos by skill relevance. No filtering — keep all, sorted."""
    logging.info("▶ skill_match_node")
    skills       = state.get("skills", [])
    news_list    = state.get("news_list", [])
    github_repos = state.get("github_repos", [])

    if not skills:
        logging.warning("  No skills — skipping scoring, order unchanged")
        return {"news_list": news_list, "github_repos": github_repos}

    logging.info(f"  Skill string: '{build_skill_string(skills)}'")

    sorted_news  = score_and_sort(skills, news_list,    "news")
    sorted_repos = score_and_sort(skills, github_repos, "github")

    logging.info(f"  News scored: {len(sorted_news)}, top score: {sorted_news[0].get('relevance_score') if sorted_news else 'n/a'}")
    logging.info(f"  Repos scored: {len(sorted_repos)}, top score: {sorted_repos[0].get('relevance_score') if sorted_repos else 'n/a'}")

    # Return plain lists — replaces accumulated state with scored+sorted version
    return {
        "news_list":    sorted_news,
        "github_repos": sorted_repos,
    }


def ner_node(state: AgentState) -> dict:
    logging.info("▶ ner_node")
    # Run NER on top 10 news items only (already sorted by relevance)
    combined_text = " ".join(
        a.get("title", "") + " " + a.get("description", "") + " " + a.get("summary", "")
        for a in state.get("news_list", [])[:10]
    )
    startup_names = list(set(_extract_startup_names(combined_text))) if combined_text.strip() else []
    logging.info(f"  Startups found: {startup_names}")
    return {"startup_names": startup_names}


def startup_news_node(state: AgentState) -> dict:
    logging.info("▶ startup_news_node")
    startup_names = state.get("startup_names", [])
    if not startup_names:
        return {"news_list": []}

    query      = " OR ".join(startup_names[:5])
    messages   = [HumanMessage(content=f"Fetch recent news about these startups: {query}.")]
    last_error = None
    for attempt in range(1, 4):
        try:
            result   = news_agent.invoke({"messages": messages})
            parsed   = extract_news_output(result)
            articles = parsed.get("articles", [])
            logging.info(f"  ✓ Startup articles fetched: {len(articles)}")
            return {"news_list": articles}
        except (ValueError, KeyError) as e:
            last_error = e
            logging.warning(f"  Attempt {attempt}/3 failed: {e}")
            messages.append(HumanMessage(
                content=f"Invalid response: {e}. Call `return_news_data`. Try again."
            ))
    logging.error(f"  startup_news_node failed: {last_error}")
    return {"news_list": []}


def startup_github_node(state: AgentState) -> dict:
    logging.info("▶ startup_github_node")
    startup_names = state.get("startup_names", [])
    if not startup_names:
        return {"github_repos": []}

    all_repos = []
    for name in startup_names[:3]:
        messages   = [HumanMessage(content=f"Fetch trending GitHub repositories related to {name}.")]
        last_error = None
        for attempt in range(1, 4):
            try:
                result = github_agent.invoke({"messages": messages})
                parsed = extract_github_output(result)
                repos  = parsed.get("repositories", [])
                all_repos.extend(repos)
                break
            except (ValueError, KeyError) as e:
                last_error = e
                logging.warning(f"  [{name}] Attempt {attempt}/3 failed: {e}")
                messages.append(HumanMessage(
                    content=f"Invalid response: {e}. Call `return_github_data`. Try again."
                ))
        else:
            logging.error(f"  startup_github_node failed for '{name}': {last_error}")

    logging.info(f"  ✓ Startup repos fetched: {len(all_repos)}")
    return {"github_repos": all_repos}


def composer_node(state: AgentState) -> dict:
    logging.info("▶ composer_node")

    skills        = state.get("skills", [])
    startup_names = state.get("startup_names", [])

    # Top 5 only — already sorted by relevance_score from skill_match_node
    top_news  = state.get("news_list", [])[:5]
    top_repos = state.get("github_repos", [])[:5]

    logging.info(f"  Newsletter: top {len(top_news)} news, top {len(top_repos)} repos")

    # ── Compose chain ──────────────────────────────────────────────────────
    class ComposedParagraph(BaseModel):
        title:     str
        paragraph: str
        url:       str

    compose_chain = ChatPromptTemplate.from_messages([
        ("system",
            "You are a newsletter writer for a tech-savvy audience. "
            "Write one engaging paragraph (3-4 sentences) summarising the item. "
            "Highlight why it matters to a software developer or startup founder. "
            "Be concise — no filler phrases like 'In conclusion' or 'It is worth noting'."
        ),
        ("human",
            "Title:       {title}\n"
            "URL:         {url}\n"
            "Description: {description}\n"
            "Source type: {source_type}\n"
        )
    ]) | model.with_structured_output(ComposedParagraph)

    def compose_paragraph(item: dict, source_type: str) -> ComposedParagraph:
        try:
            return compose_chain.invoke({
                "title":       item.get("title")   or item.get("name")        or "Untitled",
                "url":         item.get("url")     or item.get("link")        or "",
                "description": item.get("summary") or item.get("description") or item.get("text") or "",
                "source_type": source_type,
            })
        except Exception as e:
            logging.warning(f"  compose_paragraph failed for '{item.get('title', '')}': {e}")
            return ComposedParagraph(
                title     = item.get("title")   or item.get("name") or "Untitled",
                paragraph = item.get("summary") or item.get("description") or "",
                url       = item.get("url")     or item.get("link") or "",
            )

    news_paragraphs = {
        (item.get("title") or item.get("name", str(i))): compose_paragraph(item, "news")
        for i, item in enumerate(top_news)
    }
    repos_paragraphs = {
        (item.get("title") or item.get("name", str(i))): compose_paragraph(item, "github")
        for i, item in enumerate(top_repos)
    }

    # ── PDF helpers ────────────────────────────────────────────────────────
    def safe_source(article: dict) -> str:
        src = article.get("source", "")
        return src.get("name", "") if isinstance(src, dict) else str(src)

    def score_tag(item: dict) -> str:
        s = item.get("relevance_score", "")
        return f"  ·  relevance {s:.2f}" if isinstance(s, float) else ""

    def safe_url(url: str, n: int = 60) -> str:
        return (url or "")[:n]

    def hr():
        return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ddddee"), spaceAfter=6)

    # ── Styles ─────────────────────────────────────────────────────────────
    base_styles      = getSampleStyleSheet()
    title_style      = ParagraphStyle("NLTitle",     parent=base_styles["Title"],    fontSize=26, textColor=colors.HexColor("#1a1a2e"), spaceAfter=6)
    subtitle_style   = ParagraphStyle("NLSubtitle",  parent=base_styles["Normal"],   fontSize=11, textColor=colors.HexColor("#555577"), spaceAfter=20)
    section_style    = ParagraphStyle("NLSection",   parent=base_styles["Heading1"], fontSize=14, textColor=colors.HexColor("#16213e"), spaceBefore=18, spaceAfter=6)
    item_title_style = ParagraphStyle("NLItemTitle", parent=base_styles["Heading2"], fontSize=11, textColor=colors.HexColor("#0f3460"), spaceAfter=2)
    body_style       = ParagraphStyle("NLBody",      parent=base_styles["Normal"],   fontSize=9,  textColor=colors.HexColor("#333333"), leading=13, spaceAfter=10)
    tag_style        = ParagraphStyle("NLTag",       parent=base_styles["Normal"],   fontSize=8,  textColor=colors.HexColor("#888888"), spaceAfter=14)
    ai_para_style    = ParagraphStyle("NLAIPara",    parent=base_styles["Normal"],   fontSize=9,  textColor=colors.HexColor("#444466"), leading=14, spaceAfter=6, leftIndent=12)
    footer_style     = ParagraphStyle("NLFooter",    parent=base_styles["Normal"],   fontSize=7,  textColor=colors.grey, alignment=1)

    # ── Story ──────────────────────────────────────────────────────────────
    story = []
    story.append(Paragraph("Congniscan: Tech & Startup Newsletter", title_style))
    story.append(Paragraph("Personalised for your skills &amp; interests — powered by AI agents", subtitle_style))
    story.append(hr())

    if skills:
        story.append(Paragraph("🎯 Your Skills", section_style))
        story.append(Paragraph(", ".join(skills), body_style))
        story.append(hr())

    if startup_names:
        story.append(Paragraph("🏢 Startups in the News", section_style))
        story.append(Paragraph(", ".join(startup_names), body_style))
        story.append(hr())

    # ── Top 5 News ─────────────────────────────────────────────────────────
    story.append(Paragraph("📰 Top 5 News Articles", section_style))
    if top_news:
        for article in top_news:
            title  = article.get("title") or article.get("name") or "Untitled"
            url    = article.get("url")   or article.get("link") or ""
            source = safe_source(article)
            pub    = article.get("publishedAt") or article.get("published") or ""

            story.append(Paragraph(title, item_title_style))
            composed = news_paragraphs.get(title)
            if composed and composed.paragraph:
                story.append(Paragraph(composed.paragraph, ai_para_style))
            else:
                desc = article.get("description") or article.get("text") or ""
                if desc:
                    story.append(Paragraph(desc[:300] + ("..." if len(desc) > 300 else ""), body_style))
            story.append(Paragraph(
                f"<i>{source}</i>  ·  {pub[:10]}{score_tag(article)}  ·  <a href='{url}'>{safe_url(url)}</a>",
                tag_style,
            ))
    else:
        story.append(Paragraph("No news articles found.", body_style))

    story.append(PageBreak())

    # ── Top 5 Repos ────────────────────────────────────────────────────────
    story.append(Paragraph("🐙 Top 5 GitHub Repositories", section_style))
    if top_repos:
        for repo in top_repos:
            name  = repo.get("title") or repo.get("name") or "Unknown Repo"
            url   = repo.get("url")   or ""
            stars = repo.get("stars") or repo.get("stargazers_count") or repo.get("stars_total") or 0
            lang  = repo.get("language") or ""

            story.append(Paragraph(name, item_title_style))
            composed = repos_paragraphs.get(name)
            if composed and composed.paragraph:
                story.append(Paragraph(composed.paragraph, ai_para_style))
            else:
                desc = repo.get("description") or ""
                if desc:
                    story.append(Paragraph(desc[:250] + ("..." if len(desc) > 250 else ""), body_style))
            story.append(Paragraph(
                f"⭐ {stars}  ·  {lang}{score_tag(repo)}  ·  <a href='{url}'>{safe_url(url)}</a>",
                tag_style,
            ))
    else:
        story.append(Paragraph("No repositories found.", body_style))

    story.append(hr())
    story.append(Paragraph("Generated by AI Super-Agent Pipeline — LangGraph + Mistral", footer_style))

    # ── Build PDF ──────────────────────────────────────────────────────────
    output_path = "newsletter.pdf"
    doc = SimpleDocTemplate(
        output_path, pagesize=letter,
        leftMargin=0.75*inch, rightMargin=0.75*inch,
        topMargin=0.75*inch,  bottomMargin=0.75*inch,
    )
    doc.build(story)
    logging.info(f"  ✓ Newsletter written → {output_path}")
    return {"newsletter_path": output_path}


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    g.add_node("resume_node",         resume_node)
    g.add_node("news_node",           news_node)
    g.add_node("github_node",         github_node)
    g.add_node("skill_match_node",    skill_match_node)
    g.add_node("ner_node",            ner_node)
    g.add_node("startup_news_node",   startup_news_node)
    g.add_node("startup_github_node", startup_github_node)
    g.add_node("composer_node",       composer_node)

    g.set_entry_point("resume_node")
    g.add_edge("resume_node",         "news_node")
    g.add_edge("resume_node",         "github_node")
    g.add_edge("news_node",           "skill_match_node")
    g.add_edge("github_node",         "skill_match_node")
    g.add_edge("skill_match_node",    "ner_node")
    g.add_edge("ner_node",            "startup_news_node")
    g.add_edge("ner_node",            "startup_github_node")
    g.add_edge("startup_news_node",   "skill_match_node")
    g.add_edge("startup_github_node", "skill_match_node")
    g.add_edge("skill_match_node",      "composer_node")
    g.add_edge("composer_node",      END)

    return g.compile()


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_pipeline(resume_path: str) -> AgentState:
    graph = build_graph()
    initial_state: AgentState = {
        "resume_path":     resume_path,
        "skills":          [],
        "news_list":       [],
        "github_repos":    [],
        "startup_names":   [],
        "newsletter_path": "",
    }
    final_state = graph.invoke(initial_state)
    print(f"\n✅ Newsletter saved to : {final_state['newsletter_path']}")
    print(f"   Skills found       : {final_state['skills']}")
    print(f"   Startups detected  : {final_state['startup_names']}")
    print(f"   News articles      : {len(final_state['news_list'])}")
    print(f"   GitHub repos       : {len(final_state['github_repos'])}")
    print(f"   Top news scores    : {[r.get('relevance_score') for r in final_state['news_list'][:5]]}")
    print(f"   Top repo scores    : {[r.get('relevance_score') for r in final_state['github_repos'][:5]]}")
    return final_state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the AI Super-Agent Pipeline")
    parser.add_argument("--resume", type=str, default="resume.txt", help="Path to resume file")
    args = parser.parse_args()
    run_pipeline(args.resume)