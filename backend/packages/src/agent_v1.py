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
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from langchain.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import create_react_agent

# local imports
from script3 import news_fetch
from script2 import yc_funding_run
from fetch_news import run_news_fetch
from github_trending import fetch_trending, format_repo
from utility import extract_startup_names as _extract_startup_names

# PDF
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, HRFlowable, PageBreak,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors

load_dotenv()
logging.basicConfig(level=logging.INFO)

# ── Model ────────────────────────────────────────────────────────────────────

model = ChatMistralAI(
    model="mistral-small-latest",
    api_key=os.getenv("MISTRAL_KEY"),
    temperature=0.3,
)

# ── Shared State ─────────────────────────────────────────────────────────────

class AgentState(TypedDict):
    resume_path: str                          # input: path to resume file
    skills: list[str]                         # resume_node → skill_match_node
    news_list: Annotated[list[dict], operator.add]   # news_node + startup_news_node
    github_repos: Annotated[list[dict], operator.add]  # github_node + startup_github_node
    relevant_news: list[dict]                 # skill_match_node output
    relevant_repos: list[dict]                # skill_match_node output
    startup_names: list[str]                  # ner_node output
    newsletter_path: str                      # composer_node output

# ── Tools ────────────────────────────────────────────────────────────────────

@tool
def extract_info_from_resume(file_path: str) -> str:
    """Reads a resume file and extracts skills as a JSON list."""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            text = f.read()
        # Use the model directly for extraction (no sub-agent loop needed)
        resp = model.invoke([
            HumanMessage(content=(
                f"Extract all technical and professional skills from this resume. "
                f"Return ONLY a JSON array of skill strings, no explanation.\n\n{text}"
            ))
        ])
        raw = resp.content.strip().strip("```json").strip("```").strip()
        return raw
    except Exception as e:
        logging.error(f"Resume extraction error: {e}")
        return "[]"


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


# ── Sub-agents ───────────────────────────────────────────────────────────────

resume_agent = create_react_agent(
    model=model,
    tools=[extract_info_from_resume],
    prompt="You are a resume parser. Extract all skills from the resume and return them as a JSON list.",
)

news_agent = create_react_agent(
    model=model,
    tools=[news_fetch_google, startup_yc_news, fetch_news_from_newsorg],
    prompt="You are a news fetcher. Fetch startup and tech funding news. Return results as JSON.",
)

github_agent = create_react_agent(
    model=model,
    tools=[github_trending_tool],
    prompt="You are a GitHub trends fetcher. Fetch trending repositories and return them as JSON.",
)

# ── Graph Nodes ───────────────────────────────────────────────────────────────

def resume_node(state: AgentState) -> dict:
    """Runs the resume agent and stores extracted skills in state."""
    logging.info("▶ resume_node")
    result = resume_agent.invoke({
        "messages": [HumanMessage(
            content=f"Extract all skills from this resume file: {state['resume_path']}"
        )]
    })
    raw = result["messages"][-1].content.strip().strip("```json").strip("```").strip()
    try:
        skills = json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: split by comma if the model returned a plain list
        skills = [s.strip().strip('"') for s in raw.strip("[]").split(",") if s.strip()]
    logging.info(f"  Skills found: {skills}")
    return {"skills": skills}


def news_node(state: AgentState) -> dict:
    """Fetches general startup/tech news and stores in news_list."""
    logging.info("▶ news_node")
    result = news_agent.invoke({
        "messages": [HumanMessage(
            content="Fetch the latest startup funding news and tech news from all sources. Return JSON."
        )]
    })
    raw = result["messages"][-1].content
    try:
        articles = json.loads(raw)
        if isinstance(articles, dict):
            articles = articles.get("articles", [articles])
    except Exception:
        articles = [{"title": raw, "source": "agent", "url": ""}]
    logging.info(f"  Articles fetched: {len(articles)}")
    return {"news_list": articles}


def github_node(state: AgentState) -> dict:
    """Fetches trending GitHub repos and stores in github_repos."""
    logging.info("▶ github_node")
    result = github_agent.invoke({
        "messages": [HumanMessage(
            content="Fetch the top trending GitHub repositories this week across all languages."
        )]
    })
    raw = result["messages"][-1].content
    try:
        repos = json.loads(raw)
        if isinstance(repos, dict):
            repos = [repos]
    except Exception:
        repos = [{"title": raw, "url": "", "stars": 0, "language": "", "description": ""}]
    logging.info(f"  Repos fetched: {len(repos)}")
    return {"github_repos": repos}


def skill_match_node(state: AgentState) -> dict:
    """Filters news and repos to those relevant to the candidate's skills."""
    logging.info("▶ skill_match_node")
    skills = state.get("skills", [])
    news_list = state.get("news_list", [])
    github_repos = state.get("github_repos", [])

    if not skills:
        return {"relevant_news": news_list, "relevant_repos": github_repos}

    skills_lower = [s.lower() for s in skills]

    def matches(text: str) -> bool:
        t = text.lower()
        return any(skill in t for skill in skills_lower)

    relevant_news = [
        a for a in news_list
        if matches(a.get("title", "") + " " + a.get("description", "") + " " + a.get("text", ""))
    ]
    relevant_repos = [
        r for r in github_repos
        if matches(r.get("title", "") + " " + r.get("description", "") + " " + r.get("language", ""))
    ]

    logging.info(f"  Relevant news: {len(relevant_news)}, Relevant repos: {len(relevant_repos)}")
    return {
        "relevant_news": relevant_news,
        "relevant_repos": relevant_repos,
        # Merge relevant back into main lists so they persist in state
        "news_list": relevant_news,
        "github_repos": relevant_repos,
    }


def ner_node(state: AgentState) -> dict:
    """Runs NER on all relevant news to extract startup names."""
    logging.info("▶ ner_node")
    combined_text = " ".join(
        a.get("title", "") + " " + a.get("description", "") + " " + a.get("text", "")
        for a in state.get("relevant_news", [])
    )
    startup_names = list(set(_extract_startup_names(combined_text))) if combined_text.strip() else []
    logging.info(f"  Startups found: {startup_names}")
    return {"startup_names": startup_names}


def startup_news_node(state: AgentState) -> dict:
    """Fetches news specifically about the startups identified by NER."""
    logging.info("▶ startup_news_node")
    startup_names = state.get("startup_names", [])
    if not startup_names:
        return {"news_list": []}

    query = " OR ".join(startup_names[:5])  # NewsAPI supports OR operators
    result = news_agent.invoke({
        "messages": [HumanMessage(
            content=f"Fetch recent news about these startups: {query}. Return JSON."
        )]
    })
    raw = result["messages"][-1].content
    try:
        articles = json.loads(raw)
        if isinstance(articles, dict):
            articles = articles.get("articles", [articles])
    except Exception:
        articles = []
    logging.info(f"  Startup articles fetched: {len(articles)}")
    return {"news_list": articles}


def startup_github_node(state: AgentState) -> dict:
    """Fetches GitHub repos specifically for the startups identified by NER."""
    logging.info("▶ startup_github_node")
    startup_names = state.get("startup_names", [])
    if not startup_names:
        return {"github_repos": []}

    all_repos = []
    for name in startup_names[:3]:  # limit to avoid rate limits
        result = github_agent.invoke({
            "messages": [HumanMessage(
                content=f"Fetch trending GitHub repositories related to {name}."
            )]
        })
        raw = result["messages"][-1].content
        try:
            repos = json.loads(raw)
            if isinstance(repos, dict):
                repos = [repos]
            all_repos.extend(repos)
        except Exception:
            pass

    logging.info(f"  Startup repos fetched: {len(all_repos)}")
    return {"github_repos": all_repos}


def composer_node(state: AgentState) -> dict:
    """Composes a newsletter from all state data and writes it as a PDF."""
    logging.info("▶ composer_node")

    skills        = state.get("skills", [])
    relevant_news = state.get("relevant_news", [])
    relevant_repos= state.get("relevant_repos", [])
    news_list     = state.get("news_list", [])
    github_repos  = state.get("github_repos", [])
    startup_names = state.get("startup_names", [])

    # Deduplicate by title
    def dedup(items, key="title"):
        seen, out = set(), []
        for item in items:
            k = item.get(key, "")
            if k not in seen:
                seen.add(k)
                out.append(item)
        return out

    all_news  = dedup(relevant_news + news_list)
    all_repos = dedup(relevant_repos + github_repos)

    # ── Build PDF ──────────────────────────────────────────────────────────
    output_path = "newsletter.pdf"
    doc = SimpleDocTemplate(
        output_path,
        pagesize=letter,
        leftMargin=0.75 * inch,
        rightMargin=0.75 * inch,
        topMargin=0.75 * inch,
        bottomMargin=0.75 * inch,
    )

    base_styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "NLTitle",
        parent=base_styles["Title"],
        fontSize=26,
        textColor=colors.HexColor("#1a1a2e"),
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "NLSubtitle",
        parent=base_styles["Normal"],
        fontSize=11,
        textColor=colors.HexColor("#555577"),
        spaceAfter=20,
    )
    section_style = ParagraphStyle(
        "NLSection",
        parent=base_styles["Heading1"],
        fontSize=14,
        textColor=colors.HexColor("#16213e"),
        spaceBefore=18,
        spaceAfter=6,
        borderPad=4,
    )
    item_title_style = ParagraphStyle(
        "NLItemTitle",
        parent=base_styles["Heading2"],
        fontSize=11,
        textColor=colors.HexColor("#0f3460"),
        spaceAfter=2,
    )
    body_style = ParagraphStyle(
        "NLBody",
        parent=base_styles["Normal"],
        fontSize=9,
        textColor=colors.HexColor("#333333"),
        leading=13,
        spaceAfter=10,
    )
    tag_style = ParagraphStyle(
        "NLTag",
        parent=base_styles["Normal"],
        fontSize=8,
        textColor=colors.HexColor("#888888"),
        spaceAfter=14,
    )

    def hr():
        return HRFlowable(width="100%", thickness=0.5, color=colors.HexColor("#ddddee"), spaceAfter=6)

    story = []

    # Header
    story.append(Paragraph("Tech &amp; Startup Newsletter", title_style))
    story.append(Paragraph(
        "Personalised for your skills &amp; interests — powered by AI agents",
        subtitle_style,
    ))
    story.append(hr())

    # Skills
    if skills:
        story.append(Paragraph("🎯 Your Skills", section_style))
        story.append(Paragraph(", ".join(skills), body_style))
        story.append(hr())

    # Startups detected
    if startup_names:
        story.append(Paragraph("🏢 Startups in the News", section_style))
        story.append(Paragraph(", ".join(startup_names), body_style))
        story.append(hr())

    # Relevant News
    story.append(Paragraph("📰 Relevant News Articles", section_style))
    if all_news:
        for article in all_news[:20]:
            title   = article.get("title") or article.get("name") or "Untitled"
            desc    = article.get("description") or article.get("text") or ""
            url     = article.get("url") or article.get("link") or ""
            source  = article.get("source", {})
            source  = source.get("name", source) if isinstance(source, dict) else str(source)
            pub     = article.get("publishedAt") or article.get("published") or ""

            story.append(Paragraph(title, item_title_style))
            if desc:
                story.append(Paragraph(desc[:300] + ("..." if len(desc) > 300 else ""), body_style))
            story.append(Paragraph(
                f"<i>{source}</i>  ·  {pub[:10]}  ·  <a href='{url}'>{url[:60]}</a>",
                tag_style,
            ))
    else:
        story.append(Paragraph("No relevant news found.", body_style))

    story.append(PageBreak())

    # GitHub Repos
    story.append(Paragraph("🐙 Relevant GitHub Repositories", section_style))
    if all_repos:
        for repo in all_repos[:15]:
            name  = repo.get("title") or repo.get("name") or "Unknown Repo"
            desc  = repo.get("description") or ""
            url   = repo.get("url") or ""
            stars = repo.get("stars") or repo.get("stargazers_count") or 0
            lang  = repo.get("language") or ""

            story.append(Paragraph(name, item_title_style))
            if desc:
                story.append(Paragraph(desc[:250] + ("..." if len(desc) > 250 else ""), body_style))
            story.append(Paragraph(
                f"⭐ {stars}  ·  {lang}  ·  <a href='{url}'>{url[:60]}</a>",
                tag_style,
            ))
    else:
        story.append(Paragraph("No relevant repositories found.", body_style))

    # Footer
    story.append(hr())
    story.append(Paragraph(
        "Generated by AI Super-Agent Pipeline — LangGraph + Mistral",
        ParagraphStyle("footer", parent=base_styles["Normal"], fontSize=7,
                       textColor=colors.grey, alignment=1),
    ))

    doc.build(story)
    logging.info(f"  Newsletter written to: {output_path}")
    return {"newsletter_path": output_path}


# ── Graph Assembly ────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    g = StateGraph(AgentState)

    # Nodes
    g.add_node("resume_node",         resume_node)
    g.add_node("news_node",           news_node)
    g.add_node("github_node",         github_node)
    g.add_node("skill_match_node",    skill_match_node)
    g.add_node("ner_node",            ner_node)
    g.add_node("startup_news_node",   startup_news_node)
    g.add_node("startup_github_node", startup_github_node)
    g.add_node("composer_node",       composer_node)

    # Entry
    g.set_entry_point("resume_node")

    # resume → parallel (news + github)
    g.add_edge("resume_node", "news_node")
    g.add_edge("resume_node", "github_node")

    # both parallel branches → skill_match
    g.add_edge("news_node",   "skill_match_node")
    g.add_edge("github_node", "skill_match_node")

    # skill_match → ner
    g.add_edge("skill_match_node", "ner_node")

    # ner → parallel (startup news + startup github)
    g.add_edge("ner_node", "startup_news_node")
    g.add_edge("ner_node", "startup_github_node")

    # both startup branches → composer
    g.add_edge("startup_news_node",   "composer_node")
    g.add_edge("startup_github_node", "composer_node")

    # composer → END
    g.add_edge("composer_node", END)

    return g.compile()


# ── Entry Point ───────────────────────────────────────────────────────────────

def run_pipeline(resume_path: str) -> AgentState:
    """
    Run the full super-agent pipeline for a given resume file.

    Parameters
    ----------
    resume_path : str — Path to the candidate's resume (.txt or .pdf text).

    Returns
    -------
    AgentState — Final state containing skills, news, repos, startups, and
                 the path to the generated newsletter PDF.
    """
    graph = build_graph()
    initial_state: AgentState = {
        "resume_path":    resume_path,
        "skills":         [],
        "news_list":      [],
        "github_repos":   [],
        "relevant_news":  [],
        "relevant_repos": [],
        "startup_names":  [],
        "newsletter_path": "",
    }
    final_state = graph.invoke(initial_state)
    print(f"\n✅ Newsletter saved to: {final_state['newsletter_path']}")
    print(f"   Skills found      : {final_state['skills']}")
    print(f"   Startups detected : {final_state['startup_names']}")
    print(f"   News articles     : {len(final_state['news_list'])}")
    print(f"   GitHub repos      : {len(final_state['github_repos'])}")
    return final_state


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run the AI Super-Agent Pipeline")
    parser.add_argument("--resume", type=str, default="resume.txt", help="Path to the resume file (.txt or .pdf)")
    args = parser.parse_args()
    run_pipeline(args.resume)