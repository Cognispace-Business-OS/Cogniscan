from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class ComposedParagraph(BaseModel):
    title:     str
    paragraph: str   # 3-4 sentence newsletter blurb
    url:       str

class ArticleOutput(BaseModel):
    title: str
    link: str
    source: str
    published: str
    summary: str
    funding_amount: Optional[str] = None
    funding_stage: Optional[str] = None
    startup_name: Optional[str] = None
    origin: Literal["google", "newsapi", "yc"] = Field(description="Which tool fetched this article.")

class NewsOutputArgs(BaseModel):
    articles: List[ArticleOutput]
    total: int = Field(description="Total number of articles returned.")
    sources: List[Literal["google", "newsapi", "yc"]] = Field(description="Which sources were queried.")

class Contributor(BaseModel):
    username: str
    avatar_url: Optional[str] = None

class RepoOutput(BaseModel):
    rank: int = Field(description="Position in trending list (1-based).")
    name: str = Field(description="Repository name e.g. 'torvalds/linux'.")
    owner: str = Field(description="GitHub username or org.")
    url: str = Field(description="Full GitHub URL.")
    description: Optional[str] = Field(None, description="Repo description.")
    language: Optional[str] = Field(None, description="Primary programming language.")
    stars_total: int = Field(description="Total stargazers.")
    stars_gained: int = Field(description="Stars gained in the 'since' window.")
    forks: int = Field(description="Total forks.")
    built_by: Optional[List[Contributor]] = Field(None, description="Top contributors.")

class GithubQueryArgs(BaseModel):
    language: str = Field(description="Language filter used. 'all' if none.")
    since: Literal["daily", "weekly", "monthly"]
    limit: int

class GithubOutputArgs(BaseModel):
    repositories: List[RepoOutput]
    query: GithubQueryArgs = Field(description="The parameters used for this fetch.")
    total: int = Field(description="Number of repos returned.")
    fetched_at: str = Field(description="ISO 8601 UTC timestamp of fetch.")

NEWS_OUTPUT_SCHEMA = NewsOutputArgs
GITHUB_OUTPUT_SCHEMA = GithubOutputArgs
