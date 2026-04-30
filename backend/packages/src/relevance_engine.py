import math
import numpy as np
import requests
from collections import Counter
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

encoding_model = SentenceTransformer('all-MiniLM-L6-v2')

# ---------------------------------------------------------------------------
# IDF corpus — grows as documents are scored; used for TF-IDF weighting
# ---------------------------------------------------------------------------
_corpus_doc_count = 0
_skill_doc_freq: dict[str, int] = {}  # how many docs each skill appeared in


def _update_idf(skills_found: set[str], all_skills: list[str]) -> None:
    global _corpus_doc_count
    _corpus_doc_count += 1
    for skill in all_skills:
        if skill in skills_found:
            _skill_doc_freq[skill] = _skill_doc_freq.get(skill, 0) + 1


def _idf(skill: str) -> float:
    """Inverse document frequency for a skill. Falls back to 1.0 if unseen."""
    df = _skill_doc_freq.get(skill, 0)
    if df == 0 or _corpus_doc_count == 0:
        return 1.0
    return math.log((_corpus_doc_count + 1) / (df + 1)) + 1  # smoothed IDF


def _tfidf_score(skills: list[str], words: list[str]) -> float:
    """
    Compute a normalised TF-IDF score for how strongly `words` covers `skills`.
    TF  = (occurrences of skill in doc) / (total words in doc)
    IDF = smoothed corpus IDF for that skill
    Final score is the mean TF-IDF across all skills, clipped to [0, 1].
    """
    if not words:
        return 0.0

    word_counts = Counter(words)
    total_words = len(words)
    skills_found = set()

    tfidf_scores = []
    for skill in skills:
        skill_lower = skill.lower()
        tf = word_counts.get(skill_lower, 0) / total_words
        idf = _idf(skill_lower)
        tfidf_scores.append(tf * idf)
        if tf > 0:
            skills_found.add(skill_lower)

    _update_idf(skills_found, [s.lower() for s in skills])

    raw = sum(tfidf_scores) / len(skills)
    # Normalise: TF-IDF values are tiny floats; scale by total_words for comparison
    normalised = min(raw * total_words, 1.0)
    return normalised


# ---------------------------------------------------------------------------
# Shared semantic similarity helper
# ---------------------------------------------------------------------------

def _semantic_similarity(content: str, skills: list[str]) -> float:
    content_embedding = encoding_model.encode([content])
    skill_embeddings = encoding_model.encode(skills)
    mean_skill_embedding = np.mean(skill_embeddings, axis=0, keepdims=True)
    return float(cosine_similarity(content_embedding, mean_skill_embedding)[0][0])


# ---------------------------------------------------------------------------
# GitHub dependency parsing
# ---------------------------------------------------------------------------

def fetch_dependencies(api_base: str) -> set[str]:
    """
    Fetch and parse requirements.txt and package.json from a GitHub repo.
    Returns a set of lowercase dependency names.
    `api_base` e.g. 'https://api.github.com/repos/owner/repo'
    """
    deps: set[str] = set()
    headers = {"Accept": "application/vnd.github.raw+json"}

    # ---- requirements.txt ------------------------------------------------
    try:
        r = requests.get(f"{api_base}/contents/requirements.txt", headers=headers)
        if r.status_code == 200:
            for line in r.text.splitlines():
                line = line.strip()
                if line and not line.startswith("#"):
                    # strip version specifiers: numpy>=1.21 → numpy
                    pkg = line.split("=")[0].split(">")[0].split("<")[0].split("!")[0]
                    deps.add(pkg.strip().lower())
    except Exception as e:
        print(f"Could not fetch requirements.txt: {e}")

    # ---- package.json ----------------------------------------------------
    try:
        r = requests.get(f"{api_base}/contents/package.json", headers=headers)
        if r.status_code == 200:
            import json
            pkg_data = json.loads(r.text)
            for section in ("dependencies", "devDependencies", "peerDependencies"):
                for dep in pkg_data.get(section, {}):
                    deps.add(dep.lower())
    except Exception as e:
        print(f"Could not fetch package.json: {e}")

    return deps


def _dependency_boost(skills: list[str], deps: set[str]) -> float:
    """
    Returns a boost in [0, 1] based on how many skills appear as direct deps.
    An exact dependency match is a very strong relevance signal.
    """
    if not deps:
        return 0.0
    matched = sum(1 for s in skills if s.lower() in deps)
    return matched / len(skills)


# ---------------------------------------------------------------------------
# Public scoring functions
# ---------------------------------------------------------------------------

def github_relevance(skills: list[str], url: str) -> dict:
    """
    Score a GitHub repo against a skill list.
    `url` = API base, e.g. 'https://api.github.com/repos/owner/repo'

    Returns a score breakdown dict.
    """
    # README
    readme_resp = requests.get(
        f"{url}/readme",
        headers={"Accept": "application/vnd.github.raw+json"}
    )
    readme_resp.raise_for_status()
    readme_content = readme_resp.text
    words = readme_content.lower().split()

    # Scores
    keyword_score = sum(1 for s in skills if s.lower() in words) / len(skills)
    tfidf = _tfidf_score(skills, words)
    semantic = _semantic_similarity(readme_content, skills)

    # Dependency boost
    deps = fetch_dependencies(url)
    dep_boost = _dependency_boost(skills, deps)

    final = (
        0.2 * keyword_score
        + 0.3 * tfidf
        + 0.5 * semantic
        + 0.3 * dep_boost        # additive boost, can push score above 1 — clip below
    )
    final = min(final, 1.0)

    return {
        "keyword_score": round(keyword_score, 4),
        "tfidf_score": round(tfidf, 4),
        "semantic_score": round(semantic, 4),
        "dependency_boost": round(dep_boost, 4),
        "matched_dependencies": sorted(deps & {s.lower() for s in skills}),
        "final_score": round(final, 4),
    }


def news_relevance(skills: list[str], content: str) -> dict:
    """
    Score a news article string against a skill list.
    Returns a score breakdown dict.
    """
    words = content.lower().split()

    keyword_score = sum(1 for s in skills if s.lower() in words) / len(skills)
    tfidf = _tfidf_score(skills, words)
    semantic = _semantic_similarity(content, skills)

    final = min(0.2 * keyword_score + 0.3 * tfidf + 0.5 * semantic, 1.0)

    return {
        "keyword_score": round(keyword_score, 4),
        "tfidf_score": round(tfidf, 4),
        "semantic_score": round(semantic, 4),
        "final_score": round(final, 4),
    }


# ---------------------------------------------------------------------------
# Relevance engine
# ---------------------------------------------------------------------------

def relevance_engine(skills: list[str], items: list[dict]) -> list[dict]:
    """
    Score and rank items by relevance to `skills`.

    Each item should be:
      {"type": "github",  "url": "<api_base>"}
      {"type": "news",    "content": "<article text>"}

    Returns items sorted by final_score descending, with score breakdown attached.
    """
    if not skills:
        raise ValueError("skills list must not be empty")

    scored = []
    for item in items:
        try:
            if item["type"] == "github":
                breakdown = github_relevance(skills, item["url"])
            elif item["type"] == "news":
                breakdown = news_relevance(skills, item["content"])
            else:
                breakdown = {"final_score": 0.0}
        except Exception as e:
            print(f"Skipping item due to error: {e}")
            breakdown = {"final_score": 0.0, "error": str(e)}

        scored.append({**item, **breakdown})

    return sorted(scored, key=lambda x: x["final_score"], reverse=True)