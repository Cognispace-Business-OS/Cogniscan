import spacy

nlp = spacy.load("en_core_web_sm")

def extract_startup_names(news: str) -> list[str]:
    """
    Extract startup/company names from a news string using spaCy NER.

    Args:
        news: A news article or snippet as a plain string.

    Returns:
        A deduplicated list of organization names found.
    """
    doc = nlp(news)
    startups = list({
        ent.text.strip()
        for ent in doc.ents
        if ent.label_ == "ORG"
    })
    return startups
