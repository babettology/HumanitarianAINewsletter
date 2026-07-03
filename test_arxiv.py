# Test arxiv

import feedparser
from datetime import datetime, timedelta, timezone
from dateutil import parser  as dateparser

# Humanitarian key words 
KEYWORDS = ["humanitarian", "crisis", "disaster", "refugee", "displacement", "relief"]

CATEGORIES = ["cs.AI", "cs.CL", "cs.CY", "cs.LG", "stat.ML"]
LOOKBACK_DAYS = 100
MAX_RESULTS = 5


def match_keywords(text):
    """Check if any of the keywords are present in the given text."""
    text_lower = text.lower()
    return any(kw in text_lower for kw in KEYWORDS)

def fetch_arxiv_papers():
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    matched = []

    for category in CATEGORIES:
        url = (
         "http://export.arxiv.org/api/query?"
            f"search_query=cat:{category}"
            "&sortBy=submittedDate&sortOrder=descending"
            "&max_results=50"   
        )
        feed = feedparser.parse(url)
        print(f"[{category}] URL: {url}")
        print(f"[{category}] status: {getattr(feed, 'status', 'no status')}")
        print(f"[{category}] bozo (parse error?): {feed.bozo}")
        if feed.bozo:
            print(f"[{category}] bozo_exception: {feed.bozo_exception}")
        print(f"[{category}] fetched {len(feed.entries)} entries")

        for entry in feed.entries: 
            published  = dateparser.parse(entry.published)
            if published < cutoff:
                continue
            combined_text = entry.title + "" + entry.summary
            if match_keywords(combined_text):
                matched.append({
                    "title": entry.title,
                    "summary": entry.summary,
                    "published": published,
                    "link": entry.link,
                    "category": category
                })
    return matched

if __name__ == "__main__":
    print(f"Fetching papers from arXiv in the last {LOOKBACK_DAYS} days")
    results = fetch_arxiv_papers()

    print(f"Found {len(results)} matching papers total.\n")
    print(f"Showing first {MAX_RESULTS}:\n")

    for paper in results[:MAX_RESULTS]:
        print(f"- {paper['title']}")
        print(f"  Published: {paper['published']}")
        print(f"  Link: {paper['link']}\n")