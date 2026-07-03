# Test arxiv - embeddings

import feedparser
import numpy as np
from datetime import datetime, timedelta, timezone
from dateutil import parser as dateparser
from sentence_transformers import SentenceTransformer

from seed_topics_humAI import SEED_TOPICS

CATEGORIES = ["cs.AI", "cs.CL", "cs.CY", "cs.HC", ]
LOOKBACK_DAYS = 400
MAX_RESULTS = 2
MIN_SIMILARITY = 0.4

def fetch_arxiv_papers():
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    papers = []

    for category in CATEGORIES:
        url = (
            "http://export.arxiv.org/api/query?"
            f"search_query=cat:{category}"
            "&sortBy=submittedDate&sortOrder=descending"
            "&max_results=50"
        )
        feed = feedparser.parse(url)
        print(f"[{category}] fetched {len(feed.entries)} entries")

        for entry in feed.entries:
            published = dateparser.parse(entry.published)
            if published < cutoff:
                continue

            papers.append({
                "title": entry.title,
                "summary": entry.summary,
                "published": published,
                "link": entry.link,
                "category": category
            })

    return papers


def rank_by_similarity(papers, model):
    if not papers:
        return []

    seed_embeddings = model.encode(SEED_TOPICS)
    seed_centroid = seed_embeddings.mean(axis=0)

    texts = [f"{p['title']}. {p['summary']}" for p in papers]
    paper_embeddings = model.encode(texts)

    for paper, embedding in zip(papers, paper_embeddings):
        similarity = np.dot(embedding, seed_centroid) / (
            np.linalg.norm(embedding) * np.linalg.norm(seed_centroid)
        )
        paper["score"] = float(similarity)

    return sorted(papers, key=lambda p: -p["score"])

def build_newsletter_markdown(papers):
    today = datetime.now().strftime("%Y-%m-%d")
    lines = [
        f"# Humanitarian AI Weekly Newsletter — {today}",
        "",
        f"_{len(papers)} paper(s) this week, ranked by relevance to humanitarian tech and innovation._",
        "",
        "---",
        "",
    ]

    for i, paper in enumerate(papers, start=1):
        title = paper["title"].strip().replace("\n", " ")
        summary = paper["summary"].strip().replace("\n", " ")
        summary_short = summary if len(summary) <= 400 else summary[:400].rsplit(" ", 1)[0] + "…"

        lines.append(f"## {i}. {title}")
        lines.append("")
        lines.append(f"**Category:** `{paper['category']}`  |  **Relevance score:** {paper['score']:.3f}  |  **Published:** {paper['published'].strftime('%Y-%m-%d')}")
        lines.append("")
        lines.append(summary_short)
        lines.append("")
        lines.append(f"[Read the full paper]({paper['link']})")
        lines.append("")
        lines.append("---")
        lines.append("")

    return "\n".join(lines)

if __name__ == "__main__":
    print(f"Fetching papers from arXiv in the last {LOOKBACK_DAYS} days...\n")
    papers = fetch_arxiv_papers()
    print(f"\nFetched {len(papers)} candidate papers total.\n")

    print("Loading embedding model and ranking by similarity to your seed topics...\n")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    ranked = rank_by_similarity(papers, model)

    qualifying = [p for p in ranked if p["score"] >= MIN_SIMILARITY]

    if not qualifying:
        print(f"No papers this week scored above the similarity threshold ({MIN_SIMILARITY}).")
        print("Using the top 2 anyway, for reference, all below threshold:\n")
        qualifying = ranked

    top_papers = qualifying[:MAX_RESULTS]

    newsletter = build_newsletter_markdown(top_papers)

    print("\n" + "=" * 60)
    print(newsletter)
    print("=" * 60 + "\n")

    import os
    os.makedirs("newsletters", exist_ok=True)

    filename = os.path.join("newsletters", f"Humanitarian AI Newsletter - {datetime.now().strftime('%Y-%m-%d')}.md")
    with open(filename, "w") as f:
        f.write(newsletter)

    print(f"Saved to {filename}")