import arxiv
from concurrent.futures import ThreadPoolExecutor, as_completed
from db.vector import vector_search, upsert_papers
from core.state import AgentState
from core.events import emit, EventType

SIMILARITY_THRESHOLD = 0.35
ARXIV_DELAY = 1  # reduced from 3 seconds
ARXIV_TIMEOUT = 15  # max seconds per query

def fetch_arxiv(query: str, max_results: int = 10) -> list[dict]:
    print(f"[arxiv] fetching: {query[:50]}...")
    client = arxiv.Client(delay_seconds=ARXIV_DELAY, num_retries=2)
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )

    papers = []
    try:
        for r in client.results(search):
            papers.append({
                "id":         r.entry_id.split("/")[-1],
                "title":      r.title,
                "abstract":   r.summary,
                "authors":    [a.name for a in r.authors],
                "published":  r.published.date(),
                "categories": r.categories,
            })
    except Exception as e:
        print(f"[arxiv] error fetching '{query[:30]}': {e}")

    print(f"[arxiv] found {len(papers)} papers for: {query[:50]}")
    return papers

# ── dedup ─────────────────────────────────────────────────────────────────────

def dedup(papers: list[dict]) -> list[dict]:
    seen = set()
    out = []
    for p in papers:
        base_id = p["id"].split("v")[0]   # strip version suffix
        if base_id not in seen:
            seen.add(base_id)
            out.append(p)
    return out

# ── relevance gate ────────────────────────────────────────────────────────────

def filter_by_relevance(papers: list[dict], top_n: int = 30) -> list[dict]:
    """
    Drops papers below similarity threshold, then caps at top_n.
    Only applies to papers that came through vector_search (have similarity score).
    arXiv-only results pass through unfiltered.
    """
    scored = [p for p in papers if "similarity" in p]
    unscored = [p for p in papers if "similarity" not in p]

    passed = [p for p in scored if p["similarity"] >= SIMILARITY_THRESHOLD]
    passed.sort(key=lambda p: p["similarity"], reverse=True)

    combined = passed + unscored
    return combined[:top_n]

def process_query(query: str) -> list[dict]:
    """Process a single sub-query: vector search + arXiv fetch in parallel."""
    papers = []
    print(f"[retrieval] starting query: {query[:50]}...")

    # Run vector search and arXiv fetch concurrently for same query
    with ThreadPoolExecutor(max_workers=2) as executor:
        local_future = executor.submit(vector_search, query, 10)
        arxiv_future = executor.submit(fetch_arxiv, query, 10)

        # Collect local results (with timeout)
        try:
            local = local_future.result(timeout=10)
            papers.extend(local)
            print(f"[retrieval] vector search: {len(local)} papers for '{query[:30]}'")
        except Exception as e:
            print(f"[retrieval] vector search failed for '{query[:30]}': {e}")

        # Collect arXiv results (with timeout)
        try:
            live = arxiv_future.result(timeout=30)
            papers.extend(live)
        except Exception as e:
            print(f"[retrieval] arXiv timeout/failed for '{query[:30]}': {e}")

    return papers


# ── main retrieval node ───────────────────────────────────────────────────────
def retrieval_node(state: AgentState) -> AgentState:
    """Retrieve papers for all sub-queries in parallel."""
    sub_queries = state["sub_queries"]
    all_papers = []
    completed_count = 0

    emit(EventType.STAGE_START, stage="retrieval", message="Searching for papers...")

    # Process all sub-queries concurrently (3 workers to respect arXiv rate limits)
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_query = {
            executor.submit(process_query, query): query
            for query in sub_queries
        }

        for future in as_completed(future_to_query):
            query = future_to_query[future]
            completed_count += 1
            try:
                papers = future.result()
                all_papers.extend(papers)
                print(f"[retrieval] completed query: {query[:50]}...")
                emit(
                    EventType.QUERY_COMPLETE,
                    query=query,
                    papers_found=len(papers),
                    progress=completed_count,
                    total=len(sub_queries),
                )
            except Exception as e:
                print(f"[retrieval] query failed '{query}': {e}")

    live_only = [p for p in all_papers if "similarity" not in p]
    if live_only:
        upsert_papers(live_only)

    deduped = dedup(all_papers)
    filtered = filter_by_relevance(deduped, top_n=30)

    # serialize datetime.date to string for LangGraph state passing
    for p in filtered:
        if p.get("published") and not isinstance(p["published"], str):
            p["published"] = str(p["published"])

    print(f"[retrieval] {len(all_papers)} raw → {len(deduped)} deduped → {len(filtered)} after gate")

    emit(
        EventType.PAPERS_FOUND,
        raw=len(all_papers),
        deduped=len(deduped),
        filtered=len(filtered),
    )
    emit(EventType.STAGE_END, stage="retrieval", count=len(filtered))

    return {**state, "papers": filtered}