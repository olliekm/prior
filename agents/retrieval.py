import os
import arxiv
from concurrent.futures import ThreadPoolExecutor, as_completed
from core.state import AgentState
from core.events import emit, EventType

SIMILARITY_THRESHOLD = 0.35
ARXIV_DELAY = 1  # reduced from 3 seconds
ARXIV_TIMEOUT = 15  # max seconds per query

# Database is optional - will work with arXiv only if DB unavailable
DB_AVAILABLE = False
try:
    if os.environ.get("DATABASE_URL"):
        from db.vector import vector_search, upsert_papers
        DB_AVAILABLE = True
except Exception:
    pass

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

    if DB_AVAILABLE:
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
    else:
        # DB not available - fetch from arXiv only
        try:
            papers = fetch_arxiv(query, 10)
        except Exception as e:
            print(f"[retrieval] arXiv failed for '{query[:30]}': {e}")

    return papers


# ── main retrieval node ───────────────────────────────────────────────────────
def retrieval_node(state: AgentState) -> AgentState:
    """Retrieve papers for all sub-queries in parallel."""
    sub_queries = state.get("sub_queries", [])
    searched_queries = state.get("searched_queries", [])
    existing_papers = state.get("papers", [])

    # Only search queries we haven't searched yet
    new_queries = [q for q in sub_queries if q not in searched_queries]

    if not new_queries:
        print("[retrieval] no new queries to search")
        return state

    iteration = state.get("iteration", 1)
    emit(
        EventType.STAGE_START,
        stage="retrieval",
        message=f"Searching for papers (iteration {iteration})...",
        queries=len(new_queries),
    )

    all_papers = []
    completed_count = 0

    # Process all sub-queries concurrently (3 workers to respect arXiv rate limits)
    with ThreadPoolExecutor(max_workers=3) as executor:
        future_to_query = {
            executor.submit(process_query, query): query
            for query in new_queries
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
                    total=len(new_queries),
                )
            except Exception as e:
                print(f"[retrieval] query failed '{query}': {e}")

    # Track which queries we've searched
    updated_searched = searched_queries + new_queries

    live_only = [p for p in all_papers if "similarity" not in p]
    if live_only and DB_AVAILABLE:
        upsert_papers(live_only)

    # Combine with existing papers from previous iterations
    combined_papers = existing_papers + all_papers
    deduped = dedup(combined_papers)
    filtered = filter_by_relevance(deduped, top_n=50)  # Allow more papers across iterations

    # serialize datetime.date to string for LangGraph state passing
    for p in filtered:
        if p.get("published") and not isinstance(p["published"], str):
            p["published"] = str(p["published"])

    print(f"[retrieval] {len(all_papers)} new + {len(existing_papers)} existing → {len(deduped)} deduped → {len(filtered)} filtered")

    emit(
        EventType.PAPERS_FOUND,
        raw=len(all_papers),
        total=len(combined_papers),
        deduped=len(deduped),
        filtered=len(filtered),
    )
    emit(EventType.STAGE_END, stage="retrieval", count=len(filtered))

    return {
        **state,
        "papers": filtered,
        "searched_queries": updated_searched,
    }