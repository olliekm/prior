"""
MemGPT-style memory system for Prior.

Two-tier memory architecture:
1. Short-term (working memory): Recent context, current session state
2. Long-term (archival memory): Persistent insights, cross-session knowledge

The memory manager handles:
- Automatic archival when working memory gets too large
- Semantic retrieval from long-term memory
- Memory consolidation (merging similar memories)
"""
import os
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])


# ─── Memory Types ─────────────────────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """A single memory entry."""
    id: str
    content: str
    memory_type: str  # "insight", "fact", "query_result", "user_preference"
    source: str       # Where this memory came from
    timestamp: float = field(default_factory=time.time)
    importance: float = 0.5  # 0-1, used for eviction decisions
    access_count: int = 0
    last_accessed: float = field(default_factory=time.time)
    embedding: Optional[list[float]] = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryEntry":
        return cls(**data)


@dataclass
class WorkingMemory:
    """
    Short-term memory for current session.
    Limited capacity, oldest/least important items get archived.
    """
    entries: list[MemoryEntry] = field(default_factory=list)
    max_entries: int = 20
    max_tokens: int = 4000  # Approximate token budget

    def add(self, entry: MemoryEntry) -> Optional[MemoryEntry]:
        """Add entry, return evicted entry if at capacity."""
        self.entries.append(entry)
        evicted = None

        if len(self.entries) > self.max_entries:
            # Evict lowest importance entry
            self.entries.sort(key=lambda e: (e.importance, e.access_count))
            evicted = self.entries.pop(0)

        return evicted

    def get_context(self) -> str:
        """Format working memory as context string."""
        if not self.entries:
            return "No recent context."

        lines = ["## Working Memory (Recent Context)"]
        for entry in self.entries[-10:]:  # Last 10 entries
            lines.append(f"- [{entry.memory_type}] {entry.content}")
        return "\n".join(lines)

    def clear(self):
        """Clear working memory."""
        self.entries = []


@dataclass
class ArchivalMemory:
    """
    Long-term memory with semantic search.
    Persisted to disk, searchable via embeddings.
    """
    entries: list[MemoryEntry] = field(default_factory=list)
    index_path: str = ".prior_memory/archival.json"

    def __post_init__(self):
        self._load()

    def _load(self):
        """Load from disk."""
        if os.path.exists(self.index_path):
            try:
                with open(self.index_path, "r") as f:
                    data = json.load(f)
                    self.entries = [MemoryEntry.from_dict(e) for e in data]
            except Exception as e:
                print(f"[memory] failed to load archival memory: {e}")

    def _save(self):
        """Persist to disk."""
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "w") as f:
            json.dump([e.to_dict() for e in self.entries], f, indent=2)

    def add(self, entry: MemoryEntry):
        """Add entry to archival memory."""
        # Generate embedding if not present
        if entry.embedding is None:
            entry.embedding = self._embed(entry.content)

        self.entries.append(entry)
        self._save()

    def _embed(self, text: str) -> list[float]:
        """Generate embedding for text."""
        response = client.embeddings.create(
            input=text[:8000],  # Truncate if too long
            model="text-embedding-3-small"
        )
        return response.data[0].embedding

    def search(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """Semantic search over archival memory."""
        if not self.entries:
            return []

        query_embedding = self._embed(query)

        # Cosine similarity
        def cosine_sim(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            norm_a = sum(x * x for x in a) ** 0.5
            norm_b = sum(x * x for x in b) ** 0.5
            return dot / (norm_a * norm_b) if norm_a and norm_b else 0

        scored = []
        for entry in self.entries:
            if entry.embedding:
                score = cosine_sim(query_embedding, entry.embedding)
                scored.append((score, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Update access counts
        results = []
        for score, entry in scored[:top_k]:
            entry.access_count += 1
            entry.last_accessed = time.time()
            results.append(entry)

        self._save()
        return results

    def get_by_type(self, memory_type: str, limit: int = 10) -> list[MemoryEntry]:
        """Get memories by type."""
        matching = [e for e in self.entries if e.memory_type == memory_type]
        matching.sort(key=lambda e: e.timestamp, reverse=True)
        return matching[:limit]


# ─── Memory Manager ───────────────────────────────────────────────────────────

class MemoryManager:
    """
    Manages both working and archival memory.
    Handles archival, retrieval, and consolidation.
    """

    def __init__(self, memory_dir: str = ".prior_memory"):
        self.memory_dir = memory_dir
        self.working = WorkingMemory()
        self.archival = ArchivalMemory(index_path=f"{memory_dir}/archival.json")
        self._entry_counter = 0

    def _generate_id(self) -> str:
        self._entry_counter += 1
        return f"mem_{int(time.time())}_{self._entry_counter}"

    # ─── Core Operations ──────────────────────────────────────────────────────

    def remember(
        self,
        content: str,
        memory_type: str = "insight",
        source: str = "agent",
        importance: float = 0.5,
        metadata: dict = None,
    ) -> MemoryEntry:
        """
        Add something to memory.
        Goes to working memory first, may be archived later.
        """
        entry = MemoryEntry(
            id=self._generate_id(),
            content=content,
            memory_type=memory_type,
            source=source,
            importance=importance,
            metadata=metadata or {},
        )

        evicted = self.working.add(entry)

        # If something was evicted from working memory, archive it
        if evicted and evicted.importance >= 0.3:
            self.archive(evicted)

        return entry

    def archive(self, entry: MemoryEntry):
        """Move entry to long-term archival memory."""
        print(f"[memory] archiving: {entry.content[:50]}...")
        self.archival.add(entry)

    def recall(self, query: str, top_k: int = 5) -> list[MemoryEntry]:
        """
        Retrieve relevant memories for a query.
        Searches both working and archival memory.
        """
        # Search archival memory
        archival_results = self.archival.search(query, top_k=top_k)

        # Also check working memory for exact matches
        working_results = [
            e for e in self.working.entries
            if query.lower() in e.content.lower()
        ]

        # Combine and dedupe
        seen_ids = set()
        combined = []
        for entry in working_results + archival_results:
            if entry.id not in seen_ids:
                seen_ids.add(entry.id)
                combined.append(entry)

        return combined[:top_k]

    def get_context_for_prompt(self, query: str = None) -> str:
        """
        Build memory context to inject into prompts.
        Includes working memory + relevant archival memories.
        """
        sections = []

        # Working memory
        if self.working.entries:
            sections.append(self.working.get_context())

        # Relevant archival memories
        if query:
            relevant = self.archival.search(query, top_k=3)
            if relevant:
                lines = ["## Relevant Past Knowledge"]
                for entry in relevant:
                    age = self._format_age(entry.timestamp)
                    lines.append(f"- [{entry.memory_type}, {age}] {entry.content}")
                sections.append("\n".join(lines))

        return "\n\n".join(sections) if sections else ""

    def _format_age(self, timestamp: float) -> str:
        """Format timestamp as human-readable age."""
        age_seconds = time.time() - timestamp
        if age_seconds < 60:
            return "just now"
        elif age_seconds < 3600:
            return f"{int(age_seconds / 60)}m ago"
        elif age_seconds < 86400:
            return f"{int(age_seconds / 3600)}h ago"
        else:
            return f"{int(age_seconds / 86400)}d ago"

    # ─── Specialized Memory Operations ────────────────────────────────────────

    def remember_insight(self, insight: str, source: str = "analysis"):
        """Store a research insight (high importance)."""
        return self.remember(
            content=insight,
            memory_type="insight",
            source=source,
            importance=0.8,
        )

    def remember_fact(self, fact: str, source: str = "paper"):
        """Store a factual finding."""
        return self.remember(
            content=fact,
            memory_type="fact",
            source=source,
            importance=0.6,
        )

    def remember_query(self, query: str, result_summary: str):
        """Store a query and its results for future reference."""
        return self.remember(
            content=f"Query: {query} → {result_summary}",
            memory_type="query_result",
            source="retrieval",
            importance=0.4,
        )

    def remember_user_preference(self, preference: str):
        """Store user preference (high importance, persists)."""
        entry = self.remember(
            content=preference,
            memory_type="user_preference",
            source="user",
            importance=0.9,
        )
        # Immediately archive user preferences
        self.archive(entry)
        return entry

    # ─── Memory Consolidation ─────────────────────────────────────────────────

    def consolidate(self):
        """
        Consolidate similar memories to reduce redundancy.
        Uses LLM to merge related memories.
        """
        if len(self.archival.entries) < 10:
            return  # Not enough to consolidate

        # Group by type
        by_type = {}
        for entry in self.archival.entries:
            by_type.setdefault(entry.memory_type, []).append(entry)

        for memory_type, entries in by_type.items():
            if len(entries) < 5:
                continue

            # Ask LLM to identify duplicates/mergeable memories
            contents = [e.content for e in entries[:20]]
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                max_tokens=500,
                messages=[
                    {"role": "system", "content": "Identify which of these memories are duplicates or can be merged. Return JSON: {\"groups\": [[indices to merge], ...], \"summaries\": [\"merged summary\", ...]}"},
                    {"role": "user", "content": json.dumps(contents)}
                ]
            )

            # Apply merges (simplified - would need more robust parsing)
            print(f"[memory] consolidation for {memory_type}: {response.choices[0].message.content[:100]}...")

    # ─── Session Management ───────────────────────────────────────────────────

    def end_session(self):
        """
        End current session.
        Archives important working memory items.
        """
        for entry in self.working.entries:
            if entry.importance >= 0.5:
                self.archive(entry)
        self.working.clear()

    def get_stats(self) -> dict:
        """Get memory statistics."""
        return {
            "working_memory_count": len(self.working.entries),
            "archival_memory_count": len(self.archival.entries),
            "archival_by_type": {
                t: len([e for e in self.archival.entries if e.memory_type == t])
                for t in set(e.memory_type for e in self.archival.entries)
            }
        }


# ─── Global Memory Instance ───────────────────────────────────────────────────

_memory: Optional[MemoryManager] = None


def get_memory() -> MemoryManager:
    """Get or create the global memory manager."""
    global _memory
    if _memory is None:
        _memory = MemoryManager()
    return _memory


def remember(content: str, **kwargs) -> MemoryEntry:
    """Convenience function to add to memory."""
    return get_memory().remember(content, **kwargs)


def recall(query: str, top_k: int = 5) -> list[MemoryEntry]:
    """Convenience function to recall from memory."""
    return get_memory().recall(query, top_k)
