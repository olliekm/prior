"""Event system for real-time progress streaming."""
import queue
import threading
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Generator
import json
import time


class EventType(str, Enum):
    # Pipeline stages
    STAGE_START = "stage_start"
    STAGE_END = "stage_end"

    # Planner events
    PLANNING = "planning"
    SUB_QUERIES = "sub_queries"

    # Retrieval events
    QUERY_START = "query_start"
    QUERY_COMPLETE = "query_complete"
    PAPERS_FOUND = "papers_found"

    # Analysis events
    PAPER_START = "paper_start"
    PAPER_COMPLETE = "paper_complete"
    PAPER_FAILED = "paper_failed"

    # Reflector events
    REFLECTING = "reflecting"
    NEEDS_MORE_INFO = "needs_more_info"
    CONTEXT_SUMMARY = "context_summary"

    # Synthesis events
    SYNTHESIZING = "synthesizing"

    # Final
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class Event:
    type: EventType
    data: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_sse(self) -> str:
        """Format as Server-Sent Event."""
        payload = {
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp,
        }
        return f"data: {json.dumps(payload)}\n\n"


class EventEmitter:
    """Thread-safe event emitter for streaming progress updates."""

    _local = threading.local()

    @classmethod
    def get_current(cls) -> "EventEmitter | None":
        """Get the emitter for the current thread/context."""
        return getattr(cls._local, "emitter", None)

    @classmethod
    def set_current(cls, emitter: "EventEmitter | None"):
        """Set the emitter for the current thread/context."""
        cls._local.emitter = emitter

    def __init__(self):
        self._queue: queue.Queue[Event | None] = queue.Queue()
        self._closed = False

    def emit(self, event_type: EventType, **data):
        """Emit an event to all listeners."""
        if not self._closed:
            self._queue.put(Event(type=event_type, data=data))

    def close(self):
        """Signal that no more events will be emitted."""
        self._closed = True
        self._queue.put(None)  # Sentinel to unblock readers

    def stream(self) -> Generator[str, None, None]:
        """Yield SSE-formatted events as they arrive."""
        while True:
            event = self._queue.get()
            if event is None:
                break
            yield event.to_sse()


def emit(event_type: EventType, **data):
    """Convenience function to emit an event on the current emitter."""
    emitter = EventEmitter.get_current()
    if emitter:
        emitter.emit(event_type, **data)
