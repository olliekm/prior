# Prior API Documentation

Base URL: `http://localhost:8000`

## Endpoints

### POST `/analyze` (Streaming)

Run a research analysis with real-time progress updates via Server-Sent Events (SSE).

**Request:**
```json
{
  "question": "What are the latest advances in neural architecture search?"
}
```

**Response:** `text/event-stream`

Each event is a JSON object with the format:
```
data: {"type": "event_type", "data": {...}, "timestamp": 1234567890.123}
```

---

### POST `/analyze/sync` (Non-Streaming)

Run analysis and wait for the complete result.

**Request:**
```json
{
  "question": "What are the latest advances in neural architecture search?"
}
```

**Response:**
```json
{
  "question": "What are the latest advances in neural architecture search?",
  "sub_queries": ["neural architecture search methods", "NAS benchmarks", ...],
  "papers_count": 30,
  "claims_count": 28,
  "report": {
    "executive_summary": "...",
    "key_claims": [...],
    "contested_claims": [...],
    "methodology_breakdown": {...},
    "open_problems": [...],
    "suggested_queries": [...]
  }
}
```

---

### GET `/health`

Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

---

## SSE Event Types

Events are streamed in order as the analysis progresses through each stage.

### Stage Events

| Event | Description | Data |
|-------|-------------|------|
| `stage_start` | A pipeline stage is starting | `{stage, message, ?total}` |
| `stage_end` | A pipeline stage completed | `{stage, count}` |

### Planner Stage

| Event | Description | Data |
|-------|-------------|------|
| `planning` | Starting to decompose the question | `{question}` |
| `sub_queries` | Generated sub-queries | `{queries: string[]}` |

### Retrieval Stage

| Event | Description | Data |
|-------|-------------|------|
| `query_complete` | A sub-query search finished | `{query, papers_found, progress, total}` |
| `papers_found` | All papers retrieved and filtered | `{raw, deduped, filtered}` |

### Analysis Stage

| Event | Description | Data |
|-------|-------------|------|
| `paper_complete` | A paper was analyzed | `{title, progress, total}` |
| `paper_failed` | Paper analysis failed | `{title, error, progress, total}` |

### Synthesis Stage

| Event | Description | Data |
|-------|-------------|------|
| `synthesizing` | Starting synthesis | `{claims_count}` |

### Final Events

| Event | Description | Data |
|-------|-------------|------|
| `complete` | Analysis finished successfully | `{question, sub_queries, papers_count, claims_count, report}` |
| `error` | Analysis failed | `{error}` |

---

## Event Flow Example

```
stage_start     → {stage: "planner", message: "Decomposing research question..."}
planning        → {question: "What are advances in NAS?"}
sub_queries     → {queries: ["neural architecture search methods", ...]}
stage_end       → {stage: "planner", count: 5}

stage_start     → {stage: "retrieval", message: "Searching for papers..."}
query_complete  → {query: "neural architecture search methods", papers_found: 20, progress: 1, total: 5}
query_complete  → {query: "NAS benchmarks evaluation", papers_found: 18, progress: 2, total: 5}
...
papers_found    → {raw: 100, deduped: 45, filtered: 30}
stage_end       → {stage: "retrieval", count: 30}

stage_start     → {stage: "analysis", message: "Analyzing papers...", total: 30}
paper_complete  → {title: "DARTS: Differentiable Architecture Search", progress: 1, total: 30}
paper_complete  → {title: "EfficientNet: Rethinking Model Scaling", progress: 2, total: 30}
...
stage_end       → {stage: "analysis", count: 28}

stage_start     → {stage: "synthesis", message: "Synthesizing findings..."}
synthesizing    → {claims_count: 28}
stage_end       → {stage: "synthesis"}

complete        → {question: "...", report: {...}, ...}
```

---

## Frontend Integration Examples

### JavaScript/TypeScript (Fetch API)

```typescript
interface PriorEvent {
  type: string;
  data: Record<string, any>;
  timestamp: number;
}

async function analyzeWithStreaming(
  question: string,
  onEvent: (event: PriorEvent) => void
): Promise<void> {
  const response = await fetch('http://localhost:8000/analyze', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });

  const reader = response.body!.getReader();
  const decoder = new TextDecoder();
  let buffer = '';

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });
    const lines = buffer.split('\n');
    buffer = lines.pop() || '';

    for (const line of lines) {
      if (line.startsWith('data: ')) {
        const event: PriorEvent = JSON.parse(line.slice(6));
        onEvent(event);
      }
    }
  }
}

// Usage
analyzeWithStreaming('What are advances in NAS?', (event) => {
  switch (event.type) {
    case 'stage_start':
      console.log(`Starting: ${event.data.message}`);
      break;
    case 'paper_complete':
      console.log(`Analyzed ${event.data.progress}/${event.data.total}: ${event.data.title}`);
      break;
    case 'complete':
      console.log('Report:', event.data.report);
      break;
    case 'error':
      console.error('Error:', event.data.error);
      break;
  }
});
```

### React Hook

```typescript
import { useState, useCallback } from 'react';

interface AnalysisState {
  status: 'idle' | 'running' | 'complete' | 'error';
  stage: string | null;
  progress: { current: number; total: number } | null;
  message: string | null;
  report: any | null;
  error: string | null;
}

export function usePriorAnalysis() {
  const [state, setState] = useState<AnalysisState>({
    status: 'idle',
    stage: null,
    progress: null,
    message: null,
    report: null,
    error: null,
  });

  const analyze = useCallback(async (question: string) => {
    setState(s => ({ ...s, status: 'running', error: null, report: null }));

    try {
      const response = await fetch('http://localhost:8000/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question }),
      });

      const reader = response.body!.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const event = JSON.parse(line.slice(6));

            switch (event.type) {
              case 'stage_start':
                setState(s => ({
                  ...s,
                  stage: event.data.stage,
                  message: event.data.message,
                  progress: event.data.total ? { current: 0, total: event.data.total } : null,
                }));
                break;
              case 'paper_complete':
              case 'query_complete':
                setState(s => ({
                  ...s,
                  progress: { current: event.data.progress, total: event.data.total },
                }));
                break;
              case 'complete':
                setState(s => ({
                  ...s,
                  status: 'complete',
                  report: event.data.report,
                  stage: null,
                  progress: null,
                }));
                break;
              case 'error':
                setState(s => ({
                  ...s,
                  status: 'error',
                  error: event.data.error,
                }));
                break;
            }
          }
        }
      }
    } catch (err) {
      setState(s => ({
        ...s,
        status: 'error',
        error: err instanceof Error ? err.message : 'Unknown error',
      }));
    }
  }, []);

  return { ...state, analyze };
}

// Usage in component
function AnalysisPage() {
  const { status, stage, progress, message, report, error, analyze } = usePriorAnalysis();
  const [question, setQuestion] = useState('');

  return (
    <div>
      <input value={question} onChange={e => setQuestion(e.target.value)} />
      <button onClick={() => analyze(question)} disabled={status === 'running'}>
        Analyze
      </button>

      {status === 'running' && (
        <div>
          <p>{message}</p>
          {progress && <progress value={progress.current} max={progress.total} />}
        </div>
      )}

      {status === 'complete' && <pre>{JSON.stringify(report, null, 2)}</pre>}
      {status === 'error' && <p style={{ color: 'red' }}>{error}</p>}
    </div>
  );
}
```

### Python Client

```python
import requests
import json

def analyze_streaming(question: str):
    """Stream analysis events."""
    response = requests.post(
        'http://localhost:8000/analyze',
        json={'question': question},
        stream=True,
        headers={'Accept': 'text/event-stream'}
    )

    for line in response.iter_lines():
        if line:
            line = line.decode('utf-8')
            if line.startswith('data: '):
                event = json.loads(line[6:])
                yield event

# Usage
for event in analyze_streaming('What are advances in NAS?'):
    print(f"[{event['type']}]", event['data'])
```

### cURL

```bash
# Streaming
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "What are advances in neural architecture search?"}' \
  --no-buffer

# Synchronous
curl -X POST http://localhost:8000/analyze/sync \
  -H "Content-Type: application/json" \
  -d '{"question": "What are advances in neural architecture search?"}'
```

---

## Report Schema

The final report in the `complete` event has this structure:

```typescript
interface Report {
  executive_summary: string;

  key_claims: Array<{
    claim: string;
    supporting_papers: string[];
  }>;

  contested_claims: Array<{
    claim: string;
    side_a: string[];
    side_b: string[];
    reason: string;
  }>;

  methodology_breakdown: {
    empirical: number;
    theoretical: number;
    survey: number;
    benchmark: number;
    system: number;
  };

  open_problems: string[];

  suggested_queries: string[];
}
```

---

## Error Handling

- **400 Bad Request**: Empty question
- **500 Internal Server Error**: Pipeline failure (details in `error` event)

Always handle the `error` event type when streaming:

```typescript
if (event.type === 'error') {
  console.error('Analysis failed:', event.data.error);
}
```

---

## CORS

The API allows all origins by default. For production, configure allowed origins in `server.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend.com"],
    ...
)
```
