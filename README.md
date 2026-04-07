# Prior

A research assistant that searches arXiv, analyzes papers, and synthesizes literature reviews.

![Searching papers](imgs/searchingpapers.png)

## What it does

Give it a research question, and it will:

1. Break it into focused sub-queries
2. Search arXiv + your local paper database
3. Extract claims, methods, and findings from each paper
4. Synthesize everything into a structured literature review

## Setup

```bash
# Clone and enter the project
git clone https://github.com/your-username/prior.git
cd prior

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up your environment
cp .env.example .env
# Edit .env with your keys
```

### Environment variables

Create a `.env` file:

```
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql://user:pass@localhost:5432/prior
```

### Database

You'll need PostgreSQL with the `pgvector` extension:

```bash
# macOS
brew install postgresql pgvector

# Create the database
createdb prior

# The app will create tables on first run
```

## Usage

### Command line

```bash
# Run a query
python main.py "What are the latest advances in neural architecture search?"

# Interactive mode
python main.py -i

# Initialize database
python main.py --init-db
```

### API server

```bash
# Start the server
uvicorn server:app --reload

# Or just
python server.py
```

Then make requests:

```bash
# Streaming (real-time progress)
curl -X POST http://localhost:8000/analyze \
  -H "Content-Type: application/json" \
  -d '{"question": "What are advances in neural architecture search?"}' \
  --no-buffer

# Synchronous (wait for result)
curl -X POST http://localhost:8000/analyze/sync \
  -H "Content-Type: application/json" \
  -d '{"question": "What are advances in neural architecture search?"}'
```

### Seeding the database

To pre-populate your local database with papers:

```bash
python seed.py
```

Edit `seed.py` to change the seed queries.

## Project structure

```
prior/
├── agents/           # LangGraph nodes
│   ├── planner.py    # Breaks question into sub-queries
│   ├── retrieval.py  # Searches arXiv + local DB
│   ├── analysis.py   # Extracts claims from papers
│   └── synthesis.py  # Produces final report
├── core/
│   ├── graph.py      # LangGraph pipeline
│   ├── state.py      # Type definitions
│   └── events.py     # SSE event system
├── db/
│   └── vector.py     # PostgreSQL + pgvector
├── main.py           # CLI entry point
├── server.py         # FastAPI server
└── seed.py           # Database seeder
```

## Output format

The final report includes:

- **Executive summary** - Quick overview of the field
- **Key claims** - Main findings with supporting papers
- **Contested claims** - Where papers disagree
- **Methodology breakdown** - Types of papers found
- **Open problems** - Gaps in the literature
- **Suggested queries** - What to search next

![Example result](imgs/result1.png)

![Example result continued](imgs/result2.png)

## API docs

See [API.md](API.md) for the full API documentation including SSE event types and frontend integration examples.

## Memory System

Prior includes a MemGPT-style memory system that learns across sessions:

- **Working memory** - Recent context within a session
- **Archival memory** - Persisted findings, methods, and open problems

![Memory compression](imgs/memorycompression.png)

## Notes

- arXiv queries are rate-limited (built-in delays)
- The analysis step is parallelized for speed
- Papers are cached in PostgreSQL to avoid re-fetching
