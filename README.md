# Ahoum — Conversation Evaluation Benchmark

> Score every conversation turn across **300+ facets** covering linguistic quality, pragmatics, safety, and emotion.  
> Powered entirely by **local open-weights models via Ollama** utilizing **Langchain Structured Outputs** for bulletproof JSON generation.  
> Designed to scale to **≥5,000 facets** natively without architectural changes owing to dynamic batching streams.

---

## Quick Start

### Prerequisites

1. **Install [Ollama](https://ollama.com)** and pull a model:
   ```bash
   ollama pull llama3.1:8b  # Meta Llama 3.1 8B (highly recommended for 128k context)
   ollama pull qwen2        # Qwen2 7B          
   ```

2. **Install Python deps:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure** (copy and edit):
   ```bash
   cp .env.example .env
   # Set OLLAMA_MODEL to whichever model you pulled
   ```

### Run

```bash
# API server  →  http://localhost:8000
fastapi dev src/api.py

# Streamlit UI  →  http://localhost:8501
streamlit run ui/app.py

# Generate 50 sample scored conversations
python src/generate_sample_conversations.py
```

### Docker

```bash
# Ollama stays on your host; containers reach it via host.docker.internal
OLLAMA_MODEL=llama3.1:8b docker-compose up --build
# API: http://localhost:8000
# UI:  http://localhost:8501
```

---

## Architecture

```text
Input turn(s)
     │
     ▼
┌──────────┐   facets_cleaned.csv
│  Batcher │◄── 300–5000 facets, categorised
│  N=150   │    observable_in_text flag
└────┬─────┘
     │  batches of 150 facets per Langchain call
     ▼
┌──────────┐   Langchain ChatOllama
│  Ollama  │◄── http://localhost:11434
│  Judge   │   llama3.1:8b
└────┬─────┘
     │  Strict Pydantic JSON: score 1-5 + conf + rationale
     ▼
┌──────────┐
│  Results │──► FastAPI / Streamlit UI / CLI / JSON files
└──────────┘
```

### Why batched multi-call scoring?

| Concern | Solution |
|---|---|
| **No one-shot** constraint | Each call evaluates 150 facets. 300 facets = 2 strict LLM schema inferences. |
| **≥5000 facets** | Zero redesign — the algorithm organically loops: `ceil(N / BATCH_SIZE)` |
| **Output safety** | `langchain_ollama` utilizes strict JSON-schema bounding through Pydantic Object Mapping. |
| **Confidence scores** | Every facet gets `confidence: 0.0–1.0` alongside the score |

---

## Project Structure

```
ahoum-eval/
├── data/
│   └── facets_cleaned.csv              # 300+ cleaned & categorised facets
├── src/
│   ├── scorer.py                       # Core batched scoring engine heavily optimized to 150 chunks
│   ├── api.py                          # FastAPI REST server
│   └── generate_sample_conversations.py
├── ui/
│   └── app.py                          # Streamlit UI
├── conversations/
│   └── 50 × conv_XX_theme.json         # Pre-scored sample conversations
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .env.example
└── README.md
```

---

## Facet Dataset

**Source:** `data/facets_cleaned.csv`

| Column | Description |
|---|---|
| `facet_id` | Unique integer ID |
| `facet_name` | Cleaned facet name (stripped of leading indices, trailing colons) |
| `category` | `emotion`, `safety`, `linguistic`, `cognitive`, `social`, `spiritual`, `motivation`, `other` |
| `observable_in_text` | `True/False` — whether a text-based model can score this facet |
| `score_type` | `qualitative` or `quantitative` |
| `score_1_label` … `score_5_label` | Human-readable scale anchors |

---

## Score Scale

| Score | Label | Meaning |
|---|---|---|
| 1 | Very Low / Absent | No signal in the text |
| 2 | Low | Faint or implied signal |
| 3 | Moderate | Clearly present but not dominant |
| 4 | High | Prominent — a notable characteristic of this turn |
| 5 | Very High / Dominant | Defining feature of the turn |

---

## API Reference

### `GET /facets`
```
GET /facets?category=emotion&observable_only=true
```

### `POST /score/turn`
```json
{
  "speaker": "User",
  "text": "I feel completely hopeless.",
  "batch_size": 150
}
```

### `POST /score/conversation`
```json
{
  "turns": [
    {"turn_id": 1, "speaker": "User", "text": "..."},
    {"turn_id": 2, "speaker": "Assistant", "text": "..."}
  ],
  "batch_size": 150
}
```

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama native server endpoint |
| `OLLAMA_MODEL` | `llama3.1:8b` | Model name (must be pulled first) |
| `BATCH_SIZE` | `150` | Facets per LLM call — highly optimized for large 128k context windows |

---

## Scaling to 5,000+ Facets

The architecture requires **zero structural changes** to handle unlimited facets effortlessly:

```text
300 facets  @ batch 150 = 2 calls/turn
1000 facets @ batch 150 = 7 calls/turn
5000 facets @ batch 150 = 34 calls/turn
```

When connecting to external remote providers or high-throughput local servers for huge bulk scoring, Python inherently routes the sequential network requests dynamically processing 5000 facets accurately chunk by chunk. The Streams-based UI organically catches all changes automatically inside the visualization array. 

---

## License

MIT
