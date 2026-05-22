# Deploy notes

This project does not deploy on its own. The notes below show what changes if you point it at real Gemini, Vertex AI, or Cloud Run. None of the steps below have been run against a billable Google Cloud project as part of this submission.

## 1. Swap the fake provider for real Gemini (3-line change)

In `examples/batch_summarize.py`, the `_select_provider` function already checks for `GEMINI_API_KEY`. Set the env var and the script switches automatically. If you want to force the real provider in code:

```python
import os
from prod_gemini_agent import GeminiClient, ProductionAgent

provider = GeminiClient(api_key=os.environ["GEMINI_API_KEY"])
agent = ProductionAgent(provider=provider)
```

That is it. Every governance layer above the provider (retry, breaker, budget, cache, trace) sees the same `GeminiResult` shape, so nothing else has to change.

Install the optional extra:

```
pip install 'prod-gemini-agent[gemini]'
```

## 2. Vertex AI provider (15-line change)

Vertex AI uses Application Default Credentials and a project + location pair instead of an API key. Drop this class into your project (it satisfies the same provider contract as `GeminiClient`):

```python
import time
from prod_gemini_agent import GeminiResult, ProviderError

class VertexGeminiClient:
    def __init__(self, *, project: str, location: str, model: str = "gemini-2.0-flash") -> None:
        from vertexai import init                                # type: ignore
        from vertexai.generative_models import GenerativeModel   # type: ignore
        init(project=project, location=location)
        self._model = GenerativeModel(model)
        self._model_name = model

    def call(self, prompt: str) -> GeminiResult:
        started = time.perf_counter()
        try:
            resp = self._model.generate_content(prompt)
        except Exception as exc:
            msg = str(exc).lower()
            retryable = any(t in msg for t in ("429", "503", "deadline"))
            raise ProviderError(str(exc), retryable=retryable) from exc
        usage = getattr(resp, "usage_metadata", None)
        in_tok = getattr(usage, "prompt_token_count", 0) if usage else 0
        out_tok = getattr(usage, "candidates_token_count", 0) if usage else 0
        return GeminiResult(
            text=getattr(resp, "text", ""),
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=(time.perf_counter() - started) * 1000.0,
            model=self._model_name,
        )
```

Authenticate locally with `gcloud auth application-default login`. In Cloud Run, the service account attached to the revision picks up Vertex permissions automatically when you grant `roles/aiplatform.user`.

## 3. Cloud Run deploy outline

The repository ships as a library plus an example script, so the easiest containerization is a tiny FastAPI service that wraps `ProductionAgent.run`. A minimal `Dockerfile`:

```
FROM python:3.12-slim
WORKDIR /app
COPY pyproject.toml ./
COPY src/ ./src/
RUN pip install --no-cache-dir '.[gemini,vertex]' fastapi uvicorn
COPY service.py ./
ENV PORT=8080
CMD ["uvicorn", "service:app", "--host", "0.0.0.0", "--port", "8080"]
```

Where `service.py` exposes a `/summarize` POST endpoint that builds the agent on startup (so the cache is process-local) and calls `agent.run` per request. The body of the function is small enough to inline:

```python
from fastapi import FastAPI
from pydantic import BaseModel
from prod_gemini_agent import ProductionAgent
# ...build provider, build agent...

class Batch(BaseModel):
    items: list[tuple[str, str]]

app = FastAPI()
agent = ProductionAgent(provider=provider)

@app.post("/summarize")
def summarize(batch: Batch):
    report = agent.run(batch.items)
    return report.__dict__
```

Build and push:

```
gcloud builds submit --tag gcr.io/$PROJECT/prod-gemini-agent
gcloud run deploy prod-gemini-agent \
  --image gcr.io/$PROJECT/prod-gemini-agent \
  --platform managed \
  --region us-central1 \
  --service-account agent-runner@$PROJECT.iam.gserviceaccount.com \
  --set-env-vars GEMINI_API_KEY=...   # or omit if using Vertex
```

Pin `min-instances=0` for the free tier and let Cloud Run scale to zero when idle. The first request after a cold start carries the model client init cost (a few hundred ms); after that, the in-process cache amortizes well.

**Important:** Cloud Run is billable beyond the free tier. None of these commands were run against a real project as part of this submission.

## 4. Audit log persistence in Cloud Run

The default `agent.write_audit_log` writes JSONL to local disk. In Cloud Run, swap that for either:

- Append-to-GCS via `google-cloud-storage`. A few extra lines on top of `RunTrace.write_jsonl`.
- Direct ship to Cloud Logging structured logs (`print(json.dumps(record))` and let Cloud Run pick it up). This is the path with the least new dependency.

Either way the JSONL shape stays the same so downstream BigQuery loaders do not change.

## 5. Production checklist

- [ ] Set `BudgetWindow.cap_usd` to a real number for your workload.
- [ ] Move `ResponseCache` to Redis or Memorystore for cross-instance hits.
- [ ] Wire `RunTrace.write_jsonl` to Cloud Storage or BigQuery.
- [ ] Add an exporter that pushes `trace.snapshot()` to Cloud Monitoring custom metrics.
- [ ] Decide whether you want to keep `FakeGeminiProvider` as a smoke-test fallback in staging (recommended; it lets you exercise the breaker and retry paths without burning real tokens).
