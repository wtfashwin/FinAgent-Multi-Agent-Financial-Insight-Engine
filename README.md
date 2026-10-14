# FinAgent

FinAgent is a Python prototype for exploring transaction data, flagging unusual activity, and asking questions about analysis results.

## What it does

- A Streamlit app accepts CSV data, fits an Isolation Forest, and builds a conversational retrieval pipeline over an anomaly report.
- A separate FastAPI backend exposes upload, analysis, statistics, risk-summary, visualization, streaming, and workspace routes.
- LangGraph coordinates data, risk, and insight components. Supporting modules contain model training, local workspace storage, and profile utilities.

The Streamlit app processes data directly; it is not wired to the FastAPI routes.

## Getting started

Use Python in a virtual environment. The repository does not declare a supported Python version or lock package versions.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install python-multipart
```

On Windows, activate with `.venv\Scripts\activate`. `python-multipart` is needed by FastAPI's upload route but is missing from `requirements.txt`.

### Streamlit workflow

```bash
streamlit run app.py
```

Enter your own Groq API key in the sidebar, then upload a CSV containing an amount column: `Amount`, `amount`, `transaction_amount`, `txn_amount`, or `value`. Inspect flagged rows and ask questions about the resulting anomaly report. The app uses numeric features and excludes `Class` from model inputs when present.

The UI requires a Groq key before ingestion. Language-model calls send analysis context to the provider; model availability, account limits, and charges depend on that service. Embedding models may download on first use.

### API workflow

For the separate backend, add `GROQ_API_KEY` to an uncommitted root `.env` file and review the provider/model settings in `config.py`. From the repository root:

```bash
cd api
python main.py
```

The API defaults to port 8000. Open `http://localhost:8000/docs`, upload a CSV or JSON file through `/api/upload`, then call `/api/analyze` and inspect `/api/statistics` or `/api/risk/summary`. Provider/model defaults differ between the API and Streamlit app. The API insight component can return mock responses when provider setup is unavailable.

## Project structure

- `app.py` — standalone Streamlit CSV/anomaly/chat flow.
- `api/main.py` — REST endpoints and process-level state.
- `orchestrator.py`, `agents/` — analysis workflow and supporting components.
- `config.py` — paths, provider configuration, and API settings.
- `tests/` — component and API test source.

## Checks

The test source uses pytest and FastAPI's test client. After installing pytest and httpx in the development environment:

```bash
python -m pytest tests/
```

## Status and limitations

This is an experimental analysis system. An anomaly flag is not a confirmed fraud finding, and generated answers are not validated financial conclusions. No transaction dataset is bundled.

API data is held in process globals, while workspaces and profiles use local files. Routes do not enforce user authentication. Provider fallbacks, broad dependency ranges, model downloads, and incomplete integration need review before shared deployment. Separate security utilities are not proof of API protection or regulatory compliance.

## Contributing and license

Small changes with relevant tests and a clear pull-request description are welcome. Built with open-source tools including FastAPI, Streamlit, LangChain, LangGraph, Chroma, and scikit-learn. No license file is included.
