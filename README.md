# Ticket Router — Text Classification API

Classifies text into one of 5 categories (business, entertainment, politics, sport, tech),
simulating a support-ticket / content routing system. Built end-to-end: data → model → API → monitoring.

## Project structure

```
ticket-router/
  data/
    bbc-text.csv          # training data (BBC News dataset, 2225 articles)
    predictions.db         # SQLite log of every prediction made (created at runtime)
  models/
    model.joblib            # trained Logistic Regression classifier
    vectorizer.joblib       # fitted TF-IDF vectorizer
  notebooks/
    01_explore_data.py     # data exploration
    02_baseline_model.py   # training script
  api/
    main.py                # FastAPI app
  Dockerfile
  requirements.txt
```

## Results

TF-IDF + Logistic Regression baseline: **98.65% test accuracy** on held-out data.
See `notebooks/02_baseline_model.py` output for the full classification report.

Known limitation: the model was trained on full-length news articles (median 337 words).
It's noticeably less confident on short, informal text (e.g. single sentences) — this is
visible live via the `/stats` endpoint's `low_confidence_rate`. Worth fixing in v2 (see below).

## Running locally

```bash
pip install -r requirements.txt

# retrain the model from scratch (optional — model.joblib is already included)
python notebooks/02_baseline_model.py

# start the API
uvicorn api.main:app --reload --port 8000
```

Then visit `http://localhost:8000/docs` for interactive API docs, or:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "The team won the championship after a thrilling final match."}'

curl http://localhost:8000/stats
```

## Running with Docker

```bash
docker build -t ticket-router .
docker run -p 8000:8000 ticket-router
```

## Deploying (free options)

Any of these work well for a project like this:

- **Railway** (railway.app) — connect your GitHub repo, it auto-detects the Dockerfile, deploys on push.
- **Fly.io** — `fly launch` in this directory, follow prompts (it reads the Dockerfile automatically).
- **Hugging Face Spaces** (Docker SDK) — push this repo to a new Space, select "Docker" as the SDK.

All three give you a public URL and free tiers sufficient for a portfolio project.

## Monitoring

Every prediction is logged to `data/predictions.db` (SQLite) with timestamp, predicted category,
confidence score, and latency. The `/stats` endpoint aggregates this into:

- `avg_confidence` — trending down over time can signal data drift
- `low_confidence_rate` — % of predictions below 0.6 confidence; a spike means the model is
  seeing inputs unlike its training data
- `category_distribution` — sudden skew toward one category can indicate an upstream bug
  or a genuine shift in what people are submitting
- `avg_latency_ms` — catches performance regressions

For a real dashboard (not just raw JSON), point Grafana at the SQLite file, or build a small
Streamlit app that polls `/stats` periodically.

## Next steps (v2 ideas)

1. **Fix the short-text weakness**: augment training data with shorter snippets, or add a
   "review needed" flag for predictions under a confidence threshold instead of trusting them blindly.
2. **Compare against a transformer** (e.g. DistilBERT fine-tuned) — will it actually beat 98.65%,
   or is the classical model already near the ceiling for this dataset? Worth finding out.
3. **Add authentication** to the API before any real deployment (currently fully open).
4. **Add a `/feedback` endpoint** so users can flag wrong predictions — this becomes new
   labeled training data for retraining.
5. **Track model versions**: log which model version made each prediction, so you can compare
   performance across versions in `/stats`.
