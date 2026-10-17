# AI Real Estate Intelligence Platform

Price prediction · explainable recommendations · GenAI advisor · RAG over property documents, served with FastAPI.

## Architecture
```
data/            raw + cleaned dataset, RAG documents (data/docs)
src/data.py      cleaning & preprocessing        src/analysis.py   EDA + plots
src/train_model  LR / Decision Tree / Random Forest comparison, best saved with joblib
src/predict.py   inference + error range         src/recommender   scored, explainable ranking
src/rag.py       chunking + TF-IDF retrieval      src/assistant.py  prompt engineering + RAG + market stats -> Claude
app/             FastAPI routes + Jinja2 pages (predict, recommend, chat)
```

## Run it
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/generate_data.py      # or drop in a real CSV with the same columns
python -m src.analysis            # cleans data, saves plots to outputs/plots
python -m src.train_model         # trains + saves models/price_model.joblib
export ANTHROPIC_API_KEY=your_key # optional: without it the chatbot runs in retrieval-only demo mode
uvicorn app.main:app --reload     # open http://127.0.0.1:8000
```
API docs: http://127.0.0.1:8000/docs

## Design decisions
- **Model selection:** three regressors compared on a held-out 20% test set (MAE, RMSE, R²); the best is saved.
- **Explainable recommender:** each property gets a 0-100 score from budget fit, bedrooms, preferences, and value vs the ML fair price; every point of the score maps to a human-readable reason.
- **Prompt engineering:** a system prompt with role, rules, grounding requirement, citation format and length limit; context passed in XML-tagged blocks (`<market_data>`, `<knowledge_base>`).
- **RAG:** documents chunked by paragraph, indexed with TF-IDF, top-3 chunks injected into the prompt, sources shown in the UI. Users can upload `.txt/.md/.pdf` files live.
- **Graceful degradation:** works without an API key (retrieval-only mode).

## Possible extensions
Real dataset (Kaggle), embeddings + vector DB (FAISS/Chroma), Docker, deployment on Render/Railway, user accounts and saved searches.
