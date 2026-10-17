from pathlib import Path
from fastapi import FastAPI, Request, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from src import assistant
from src.data import load_clean, ROOT
from src.predict import predict_price, metrics
from src.rag import kb, DOCS_DIR
from src.recommender import recommend

BASE = Path(__file__).parent
app = FastAPI(title="AI Real Estate Intelligence Platform")
app.mount("/static", StaticFiles(directory=BASE / "static"), name="static")
app.mount("/plots", StaticFiles(directory=ROOT / "outputs" / "plots"), name="plots")
templates = Jinja2Templates(directory=BASE / "templates")
LOCATIONS = sorted(load_clean()["location"].unique())

class PredictIn(BaseModel):
    location: str
    area_sqft: float = Field(gt=100, lt=20000)
    bedrooms: int = Field(ge=1, le=10)
    bathrooms: int = Field(ge=1, le=10)
    age_years: float = Field(ge=0, le=100)
    parking: int = Field(ge=0, le=5)
    furnished: int = Field(ge=0, le=1)
    near_metro: int = Field(ge=0, le=1)
    year_listed: int = 2025

class RecommendIn(BaseModel):
    budget: float = Field(gt=0)
    location: str | None = None
    bedrooms: int | None = None
    min_area: float | None = None
    prefs: list[str] = []

class ChatIn(BaseModel):
    message: str = Field(min_length=1, max_length=2000)
    history: list[dict] = []
    use_docs: bool = True

def page(request: Request, name: str, **ctx):
    return templates.TemplateResponse(request, name, {"locations": LOCATIONS, "active": name.split(".")[0], **ctx})

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return page(request, "home.html", metrics=metrics())

@app.get("/predict", response_class=HTMLResponse)
def predict_page(request: Request):
    return page(request, "predict.html")

@app.get("/recommend", response_class=HTMLResponse)
def recommend_page(request: Request):
    return page(request, "recommend.html")

@app.get("/chat", response_class=HTMLResponse)
def chat_page(request: Request):
    return page(request, "chat.html", docs=kb.documents())

@app.post("/api/predict")
def api_predict(p: PredictIn):
    if p.location not in LOCATIONS:
        raise HTTPException(422, "Unknown location")
    return predict_price(**p.model_dump())

@app.post("/api/recommend")
def api_recommend(r: RecommendIn):
    return {"results": recommend(r.budget, r.location or None, r.bedrooms, r.min_area, r.prefs)}

@app.post("/api/chat")
def api_chat(c: ChatIn):
    try:
        return assistant.chat(c.message, c.history, c.use_docs)
    except Exception as e:
        raise HTTPException(502, f"AI service error: {e}")

@app.post("/api/upload")
async def upload(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".txt", ".md", ".pdf")):
        raise HTTPException(400, "Only .txt, .md and .pdf files are supported")
    (DOCS_DIR / Path(file.filename).name).write_bytes(await file.read())
    kb.rebuild()
    return {"documents": kb.documents()}

@app.get("/api/health")
def health():
    return {"status": "ok", "model": metrics()["best_model"], "documents": kb.documents()}
