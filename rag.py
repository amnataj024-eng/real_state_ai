"""Lightweight RAG: chunk documents -> TF-IDF index -> cosine-similarity retrieval.
The retrieved chunks are injected into the LLM prompt (see assistant.py)."""
import re
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from src.data import ROOT

DOCS_DIR = ROOT / "data" / "docs"

def _read(path: Path) -> str:
    if path.suffix.lower() == ".pdf":
        from pypdf import PdfReader
        return "\n".join(p.extract_text() or "" for p in PdfReader(str(path)).pages)
    return path.read_text(encoding="utf-8", errors="ignore")

def _chunk(text: str, size: int = 700) -> list[str]:
    paras, chunks, cur = [p.strip() for p in re.split(r"\n\s*\n|\n(?=#)", text) if p.strip()], [], ""
    for p in paras:
        if len(cur) + len(p) > size and cur:
            chunks.append(cur); cur = p
        else:
            cur = f"{cur}\n{p}".strip()
    if cur:
        chunks.append(cur)
    return chunks

class KnowledgeBase:
    def __init__(self):
        self.rebuild()

    def rebuild(self):
        self.chunks = []
        for f in sorted(DOCS_DIR.glob("*")):
            if f.suffix.lower() in {".md", ".txt", ".pdf"}:
                self.chunks += [{"source": f.name, "text": c} for c in _chunk(_read(f))]
        self.vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks]) if self.chunks else None

    def search(self, query: str, k: int = 3, min_score: float = 0.05) -> list[dict]:
        if self.matrix is None:
            return []
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        top = sims.argsort()[::-1][:k]
        return [{**self.chunks[i], "score": round(float(sims[i]), 3)} for i in top if sims[i] >= min_score]

    def documents(self) -> list[str]:
        return sorted({c["source"] for c in self.chunks})

kb = KnowledgeBase()
