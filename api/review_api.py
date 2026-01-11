from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="NOVOTERGUM Review API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://ticket.novotergum.de"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# TODO: später durch echte Datenquelle ersetzen
REVIEWS = {
    "PkfWgyVx8p_-nZXu5WtJWRJp": "Ich habe drei Termine vereinbart, die alle per E-Mail am nächsten Tag abgesagt wurden."
}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/review-by-rid")
def review_by_rid(rid: str):
    text = REVIEWS.get(rid)
    if not text:
        raise HTTPException(status_code=404, detail="Review not found")
    return {"review_text": text}
