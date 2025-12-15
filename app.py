import os
import json
import time
import secrets
import sqlite3
from datetime import datetime, timezone
from urllib import request as urlrequest

from flask import Flask, render_template, request, jsonify, abort
from openai import OpenAI
from dotenv import load_dotenv

from generate_prompt import build_prompt, split_public_and_insights

load_dotenv()

app = Flask(__name__)

# OpenAI Client – API-Key kommt aus der Umgebung (Railway / .env)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Serverseitiges Hard-Limit: maximal 10 Bewertungen pro Request
MAX_REVIEWS = 10

# Make-Webhook-URL (in Railway als Variable MAKE_WEBHOOK_URL setzen)
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "").strip()

# ---- Prefill (Zapier/Make -> rid -> vorbefülltes Formular) ----
PREFILL_SECRET = os.getenv("PREFILL_SECRET", "").strip()
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))  # default: 3 Tage
PREFILL_DB_PATH = os.getenv("PREFILL_DB_PATH", "prefill.db")


def _prefill_db():
    conn = sqlite3.connect(PREFILL_DB_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS prefill (
          rid TEXT PRIMARY KEY,
          payload TEXT NOT NULL,
          created_at INTEGER NOT NULL
        )
        """
    )
    return conn


def _prefill_cleanup(conn):
    cutoff = int(time.time()) - PREFILL_TTL_SECONDS
    conn.execute("DELETE FROM prefill WHERE created_at < ?", (cutoff,))
    conn.commit()


def prefill_insert(payload: dict) -> str:
    rid = secrets.token_urlsafe(18)  # ~144-bit capability token
    now = int(time.time())
    conn = _prefill_db()
    try:
        _prefill_cleanup(conn)
        conn.execute(
            "INSERT INTO prefill (rid, payload, created_at) VALUES (?, ?, ?)",
            (rid, json.dumps(payload, ensure_ascii=False), now),
        )
        conn.commit()
    finally:
        conn.close()
    return rid


def prefill_get(rid: str):
    conn = _prefill_db()
    try:
        _prefill_cleanup(conn)
        row = conn.execute("SELECT payload FROM prefill WHERE rid = ?", (rid,)).fetchone()
        if not row:
            return None
        return json.loads(row["payload"])
    finally:
        conn.close()


def prefill_delete(rid: str):
    conn = _prefill_db()
    try:
        conn.execute("DELETE FROM prefill WHERE rid = ?", (rid,))
        conn.commit()
    finally:
        conn.close()


@app.post("/api/prefill")
def api_prefill_create():
    """
    Zapier/Make ruft diesen Endpoint pro Review auf und bekommt eine rid zurück.
    Header:
      X-Prefill-Secret: <PREFILL_SECRET>
    JSON Body:
      { "review": "...", "rating": "1-5", ...optional }
    """
    if not PREFILL_SECRET:
        return jsonify({"error": "PREFILL_SECRET not configured"}), 500

    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    data = request.get_json(silent=True) or {}
    review = (data.get("review") or "").strip()
    rating = (data.get("rating") or "").strip()

    if not review or len(review) > 8000:
        return jsonify({"error": "invalid review"}), 400

    # rating optional, aber wenn gesetzt: 1–5
    if rating:
        try:
            r = int(rating)
            if r < 1 or r > 5:
                return jsonify({"error": "invalid rating"}), 400
            rating = str(r)
        except Exception:
            return jsonify({"error": "invalid rating"}), 400

    payload = {
        "review": review,
        "rating": rating,
        "reviewType": (data.get("reviewType") or "").strip(),
        "salutation": (data.get("salutation") or "").strip(),
        "selectedTone": (data.get("selectedTone") or "friendly").strip(),
        "corporateSignature": (data.get("corporateSignature") or "Ihr NOVOTERGUM Team").strip(),
        "contactEmail": (data.get("contactEmail") or "").strip(),
        "languageMode": (data.get("languageMode") or "de").strip(),
        "source": (data.get("source") or "").strip(),
        "createdAt": int(time.time()),
    }

    rid = prefill_insert(payload)
    resp = jsonify({"rid": rid})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/", methods=["GET"])
def index():
    rid = (request.args.get("rid") or "").strip()

    # Default-Werte für das Formular
    values = {
        "selectedTone": "friendly",
        "corporateSignature": "Ihr NOVOTERGUM Team",
        "contactEmail": "",
        "languageMode": "de",
    }

    # Mindestens ein leerer Block für die erste Bewertung
    reviews = [{}]

    # Wenn rid vorhanden, serverseitig prefillen
    if rid:
        payload = prefill_get(rid)
        if payload:
            values = {
                "selectedTone": payload.get("selectedTone") or "friendly",
                "corporateSignature": payload.get("corporateSignature") or "Ihr NOVOTERGUM Team",
                "contactEmail": payload.get("contactEmail") or "",
                "languageMode": payload.get("languageMode") or "de",
            }
            reviews = [
                {
                    "review": payload.get("review") or "",
                    "rating": payload.get("rating") or "",
                    "reviewType": payload.get("reviewType") or "",
                    "salutation": payload.get("salutation") or "",
                }
            ]

    return render_template("index.html", values=values, reviews=reviews, replies=None, rid=rid)


@app.route("/generate", methods=["POST"])
def generate():
    form = request.form

    # Optional: Token nach erstem Submit verbrauchen (damit Links nicht ewig funktionieren)
    rid = (form.get("rid") or "").strip()
    if rid:
        try:
            prefill_delete(rid)
        except Exception as e:
            app.logger.warning("Prefill delete failed: %s", e)

    # Globale Einstellungen (gelten für alle Bewertungen)
    values = {
        "selectedTone": form.get("selectedTone", "friendly"),
        "corporateSignature": form.get("corporateSignature", "Ihr NOVOTERGUM Team"),
        "contactEmail": form.get("contactEmail", "").strip(),
        "languageMode": form.get("languageMode", "de"),
    }

    # Mehrere Bewertungen einlesen
    reviews_list = [t.strip() for t in form.getlist("review")]
    ratings_list = form.getlist("rating")
    types_list = form.getlist("reviewType")
    sal_list = form.getlist("salutation")

    # Hard-Limit serverseitig durchsetzen
    reviews_list = reviews_list[:MAX_REVIEWS]
    ratings_list = ratings_list[:MAX_REVIEWS]
    types_list = types_list[:MAX_REVIEWS]
    sal_list = sal_list[:MAX_REVIEWS]

    replies = []
    review_blocks = []

    # Basisdaten, die für jede Bewertung identisch sind
    base = {
        "selectedTone": values["selectedTone"],
        "languageMode": values["languageMode"],
        "corporateSignature": values["corporateSignature"],
        "contactEmail": values["contactEmail"],
    }

    for idx, review_text in enumerate(reviews_list):
        rating_raw = ratings_list[idx] if idx < len(ratings_list) else ""
        rtype = types_list[idx] if idx < len(types_list) else ""
        sal = sal_list[idx] if idx < len(sal_list) else ""

        # Daten für das erneute Rendern des Formulars
        review_blocks.append(
            {
                "review": review_text,
                "rating": rating_raw,
                "reviewType": rtype,
                "salutation": sal,
            }
        )

        # Leere Bewertungen nicht an die API schicken
        if not review_text:
            continue

        # Inputs für den Prompt für genau diese Bewertung
        data = dict(base)
        data["review"] = review_text
        data["rating"] = rating_raw
        data["reviewType"] = rtype
        data["salutation"] = sal

        prompt = build_prompt(data)

        try:
            response = client.chat.completions.create(
                model="gpt-4.1-mini",  # ggf. Modell anpassen
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            app.logger.error("OpenAI API error: %s", e)
            continue

        raw_reply = (response.choices[0].message.content or "").strip()
        public_answer, insights = split_public_and_insights(raw_reply)

        replies.append(
            {
                "review": review_text,
                "reply": public_answer,
                "insights": insights,  # interne Analyse
            }
        )

        # --- Insights an Make senden (wenn konfiguriert und vorhanden) ---
        if insights and MAKE_WEBHOOK_URL:
            try:
                payload = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "review_text": review_text,
                    "rating_input": rating_raw,
                    "tone": values["selectedTone"],
                    "language": values["languageMode"],
                    "insights": insights,
                }

                data_bytes = json.dumps(payload).encode("utf-8")
                req = urlrequest.Request(
                    MAKE_WEBHOOK_URL,
                    data=data_bytes,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                urlrequest.urlopen(req, timeout=3)

            except Exception as e:
                app.logger.warning("Make webhook failed: %s", e)

    if not review_blocks:
        review_blocks = [{}]

    return render_template(
        "index.html",
        values=values,
        reviews=review_blocks,
        replies=replies,
        rid="",  # nach Submit keinen alten rid weitertragen
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
