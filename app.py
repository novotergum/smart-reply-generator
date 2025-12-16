import os
import json
import time
import secrets
from datetime import datetime, timezone
from urllib import request as urlrequest

import psycopg2
from psycopg2.extras import Json

from flask import Flask, render_template, request, jsonify, abort
from openai import OpenAI
from dotenv import load_dotenv

from generate_prompt import build_prompt, split_public_and_insights

load_dotenv()
app = Flask(__name__)

# OpenAI
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Limits / config
MAX_REVIEWS = 10
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "").strip()

PREFILL_SECRET = os.getenv("PREFILL_SECRET", "").strip()
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))  # 3 Tage
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()

# Optional: falls du CORS explizit brauchst (z.B. Ticket-Domain)
# Beispiel: PREFILL_CORS_ORIGINS=https://ticket.novotergum.de,https://smart-reply-generator-production2.up.railway.app
PREFILL_CORS_ORIGINS = [
    o.strip() for o in os.getenv("PREFILL_CORS_ORIGINS", "").split(",") if o.strip()
]

# DB connect timeout (wichtig gegen Gunicorn worker timeout)
PG_CONNECT_TIMEOUT = int(os.getenv("PGCONNECT_TIMEOUT", "3"))
PG_SSLMODE = os.getenv("PGSSLMODE", "prefer")


def pg_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing (Railway Postgres variable reference required)")
    return psycopg2.connect(
        DATABASE_URL,
        sslmode=PG_SSLMODE,
        connect_timeout=PG_CONNECT_TIMEOUT,
    )


_prefill_table_ready = False


def prefill_init():
    """Create table if not exists. Must never block long."""
    global _prefill_table_ready
    if _prefill_table_ready:
        return True

    try:
        with pg_connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS prefill (
                      rid TEXT PRIMARY KEY,
                      payload JSONB NOT NULL,
                      created_at BIGINT NOT NULL
                    )
                    """
                )
            conn.commit()
        _prefill_table_ready = True
        return True
    except Exception as e:
        app.logger.error("Prefill init failed: %s", e)
        return False


def prefill_cleanup():
    if not prefill_init():
        return
    cutoff = int(time.time()) - PREFILL_TTL_SECONDS
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM prefill WHERE created_at < %s", (cutoff,))
        conn.commit()


def prefill_insert(payload: dict) -> str:
    if not prefill_init():
        raise RuntimeError("DB not ready")
    rid = secrets.token_urlsafe(18)
    now = int(time.time())

    # Cleanup should never make the request hang forever -> connect_timeout guards it
    prefill_cleanup()

    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prefill (rid, payload, created_at) VALUES (%s, %s, %s)",
                (rid, Json(payload), now),
            )
        conn.commit()
    return rid


def prefill_get(rid: str):
    if not rid:
        return None
    if not prefill_init():
        return None

    prefill_cleanup()

    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM prefill WHERE rid = %s", (rid,))
            row = cur.fetchone()
            if not row:
                return None
            return row[0]


def prefill_delete(rid: str):
    if not rid:
        return
    if not prefill_init():
        return
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM prefill WHERE rid = %s", (rid,))
        conn.commit()


def _cors_maybe(resp):
    """Only add CORS when explicitly configured."""
    origin = request.headers.get("Origin")
    if origin and PREFILL_CORS_ORIGINS and origin in PREFILL_CORS_ORIGINS:
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Prefill-Secret"
    return resp


@app.after_request
def after_request(resp):
    # Only relevant for /api/prefill (optional)
    if request.path.startswith("/api/prefill"):
        return _cors_maybe(resp)
    return resp


@app.route("/api/prefill", methods=["OPTIONS"])
def api_prefill_options():
    resp = jsonify({"ok": True})
    resp.status_code = 204
    return resp


@app.post("/api/prefill")
def api_prefill_create():
    if not PREFILL_SECRET:
        return jsonify({"error": "PREFILL_SECRET not configured"}), 500
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    # Support JSON and form payloads (Zapier kann beides)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        data = request.form.to_dict(flat=True)

    review = (data.get("review") or "").strip()
    rating = (str(data.get("rating") or "")).strip()

    reviewer = (data.get("reviewer") or "").strip()
    reviewed_at = (data.get("reviewed_at") or "").strip()

    if not review or len(review) > 8000:
        return jsonify({"error": "invalid review"}), 400

    # rating: allow empty; accept 1..5; if upstream sometimes sends 0 -> treat as empty
    if rating:
        try:
            r = int(float(rating))  # tolerates "5.0" etc.
            if r == 0:
                rating = ""
            elif r < 1 or r > 5:
                return jsonify({"error": "invalid rating"}), 400
            else:
                rating = str(r)
        except Exception:
            return jsonify({"error": "invalid rating"}), 400

    payload = {
        "review": review,
        "rating": rating,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
    }

    try:
        rid = prefill_insert(payload)
    except Exception as e:
        app.logger.error("Prefill insert failed: %s", e)
        # Wichtig: schnell fehlschlagen statt hängen -> vermeidet Gunicorn timeouts
        return jsonify({"error": "db unavailable"}), 503

    resp = jsonify({"rid": rid})
    resp.headers["Cache-Control"] = "no-store"
    return resp


@app.route("/", methods=["GET"])
def index():
    rid = (request.args.get("rid") or "").strip()

    values = {
        "selectedTone": "friendly",
        "corporateSignature": "Ihr NOVOTERGUM Team",
        "contactEmail": "",
        "languageMode": "de",
    }

    reviews = [{}]

    if rid:
        try:
            payload = prefill_get(rid)
        except Exception as e:
            app.logger.warning("Prefill get failed: %s", e)
            payload = None

        if payload:
            # Reviewer/Datetime in das Kommentarfeld "review" schreiben (wie dein Beispiel)
            review_text = (payload.get("review") or "").strip()

            reviewer = (payload.get("reviewer") or "").strip()
            reviewed_at = (payload.get("reviewed_at") or "").strip()

            if reviewer or reviewed_at:
                tail = "— " + (reviewer or "Unbekannt")
                if reviewed_at:
                    tail += f", am {reviewed_at}"
                review_text = f"{review_text}\n\n{tail}"

            reviews = [{
                "review": review_text,
                "rating": (payload.get("rating") or ""),
                "reviewType": "",
                "salutation": "",
            }]

    return render_template("index.html", values=values, reviews=reviews, replies=None, rid=rid)


@app.route("/generate", methods=["POST"])
def generate():
    form = request.form
    rid = (form.get("rid") or "").strip()

    # Token nach Nutzung verbrauchen
    if rid:
        try:
            prefill_delete(rid)
        except Exception as e:
            app.logger.warning("Prefill delete failed: %s", e)

    values = {
        "selectedTone": form.get("selectedTone", "friendly"),
        "corporateSignature": form.get("corporateSignature", "Ihr NOVOTERGUM Team"),
        "contactEmail": form.get("contactEmail", "").strip(),
        "languageMode": form.get("languageMode", "de"),
    }

    reviews_list = [t.strip() for t in form.getlist("review")]
    ratings_list = form.getlist("rating")
    types_list = form.getlist("reviewType")
    sal_list = form.getlist("salutation")

    reviews_list = reviews_list[:MAX_REVIEWS]
    ratings_list = ratings_list[:MAX_REVIEWS]
    types_list = types_list[:MAX_REVIEWS]
    sal_list = sal_list[:MAX_REVIEWS]

    replies = []
    review_blocks = []

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

        review_blocks.append({
            "review": review_text,
            "rating": rating_raw,
            "reviewType": rtype,
            "salutation": sal,
        })

        if not review_text:
            continue

        data = dict(base)
        data["review"] = review_text
        data["rating"] = rating_raw
        data["reviewType"] = rtype
        data["salutation"] = sal

        prompt = build_prompt(data)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        raw_reply = (response.choices[0].message.content or "").strip()
        public_answer, insights = split_public_and_insights(raw_reply)

        replies.append({
            "review": review_text,
            "reply": public_answer,
            "insights": insights,
        })

        # Insights an Make senden (optional)
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
                data_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
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

    # rid nach Submit leeren (damit nicht weiterverwendet wird)
    return render_template("index.html", values=values, reviews=review_blocks, replies=replies, rid="")


# Init nicht hart failen lassen; nur versuchen
prefill_init()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
