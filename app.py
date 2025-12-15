import os
import json
import time
import secrets
from datetime import datetime, timezone
from urllib import request as urlrequest

import psycopg2
from flask import Flask, render_template, request, jsonify, abort
from openai import OpenAI
from dotenv import load_dotenv

from generate_prompt import build_prompt, split_public_and_insights

load_dotenv()
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_REVIEWS = 10
MAKE_WEBHOOK_URL = os.getenv("MAKE_WEBHOOK_URL", "").strip()

PREFILL_SECRET = os.getenv("PREFILL_SECRET", "").strip()
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))  # 3 Tage
DATABASE_URL = os.getenv("DATABASE_URL", "").strip()


def pg_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing (add Railway Postgres)")
    # sslmode=prefer ist auf Railway i.d.R. ok
    return psycopg2.connect(DATABASE_URL, sslmode=os.getenv("PGSSLMODE", "prefer"))


def prefill_init():
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


def prefill_cleanup():
    cutoff = int(time.time()) - PREFILL_TTL_SECONDS
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM prefill WHERE created_at < %s", (cutoff,))
        conn.commit()


def prefill_insert(payload: dict) -> str:
    rid = secrets.token_urlsafe(18)
    now = int(time.time())
    prefill_cleanup()
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prefill (rid, payload, created_at) VALUES (%s, %s, %s)",
                (rid, json.dumps(payload, ensure_ascii=False), now),
            )
        conn.commit()
    return rid


def prefill_get(rid: str):
    if not rid:
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
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM prefill WHERE rid = %s", (rid,))
        conn.commit()


@app.post("/api/prefill")
def api_prefill_create():
    if not PREFILL_SECRET:
        return jsonify({"error": "PREFILL_SECRET not configured"}), 500
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    data = request.get_json(silent=True) or {}
    review = (data.get("review") or "").strip()
    rating = (data.get("rating") or "").strip()

    if not review or len(review) > 8000:
        return jsonify({"error": "invalid review"}), 400

    if rating:
        try:
            r = int(rating)
            if r < 1 or r > 5:
                return jsonify({"error": "invalid rating"}), 400
            rating = str(r)
        except Exception:
            return jsonify({"error": "invalid rating"}), 400

    payload = {"review": review, "rating": rating}
    rid = prefill_insert(payload)

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
        payload = prefill_get(rid)
        if payload:
            reviews = [{
                "review": payload.get("review") or "",
                "rating": payload.get("rating") or "",
                "reviewType": "",
                "salutation": "",
            }]

    return render_template("index.html", values=values, reviews=reviews, replies=None, rid=rid)


@app.route("/generate", methods=["POST"])
def generate():
    form = request.form
    rid = (form.get("rid") or "").strip()
    if rid:
        try:
            prefill_delete(rid)  # Token nach Nutzung verbrauchen
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

        review_blocks.append(
            {"review": review_text, "rating": rating_raw, "reviewType": rtype, "salutation": sal}
        )

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

        replies.append({"review": review_text, "reply": public_answer, "insights": insights})

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

    return render_template("index.html", values=values, reviews=review_blocks, replies=replies, rid="")


# Tabelle beim Start sicherstellen
if DATABASE_URL:
    try:
        prefill_init()
    except Exception as e:
        app.logger.error("Prefill init failed: %s", e)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
