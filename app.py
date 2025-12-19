import os
import json
import time
import secrets
import re
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
    return psycopg2.connect(
        DATABASE_URL,
        sslmode=os.getenv("PGSSLMODE", "prefer"),
        connect_timeout=int(os.getenv("PG_CONNECT_TIMEOUT", "5")),
    )


def prefill_init():
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS prefill (
                  rid TEXT PRIMARY KEY,
                  payload JSONB NOT NULL,
                  created_at BIGINT NOT NULL,
                  used_at BIGINT,
                  used_count INT NOT NULL DEFAULT 0
                )
                """
            )
            # Falls Tabelle schon existiert (ältere Version), fehlende Spalten nachziehen
            cur.execute("ALTER TABLE prefill ADD COLUMN IF NOT EXISTS used_at BIGINT;")
            cur.execute(
                "ALTER TABLE prefill ADD COLUMN IF NOT EXISTS used_count INT NOT NULL DEFAULT 0;"
            )
        conn.commit()


def prefill_cleanup():
    cutoff = int(time.time()) - PREFILL_TTL_SECONDS
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM prefill WHERE created_at < %s", (cutoff,))
        conn.commit()


def _strip_trailing_meta_lines(text: str) -> str:
    """
    Entfernt am Ende des Textes typische Abbinder-Zeilen wie:
    '— Name, am 18.12.2025 14:39:53'
    ' - Name, am ...'
    und auch doppelte Wiederholungen davon.
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Leere Zeilen entfernen, aber innere Zeilen beibehalten
    lines = [ln.rstrip() for ln in text.splitlines()]
    # trim trailing empty lines
    while lines and not lines[-1].strip():
        lines.pop()

    # Heuristik: Meta-Zeile startet mit — oder - und enthält "am " oder ein Datumsformat
    date_hint = re.compile(r"\b\d{1,2}\.\d{1,2}\.\d{2,4}\b")
    def is_meta_line(ln: str) -> bool:
        s = ln.strip()
        if not s:
            return False
        if not re.match(r"^[—-]\s+", s):
            return False
        if " am " in s:
            return True
        if date_hint.search(s):
            return True
        return False

    # Entferne alle Meta-Zeilen am Ende (auch mehrfach)
    while lines and is_meta_line(lines[-1]):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()

    return "\n".join([ln for ln in lines]).strip()


def _compose_review_text(review_raw: str, reviewer: str = "", reviewed_at: str = "") -> str:
    """
    Baut den sichtbaren Text genau einmal zusammen:
    <review>
    — <reviewer>, am <reviewed_at>

    Verhindert zuverlässig doppelte Abbinder, auch wenn review_raw schon einen hat.
    """
    review_raw = (review_raw or "").strip()
    reviewer = (reviewer or "").strip()
    reviewed_at = (reviewed_at or "").strip()

    # Sanitization: keine Zeilenumbrüche in Meta-Feldern
    if reviewer:
        reviewer = reviewer.splitlines()[0].strip()
    if reviewed_at:
        reviewed_at = reviewed_at.splitlines()[0].strip()

    # Wenn keine Metadaten vorhanden sind, einfach raw zurückgeben
    if not reviewer and not reviewed_at:
        return review_raw

    # Sicherheit: falls raw versehentlich comment_full enthält -> Meta entfernen
    base = _strip_trailing_meta_lines(review_raw)

    suffix_parts = []
    if reviewer:
        suffix_parts.append(reviewer)
    if reviewed_at:
        suffix_parts.append(f"am {reviewed_at}")

    suffix = ", ".join([p for p in suffix_parts if p]).strip()
    meta_line = f"— {suffix}".strip()

    if base:
        return f"{base}\n{meta_line}"
    return meta_line


def prefill_insert(payload: dict) -> str:
    rid = secrets.token_urlsafe(18)
    now = int(time.time())
    prefill_cleanup()
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prefill (rid, payload, created_at, used_at, used_count) VALUES (%s, %s, %s, %s, %s)",
                (rid, json.dumps(payload, ensure_ascii=False), now, None, 0),
            )
        conn.commit()
    return rid


def prefill_get(rid: str):
    if not rid:
        return None
    # Cleanup nicht zwingend bei jedem GET, aber ok bei eurem Traffic
    prefill_cleanup()
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload FROM prefill WHERE rid = %s", (rid,))
            row = cur.fetchone()
            if not row:
                return None
            payload = row[0]
            # je nach psycopg2-config kann JSONB als dict oder str kommen
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    return None
            return payload


def prefill_mark_used(rid: str):
    if not rid:
        return
    now = int(time.time())
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET used_at = %s, used_count = used_count + 1 WHERE rid = %s",
                (now, rid),
            )
        conn.commit()


@app.post("/api/prefill")
def api_prefill_create():
    if not PREFILL_SECRET:
        return jsonify({"error": "PREFILL_SECRET not configured"}), 500
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    data = request.get_json(silent=True) or {}

    # Input: review soll RAW sein (ohne Abbinder).
    # Falls Make/Zapier aber versehentlich comment_full schickt, fangen wir es ab.
    review_in = (data.get("review") or "").strip()
    comment_full = (data.get("comment_full") or "").strip()
    if not review_in and comment_full:
        review_in = comment_full

    rating_raw = data.get("rating")

    reviewer = (data.get("reviewer") or "").strip()
    reviewed_at = (data.get("reviewed_at") or "").strip()

    if not review_in or len(review_in) > 8000:
        return jsonify({"error": "invalid review"}), 400

    rating = ""
    if rating_raw is not None and str(rating_raw).strip() != "":
        try:
            r = int(str(rating_raw).strip())
            if r < 1 or r > 5:
                return jsonify({"error": "invalid rating"}), 400
            rating = str(r)
        except Exception:
            return jsonify({"error": "invalid rating"}), 400

    # Wichtig: Wir speichern RAW + Meta separat (nicht den zusammengesetzten Text)
    review_raw = _strip_trailing_meta_lines(review_in)

    payload = {
        "review_raw": review_raw,
        "rating": rating,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        # optional: falls du es später brauchst (nicht fürs UI)
        "comment_full": comment_full or "",
    }

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
            # Neue Struktur bevorzugen
            if payload.get("review_raw") is not None:
                review_text = _compose_review_text(
                    payload.get("review_raw") or "",
                    reviewer=payload.get("reviewer") or "",
                    reviewed_at=payload.get("reviewed_at") or "",
                )
            else:
                # Legacy: alte Einträge hatten ggf. nur "review"
                review_text = (payload.get("review") or "").strip()

            reviews = [
                {
                    "review": review_text,
                    "rating": payload.get("rating") or "",
                    "reviewType": "",
                    "salutation": "",
                }
            ]

    return render_template("index.html", values=values, reviews=reviews, replies=None, rid=rid)


@app.route("/generate", methods=["POST"])
def generate():
    form = request.form
    rid = (form.get("rid") or "").strip()
    if rid:
        # NICHT löschen, nur markieren
        try:
            prefill_mark_used(rid)
        except Exception as e:
            app.logger.warning("Prefill mark_used failed: %s", e)

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

    # rid im UI nicht weitertragen (optional). DB bleibt erhalten.
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
