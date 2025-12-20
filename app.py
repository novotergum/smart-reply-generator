# app.py
import os
import json
import time
import secrets
from typing import Any, Dict, Optional, Tuple, List

import psycopg2
from flask import Flask, render_template, request, jsonify, redirect, url_for, abort
from openai import OpenAI
from dotenv import load_dotenv

from generate_prompt import build_prompt, split_public_and_insights

load_dotenv()
app = Flask(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_REVIEWS = 10
PREFILL_SECRET = os.getenv("PREFILL_SECRET", "")
DATABASE_URL = os.getenv("DATABASE_URL", "")
PREFILL_TTL_SECONDS = int(os.getenv("PREFILL_TTL_SECONDS", "259200"))  # 3 Tage


def env_truthy(v: str) -> bool:
    return str(os.getenv(v, "")).strip().lower() in ("1", "true", "yes", "on")


ENABLE_PUBLISH = env_truthy("ENABLE_PUBLISH")
PUBLISH_UI_ENABLED = env_truthy("PUBLISH_UI_ENABLED")
PUBLISH_DRY_RUN = env_truthy("PUBLISH_DRY_RUN")


# -------------------- DB --------------------

def pg_connect():
    if not DATABASE_URL:
        raise RuntimeError("DATABASE_URL missing")
    return psycopg2.connect(DATABASE_URL, sslmode="prefer", connect_timeout=5)


def prefill_init():
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS prefill (
                    rid TEXT PRIMARY KEY,
                    payload JSONB NOT NULL,
                    created_at BIGINT NOT NULL,
                    used_at BIGINT,
                    used_count INT DEFAULT 0,
                    generated JSONB,
                    generated_at BIGINT
                )
            """)
        conn.commit()


def prefill_insert(payload: dict) -> str:
    rid = secrets.token_urlsafe(18)
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO prefill (rid,payload,created_at) VALUES (%s,%s,%s)",
                (rid, json.dumps(payload, ensure_ascii=False), int(time.time())),
            )
        conn.commit()
    return rid


def prefill_get_row(rid: str) -> Optional[dict]:
    if not rid:
        return None
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT payload,generated,created_at FROM prefill WHERE rid=%s", (rid,))
            row = cur.fetchone()
            if not row:
                return None

            payload_raw, generated_raw, created_at = row
            payload = json.loads(payload_raw) if isinstance(payload_raw, str) else payload_raw
            generated = json.loads(generated_raw) if isinstance(generated_raw, str) else generated_raw

            return {"payload": payload, "generated": generated, "created_at": created_at}


def prefill_set_generated(rid: str, data: dict):
    if not rid:
        return
    with pg_connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE prefill SET generated=%s,generated_at=%s WHERE rid=%s",
                (json.dumps(data, ensure_ascii=False), int(time.time()), rid),
            )
        conn.commit()


# -------------------- Reviewer suffix dedupe --------------------

def _suffix_line(name: str, date: str) -> str:
    parts = []
    if (name or "").strip():
        parts.append(name.strip())
    if (date or "").strip():
        parts.append(f"am {date.strip()}")
    if not parts:
        return ""
    return "— " + ", ".join(parts)

CORS_ALLOW_ORIGINS = [o.strip() for o in os.getenv("CORS_ALLOW_ORIGINS", "").split(",") if o.strip()]

def _dedupe_reviewer(text: str, reviewer: str, reviewed_at: str) -> str:
    text = (text or "").strip()
    if not text:
        return text
    suffix = _suffix_line(reviewer, reviewed_at)
    if not suffix:
        return text

    normalized_suffix = suffix.replace("–", "—").replace("-", "—")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    while lines and lines[-1].replace("–", "—").replace("-", "—") == normalized_suffix:
        lines.pop()

    lines.append(normalized_suffix)
    return "\n".join(lines)


# -------------------- Publishing readiness --------------------

def compute_publish_ready(payload: Dict[str, Any]) -> Tuple[bool, List[str]]:
    p = payload or {}
    missing = []
    for k in ("accountId", "locationId", "reviewId"):
        v = p.get(k)
        if v is None or str(v).strip() == "":
            missing.append(k)
    return (len(missing) == 0), missing


# -------------------- API --------------------

@app.post("/api/prefill")
def api_prefill():
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    data = request.get_json(force=True) or {}

    review = (data.get("review") or "").strip()
    reviewer = (data.get("reviewer") or "").strip()
    reviewed_at = (data.get("reviewed_at") or "").strip()
    rating = str(data.get("rating") or "").strip()

    account_id = (data.get("accountId") or data.get("account_id") or "").strip()
    location_id = (data.get("locationId") or data.get("location_id") or "").strip()
    review_id = (data.get("reviewId") or data.get("review_id") or "").strip()

    review = _dedupe_reviewer(review, reviewer, reviewed_at)

    payload = {
        "review": review,
        "reviewer": reviewer,
        "reviewed_at": reviewed_at,
        "rating": rating,

        "accountId": account_id,
        "locationId": location_id,
        "reviewId": review_id,

        "storeCode": data.get("storeCode"),
        "locationTitle": data.get("locationTitle"),

        "maps_uri": data.get("maps_uri"),
        "new_review_uri": data.get("new_review_uri"),
        "place_id": data.get("place_id"),
        "maps_place_url": data.get("maps_place_url"),
    }

    rid = prefill_insert(payload)

    ready, missing = compute_publish_ready(payload)
    app.logger.info(
        "prefill_in rid=%s ready=%s missing=%s accountId=%s locationId=%s reviewId=%s",
        rid,
        ready,
        ",".join(missing) if missing else "-",
        "Y" if payload.get("accountId") else "N",
        "Y" if payload.get("locationId") else "N",
        "Y" if payload.get("reviewId") else "N",
    )

    return jsonify({"rid": rid})


@app.route("/api/debug/prefill", methods=["GET"])
def api_debug_prefill():
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    rid = (request.args.get("rid") or "").strip()
    row = prefill_get_row(rid)
    if not row:
        return jsonify({"ok": False, "error": "rid not found"}), 404

    p = row.get("payload") or {}
    ready, missing = compute_publish_ready(p)

    return jsonify({
        "ok": True,
        "rid": rid,
        "created_at": row.get("created_at"),
        "publish_ready": ready,
        "publish_missing": missing,
        "payload_keys": sorted(list(p.keys())),
        "accountId": p.get("accountId"),
        "locationId": p.get("locationId"),
        "reviewId": p.get("reviewId"),
    })


@app.route("/api/publish", methods=["POST", "OPTIONS"])
def api_publish():
    # Preflight
    if request.method == "OPTIONS":
        return ("", 204)

    if not ENABLE_PUBLISH:
        abort(403)

    # Auth (fürs Testen ok; später besser token-basiert statt Secret im Browser)
    if request.headers.get("X-Prefill-Secret") != PREFILL_SECRET:
        abort(401)

    data = request.get_json(force=True, silent=True) or {}

    rid = (data.get("rid") or request.args.get("rid") or "").strip()
    if not rid:
        return jsonify({"ok": False, "error": "missing rid"}), 400

    row = prefill_get_row(rid)
    if not row:
        return jsonify({"ok": False, "error": "rid not found"}), 404

    payload = row.get("payload") or {}
    ready, missing = compute_publish_ready(payload)
    if not ready:
        return jsonify({"ok": False, "error": "publish not ready", "missing": missing}), 400

    reply = (data.get("reply") or "").strip()
    if not reply:
        gen = row.get("generated") or {}
        reps = (gen.get("replies") or [])
        if reps and isinstance(reps[0], dict):
            reply = (reps[0].get("reply") or "").strip()

    if not reply:
        return jsonify({"ok": False, "error": "missing reply text"}), 400

    if PUBLISH_DRY_RUN:
        app.logger.info(
            "publish_dry_run rid=%s accountId=%s locationId=%s reviewId=%s reply_len=%s",
            rid,
            payload.get("accountId"),
            payload.get("locationId"),
            payload.get("reviewId"),
            len(reply),
        )
        return jsonify({"ok": True, "dry_run": True})

    return jsonify({"ok": False, "error": "publishing not implemented"}), 501

@app.after_request
def add_cors_headers(resp):
    origin = request.headers.get("Origin")

    # Nur für /api/* relevant
    if request.path.startswith("/api/"):
        # Wenn keine Liste gesetzt ist, nimm fail-closed (empfohlen): nichts erlauben.
        # Für Debug/Test kannst du CORS_ALLOW_ORIGINS setzen.
        if origin:
            allow = False

            # explizite Whitelist
            if CORS_ALLOW_ORIGINS and origin in CORS_ALLOW_ORIGINS:
                allow = True

            # manche iframe/sandbox Szenarien senden Origin: null
            if origin == "null" and os.getenv("CORS_ALLOW_NULL_ORIGIN", "").strip().lower() in ("1", "true", "yes", "on"):
                allow = True

            if allow:
                resp.headers["Access-Control-Allow-Origin"] = origin
                resp.headers["Vary"] = "Origin"
                resp.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
                resp.headers["Access-Control-Allow-Headers"] = "Content-Type, X-Prefill-Secret"

    return resp


# -------------------- UI --------------------

@app.route("/", methods=["GET"])
def index():
    rid = (request.args.get("rid") or "").strip()
    prefill_mode = bool(rid)

    reviews, replies = [{}], None

    publish_ready = False
    publish_missing: List[str] = []

    if rid:
        row = prefill_get_row(rid)
        if row and row.get("payload"):
            p = row["payload"]

            publish_ready, publish_missing = compute_publish_ready(p)

            reviews = [
                {
                    "review": _dedupe_reviewer(p.get("review", ""), p.get("reviewer", ""), p.get("reviewed_at", "")),
                    "rating": p.get("rating", ""),
                    "reviewType": "",
                    "salutation": "",
                }
            ]
            if row.get("generated"):
                replies = (row["generated"] or {}).get("replies")

    values = {
        "selectedTone": "friendly",
        "corporateSignature": "Ihr NOVOTERGUM Team",
        "contactEmail": "",
        "languageMode": "de",
    }

    return render_template(
        "index.html",
        values=values,
        reviews=reviews,
        replies=replies,
        rid=rid,
        prefill_mode=prefill_mode,

        publish_enabled=ENABLE_PUBLISH,
        publish_ui_enabled=PUBLISH_UI_ENABLED,

        publish_ready=publish_ready,
        publish_missing=publish_missing,
    )


def _first_non_empty_pairs(reviews: List[str], ratings: List[str]) -> List[Tuple[str, str]]:
    pairs: List[Tuple[str, str]] = []
    for idx, rev in enumerate(reviews[:MAX_REVIEWS]):
        rev_txt = (rev or "").strip()
        if not rev_txt:
            continue
        rat = (ratings[idx] if idx < len(ratings) else "") or ""
        pairs.append((rev_txt, str(rat)))
    return pairs


@app.post("/generate")
def generate():
    rid = (request.form.get("rid") or "").strip()
    prefill_mode = bool(rid)

    reviews = request.form.getlist("review")
    ratings = request.form.getlist("rating")

    values = {
        "selectedTone": request.form.get("selectedTone", "friendly"),
        "corporateSignature": request.form.get("corporateSignature", "Ihr NOVOTERGUM Team"),
        "contactEmail": request.form.get("contactEmail", ""),
        "languageMode": request.form.get("languageMode", "de"),
    }

    pairs = _first_non_empty_pairs(reviews, ratings)

    # rid => Single-Review erzwingen (auch wenn jemand via DevTools mehr Felder einschleust)
    if rid and pairs:
        if len(pairs) > 1:
            app.logger.warning("generate rid=%s received %s reviews; forcing single-review mode", rid, len(pairs))
        pairs = [pairs[0]]

    replies = []
    for (rev, rating) in pairs:
        prompt = build_prompt(
            {
                "review": rev,
                "rating": rating,
                "selectedTone": values["selectedTone"],
                "corporateSignature": values["corporateSignature"],
                "contactEmail": values["contactEmail"],
                "languageMode": values["languageMode"],
            }
        )

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )

        raw = (response.choices[0].message.content or "").strip()
        public, insights = split_public_and_insights(raw)
        replies.append({"review": rev, "reply": public, "insights": insights})

    if rid:
        prefill_set_generated(rid, {"replies": replies})
        return redirect(url_for("index", rid=rid))

    # Ohne rid: multi-review modus
    return render_template(
        "index.html",
        values=values,
        reviews=[{"review": r} for r in reviews],
        replies=replies,
        rid=rid,
        prefill_mode=prefill_mode,

        publish_enabled=ENABLE_PUBLISH,
        publish_ui_enabled=PUBLISH_UI_ENABLED,

        publish_ready=False,
        publish_missing=["accountId", "locationId", "reviewId"],
    )


# -------------------- Start --------------------

if DATABASE_URL:
    prefill_init()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
