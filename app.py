import os
import json
from datetime import datetime, timezone
from urllib import request as urlrequest

from flask import Flask, render_template, request
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


@app.route("/", methods=["GET"])
def index():
    # Default-Werte für das Formular
    values = {
        "selectedTone": "friendly",
        "corporateSignature": "Ihr NOVOTERGUM Team",
        "contactEmail": "",
        "languageMode": "de",
    }
    # Mindestens ein leerer Block für die erste Bewertung
    return render_template("index.html", values=values, reviews=[{}], replies=None)


@app.route("/generate", methods=["POST"])
def generate():
    form = request.form

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

        response = client.chat.completions.create(
            model="gpt-4.1-mini",  # ggf. Modell anpassen
            messages=[{"role": "user", "content": prompt}],
        )
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
                # Fire-and-forget – Response wird nicht weiterverarbeitet
                urlrequest.urlopen(req, timeout=3)

            except Exception as e:
                app.logger.warning("Make webhook failed: %s", e)

    # Falls gar keine Bewertung im POST war, trotzdem einen leeren Block anzeigen
    if not review_blocks:
        review_blocks = [{}]

    return render_template(
        "index.html",
        values=values,
        reviews=review_blocks,
        replies=replies,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port)
