import os
from flask import Flask, render_template, request
from openai import OpenAI
from generate_prompt import build_prompt

app = Flask(__name__)

# OpenAI Client – API-Key kommt aus der Umgebung / .env
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

MAX_REVIEWS = 10


def derive_rating_category(stars: str) -> str:
    """Leitet aus der Sternanzahl eine Kategorie ab: positiv / neutral / kritisch."""
    try:
        v = int(stars)
    except (TypeError, ValueError):
        return ""
    if v >= 4:
        return "positiv"
    if v == 3:
        return "neutral"
    return "kritisch"


def map_review_tag(review_type: str) -> str:
    """
    Mapping der Auswahl im Formular auf die in prompt.xml verwendeten Tags.
    """
    mapping = {
        "therapy": "Therapie/Behandlung",
        "service": "Service/Erreichbarkeit",
        "staff": "Allgemeines Feedback",
        "organisation": "Allgemeines Feedback",
        "critical": "Kritische Inhalte",
        "other": "Allgemeines Feedback",
    }
    return mapping.get((review_type or "").strip(), "Allgemeines Feedback")


@app.route("/", methods=["GET"])
def index():
    # Default-Werte für das Formular
    values = {
        "selectedTone": "friendly",
        "corporateSignature": "Ihr NOVOTERGUM Team",
        "contactEmail": "",
        "languageMode": "de",
    }
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

    # Hard-Limit serverseitig
    reviews_list = reviews_list[:MAX_REVIEWS]
    ratings_list = ratings_list[:MAX_REVIEWS]
    types_list = types_list[:MAX_REVIEWS]
    sal_list = sal_list[:MAX_REVIEWS]

    replies = []
    # Daten für das Re-Rendern des Formulars
    review_blocks = []

    base = {
        "selectedTone": values["selectedTone"],
        "languageMode": values["languageMode"],
        "corporateSignature": values["corporateSignature"],
        "kontaktMail": values["contactEmail"],
    }

    for idx, review_text in enumerate(reviews_list):
        rating_raw = ratings_list[idx] if idx < len(ratings_list) else ""
        rtype = types_list[idx] if idx < len(types_list) else ""
        sal = sal_list[idx] if idx < len(sal_list) else ""

        # Für das Template
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

        data = dict(base)
        data["review"] = review_text
        data["ratingStars"] = rating_raw
        data["ratingCategory"] = derive_rating_category(rating_raw)
        data["reviewTag"] = map_review_tag(rtype)
        if sal:
            data["anrede"] = sal

        prompt = build_prompt(data)

        response = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        reply_text = (response.choices[0].message.content or "").strip()

        replies.append({"review": review_text, "reply": reply_text})

    # Falls der Nutzer z.B. weniger als 1 Bewertung hatte, trotzdem mind. einen Block anzeigen
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
