from flask import Flask, request, render_template
from generate_prompt import build_prompt
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = Flask(__name__)


@app.route("/", methods=["GET"])
def form():
    # Defaultwerte für globale Felder
    values = {
        "selectedTone": "friendly",
        "languageMode": "de",
        "corporateSignature": "Ihr NOVOTERGUM Team",
        "contactEmail": "",
    }

    # Mindestens ein leerer Bewertungsblock
    reviews = [
        {"review": "", "rating": "", "reviewType": "", "salutation": ""}
    ]

    return render_template(
        "index.html",
        values=values,
        reviews=reviews,
        replies=None,
    )


@app.route("/generate", methods=["POST"])
def generate():
    form_data = request.form

    # Globale Einstellungen (gelten für alle Antworten)
    values = {
        "selectedTone": form_data.get("selectedTone", "friendly"),
        "languageMode": form_data.get("languageMode", "de"),
        "corporateSignature": form_data.get("corporateSignature", ""),
        "contactEmail": form_data.get("contactEmail", ""),
    }

    # Mehrere Bewertungen einsammeln (gleicher Feldname, mehrere Werte)
    text_list = form_data.getlist("review")
    rating_list = form_data.getlist("rating")
    type_list = form_data.getlist("reviewType")
    salutation_list = form_data.getlist("salutation")

    reviews = []
    for idx, text in enumerate(text_list):
        text = (text or "").strip()
        if not text:
            continue

        review_data = {
            "review": text,
            "rating": rating_list[idx] if idx < len(rating_list) else "",
            "reviewType": type_list[idx] if idx < len(type_list) else "",
            "salutation": salutation_list[idx] if idx < len(salutation_list) else "",
        }
        reviews.append(review_data)

    replies = []

    for review_data in reviews:
        # Für die Prompt-Erzeugung Review-spezifische Daten + globale Optionen zusammenführen
        prompt_input = {}
        prompt_input.update(values)
        prompt_input.update(review_data)

        prompt = build_prompt(prompt_input)

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )

        reply_text = response.choices[0].message.content

        replies.append(
            {
                "review": review_data["review"],
                "rating": review_data["rating"],
                "reviewType": review_data["reviewType"],
                "salutation": review_data["salutation"],
                "reply": reply_text,
            }
        )

    # Wenn alle Textfelder leer waren, trotzdem einen leeren Block anzeigen
    if not reviews:
        reviews = [{"review": "", "rating": "", "reviewType": "", "salutation": ""}]

    return render_template(
        "index.html",
        values=values,
        reviews=reviews,
        replies=replies,
    )


if __name__ == "__main__":
    app.run(debug=True)
