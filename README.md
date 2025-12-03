# Smart Reply Generator für Bewertungen

Kleine Flask-App, die aus Bewertungs-Texten (z. B. Google Reviews) automatisch passende Antwortvorschläge
im NOVOTERGUM-Stil mit Hilfe der OpenAI API erzeugt. Oberfläche und Prompt-Logik sind auf Deutsch optimiert.

## Projektstruktur

```text
.
├── app.py
├── generate_prompt.py
├── prompt.xml
├── requirements.txt
├── README.md
├── .gitignore
└── templates
    └── index.html
