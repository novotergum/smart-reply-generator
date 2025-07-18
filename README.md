# Smart Reply Generator 🧠💬

Ein intelligenter Antwortgenerator für Kundenbewertungen, entwickelt mit Flask und OpenAI.

---

## ✨ Funktionen

- Automatische Formulierung höflicher, passender Antworten auf Bewertungen
- Steuerung von **Tonalität** (z. B. sachlich, warm, empathisch)
- Auswahl der **Sprache** (z. B. Deutsch, Englisch)
- Optionale Signatur einfügen
- Lokal ausführbar im Browser über eine einfache Weboberfläche

---

## 🚀 Schnellstart

### 1. Projekt klonen

```bash
git clone https://github.com/<username>/smart-reply-generator.git
cd smart-reply-generator
```
### 2. Virtuelle Umgebung erstellen (optional)

python -m venv venv
source venv/bin/activate # Windows: venv\Scripts\activate

### 3. Abhängigkeiten installieren

pip install -r requirements.txt

### 4. Umgebungsvariablen konfigurieren

Erstelle eine Datei `.env` mit folgendem Inhalt:

OPENAI_API_KEY=dein-api-key-hier

### 5. Lokale Ausführung

flask run

Rufe im Browser auf: http://127.0.0.1:5000/generate

---

## 📁 Projektstruktur

```text
smart-reply-generator/
├── main.py
├── generate_prompt.py
├── templates/
│   └── index.html
├── .env
├── prompt.xml
├── requirements.txt
└── README.md
```

## 🛡 Lizenz

MIT License

---

## 💡 Hinweise

- Das Projekt nutzt die OpenAI API zur Textgenerierung. Ein gültiger API-Schlüssel ist erforderlich.
- Die HTML-Oberfläche ist absichtlich minimal gehalten, lässt sich aber mit CSS/Bootstrap erweitern.
