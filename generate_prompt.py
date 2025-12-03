import xml.etree.ElementTree as ET


def render_line_text(line, data: dict) -> str:
    """
    Ersetzt Platzhalter {{ key }} in einer <line>-Zeile mit Werten aus data.
    """
    text = (line.text or "").strip()
    if not text:
        return ""

    # alle Keys, die im data-Dict vorhanden sind, dürfen verwendet werden
    for key, value in data.items():
        placeholder = "{{ " + key + " }}"
        if placeholder in text:
            text = text.replace(placeholder, str(value))
    return text


def evaluate_condition(condition: str | None, data: dict) -> bool:
    """
    Unterstützt:
    - kein condition                 -> True
    - "isset:feld"                  -> bool(data[feld])
    - "if:feld=wert"                -> data[feld] == wert
    """
    if not condition:
        return True

    if condition.startswith("isset:"):
        key = condition.split(":", 1)[1]
        val = data.get(key)
        return bool(val)

    if condition.startswith("if:"):
        expr = condition[3:]
        if "=" in expr:
            key, value = expr.split("=", 1)
            return str(data.get(key, "")).strip() == value.strip()

    return False


def build_prompt(data: dict) -> str:
    """
    Baut den Prompt auf Basis der prompt.xml und der übergebenen Daten.
    """
    tree = ET.parse("prompt.xml")
    root = tree.getroot()
    lines: list[str] = []

    for line in root.findall("line"):
        condition = line.attrib.get("condition")
        if evaluate_condition(condition, data):
            rendered = render_line_text(line, data)
            if rendered:
                lines.append(rendered)

    # Am Ende immer die Originalbewertung anhängen
    review_text = (data.get("review") or "").strip()
    if review_text:
        lines.append("Originalbewertung (unverändert):")
        lines.append(review_text)

    return "\n\n".join(lines)
