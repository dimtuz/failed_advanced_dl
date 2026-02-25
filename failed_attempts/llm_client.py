"""
Local LLM client for NYC neighborhood feature engineering.
Uses Ollama (e.g. llama3.1:8b). Run: ollama run llama3.1:8b
"""
import json
import re


def ollama_available() -> bool:
    """True if Ollama is running and at least one model is available."""
    try:
        import ollama
        resp = ollama.list()
        return bool(getattr(resp, "models", None))
    except Exception:
        return False


def _pick_model(preferred: str = "llama3.1:8b") -> str:
    """Return preferred model if available, else first listed model."""
    import ollama
    resp = ollama.list()
    models = getattr(resp, "models", None) or []
    names = [m.model for m in models if getattr(m, "model", None)]
    for n in names:
        if preferred in n or (preferred.split(":")[0] in n if ":" in preferred else preferred in n):
            return n
    return names[0] if names else preferred


def _parse_mappings_json(text: str) -> dict:
    """Extract and parse JSON with 'mappings' array from LLM output."""
    text = text.strip()
    # Strip markdown code block if present
    m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    if m:
        text = m.group(1)
    data = json.loads(text)
    mappings = data.get("mappings") or data.get("mapping") or []
    if not isinstance(mappings, list):
        raise ValueError("Expected 'mappings' array")
    result = {}
    for item in mappings:
        if not isinstance(item, dict):
            continue
        name = item.get("original_name")
        sub = item.get("sub_region", "Unknown")
        aff = item.get("affluence_score", 5)
        if name is not None:
            try:
                aff = int(aff)
            except (TypeError, ValueError):
                aff = 5
            aff = max(1, min(10, aff))
            result[str(name)] = {"sub_region": str(sub), "affluence_score": aff}
    return result


def get_neighborhood_data(neighborhoods: list[str], model: str | None = None) -> dict:
    """
    Use local Ollama to map NYC neighborhoods to sub_region and affluence_score.
    Returns {neighborhood_name: {"sub_region": str, "affluence_score": int}}
    """
    import ollama

    if model is None:
        model = _pick_model()

    system_prompt = (
        "You are a NYC real estate analyst. Map each neighborhood to a broader sub-region "
        "(e.g., Lower Manhattan, Upper East Side, Brooklyn Heights) and an affluence score 1-10 "
        "(1=least affluent, 10=most affluent). Use original_name exactly as given."
    )
    user_prompt = (
        f"Return a JSON object with a 'mappings' array. Each item: original_name, sub_region, affluence_score (1-10). "
        f"Neighborhoods: {', '.join(neighborhoods)}"
    )

    response = ollama.chat(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        options={"temperature": 0},
    )
    content = getattr(response.message, "content", None) or ""
    return _parse_mappings_json(content)
