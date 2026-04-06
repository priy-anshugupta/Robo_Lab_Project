"""
ISL-Multilingual Bridge — Prompt Templates
Centralized prompt management for GPT-4.1 Mini integration.
"""

# ─── Stage 1: ISL Gloss → Grammatical English ─────────────────────────

ISL_TO_ENGLISH_SYSTEM = """\
You are an expert Indian Sign Language (ISL) interpreter with deep knowledge \
of ISL grammar and linguistics.

Key ISL grammar rules:
- ISL uses Topic-Comment word order (object before verb)
- Time expressions come first ("TOMORROW I GO" = "I will go tomorrow")
- Pronouns often omitted when context is clear
- No articles (a, an, the) in ISL
- Questions marked by facial expression, not word order

Your task: Convert the raw ISL sign sequence (gloss notation) into a \
natural, grammatically correct English sentence. Preserve exact meaning.
Output ONLY the English sentence, nothing else.\
"""

ISL_TO_ENGLISH_USER = "ISL gloss sequence: {gloss}"


# ─── Stage 2: English → Target Language Translation ───────────────────

TRANSLATION_SYSTEM = """\
You are a professional translator specializing in Indian languages.
Translate the given English text to {language} naturally and accurately.
Preserve the tone and intent of the original.
Output ONLY the {language} translation. No romanization, no explanation.\
"""

TRANSLATION_USER = "Translate to {language}: {english_text}"


# ─── Context Enrichment (for ambiguous/incomplete signs) ──────────────

CONTEXT_ENRICHMENT_SYSTEM = """\
You are interpreting Indian Sign Language in real-time. Given a sequence of \
recognized signs that may contain errors or missing context, produce the most \
likely intended meaning as a natural English sentence.
Consider common ISL conversational patterns. If uncertain, produce the most \
plausible interpretation.
Output ONLY the interpreted English sentence.\
"""

CONTEXT_ENRICHMENT_USER = "Signs recognized (may contain errors): {signs}"


# ─── Utility: Format templates ────────────────────────────────────────

def format_isl_to_english(gloss: str) -> dict:
    """
    Format the ISL → English prompt pair.

    Args:
        gloss: Space-separated ISL gloss string (e.g., "HELLO MY NAME PRIYA")

    Returns:
        dict with 'system' and 'user' prompt strings
    """
    return {
        "system": ISL_TO_ENGLISH_SYSTEM,
        "user": ISL_TO_ENGLISH_USER.format(gloss=gloss),
    }


def format_translation(language: str, english_text: str) -> dict:
    """
    Format the English → target language translation prompt pair.

    Args:
        language: Target language name (e.g., "Hindi")
        english_text: English sentence to translate

    Returns:
        dict with 'system' and 'user' prompt strings
    """
    return {
        "system": TRANSLATION_SYSTEM.format(language=language),
        "user": TRANSLATION_USER.format(language=language, english_text=english_text),
    }


def format_context_enrichment(signs: str) -> dict:
    """
    Format the context enrichment prompt pair.

    Args:
        signs: Space-separated sign sequence (may contain errors)

    Returns:
        dict with 'system' and 'user' prompt strings
    """
    return {
        "system": CONTEXT_ENRICHMENT_SYSTEM,
        "user": CONTEXT_ENRICHMENT_USER.format(signs=signs),
    }
