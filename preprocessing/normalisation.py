"""
Normalisation registry and resolver for anvay.

This module defines:
- which lexical normalisers exist per language
- which are preferred
- how user requests are resolved into a concrete normaliser

IMPORTANT:
- This file contains NO token-level logic.
- It is pure policy + resolution.
- Language-specific implementations live in bn.py / en.py / hi.py etc.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


# ---------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class Normaliser:
    """
    Represents a concrete lexical normaliser implementation.
    """
    key: str                 # e.g. "en_lemma", "bn_stem"
    language: str            # "en", "bn", "hi", etc.
    kind: str                # "lemma" | "stem"
    description: str         # human-readable description


# ---------------------------------------------------------------------
# Registry: what exists
# ---------------------------------------------------------------------

# NOTE:
# This is the single source of truth for what each language supports.
# Adding a new language or normaliser starts here.

NORMALISERS: Dict[str, Dict[str, Normaliser]] = {
    "en": {
        "lemma": Normaliser(
            key="en_lemma",
            language="en",
            kind="lemma",
            description="English lemmatization (WordNet)"
        ),
        # "stem": not available (by design, for now)
    },
    "bn": {
        "stem": Normaliser(
            key="bn_stem",
            language="bn",
            kind="stem",
            description="Bengali dictionary-based stemming"
        ),
        # "lemma": not available (yet)
    },

    "hi": {
        "lemma": Normaliser(
            key="hi_lemma",
            language="hi",
            kind="lemma",
            description="Hindi lemmatization (simplemma, experimental)"
        ),
    },

       "ta": {
        "stem": Normaliser(
            key="ta_stem",
            language="ta",
            kind="stem",
            description="Tamil dictionary-based stemming"
        ),
    },

    # Future example:
    # "hi": {
    #     "lemma": Normaliser(...),
    #     "stem": Normaliser(...),
    # }
}


# ---------------------------------------------------------------------
# Preference policy
# ---------------------------------------------------------------------

# Global preference order.
# Interpreted as: try earlier entries first.
PREFERENCE_ORDER: List[str] = ["lemma", "stem"]


# ---------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------

def resolve_normaliser(
    language: str,
    requested: Optional[str]
) -> Optional[Normaliser]:
    """
    Resolve a user-requested normalisation into a concrete Normaliser.

    Parameters
    ----------
    language : str
        Language code ("en", "bn", etc.)
    requested : str or None
        User request: "lemma", "stem", "none", or None.

    Returns
    -------
    Normaliser or None
        The applied normaliser, or None if no normalisation is used.

    Raises
    ------
    ValueError
        If the language is unsupported or the request is invalid.
    """

    if language not in NORMALISERS:
        raise ValueError(f"Unsupported language: {language}")

    available = NORMALISERS[language]

    # Explicit opt-out
    if requested in (None, "none"):
        return None

    # Explicit request
    if requested in ("lemma", "stem"):
        if requested in available:
            return available[requested]
        else:
            raise ValueError(
                f"Normalisation '{requested}' is not available for language '{language}'"
            )

    # Defensive: unknown request
    raise ValueError(f"Unknown normalisation request: {requested}")


# ---------------------------------------------------------------------
# Default resolver (lemma-first fallback)
# ---------------------------------------------------------------------

def resolve_default_normaliser(language: str) -> Optional[Normaliser]:
    """
    Resolve the preferred normaliser for a language,
    following the global lemma → stem → none policy.

    Parameters
    ----------
    language : str

    Returns
    -------
    Normaliser or None
    """

    if language not in NORMALISERS:
        raise ValueError(f"Unsupported language: {language}")

    available = NORMALISERS[language]

    for kind in PREFERENCE_ORDER:
        if kind in available:
            return available[kind]

    return None


# ---------------------------------------------------------------------
# Reporting helpers
# ---------------------------------------------------------------------

def describe_normalisation(
    normaliser: Optional[Normaliser]
) -> str:
    """
    Human-readable description for reports and logs.
    """

    if normaliser is None:
        return "None"

    return normaliser.description


def normaliser_key(
    normaliser: Optional[Normaliser]
) -> Optional[str]:
    """
    Stable machine-readable key for logging / metadata.
    """

    if normaliser is None:
        return None

    return normaliser.key
