"""Shared helpers for the Agent's chat-facing request parsing."""

from __future__ import annotations

#: Property packages the DWSIM exporter accepts, in canonical spelling.
CANONICAL_PROPERTY_PACKAGES: tuple[str, ...] = (
    "NRTL",
    "UNIQUAC",
    "Wilson",
    "UNIFAC",
    "UNIFAC-LL",
    "Modified UNIFAC (Dortmund)",
)

#: Default when the user names none.
DEFAULT_PROPERTY_PACKAGE = "NRTL"

#: Lower-cased spellings (and common Chinese names) mapped to the canonical key.
PROPERTY_PACKAGE_ALIASES: dict[str, str] = {
    "nrtl": "NRTL",
    "uniquac": "UNIQUAC",
    "unique": "UNIQUAC",
    "wilson": "Wilson",
    "unifac": "UNIFAC",
    "unifac-ll": "UNIFAC-LL",
    "unifac ll": "UNIFAC-LL",
    "modified unifac": "Modified UNIFAC (Dortmund)",
    "modified unifac (dortmund)": "Modified UNIFAC (Dortmund)",
    "dortmund": "Modified UNIFAC (Dortmund)",
}


def resolve_property_package(message: str, default: str = DEFAULT_PROPERTY_PACKAGE) -> str:
    """Return the property package named in ``message``.

    Falls back to ``default`` (NRTL) when nothing recognisable is present, so an
    unspecified request keeps the repository's standard package.
    """
    lower = message.casefold()
    # Longest alias first so "modified unifac" is not shadowed by "unifac".
    for alias in sorted(PROPERTY_PACKAGE_ALIASES, key=len, reverse=True):
        if alias in lower:
            return PROPERTY_PACKAGE_ALIASES[alias]
    return default


def is_known_property_package(name: str) -> bool:
    return name in CANONICAL_PROPERTY_PACKAGES
