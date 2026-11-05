# UNFILED-IP
"""Schema version policy: same major is readable, anything else fails closed."""

from __future__ import annotations

SUPPORTED_MAJOR = 1


class SchemaVersionError(ValueError):
    """Raised when an artifact declares a schema this build cannot interpret."""


def check_schema_version(version: object, *, kind: str, supported_major: int = SUPPORTED_MAJOR) -> str:
    if not isinstance(version, str) or not version:
        raise SchemaVersionError(f"{kind}: schema_version missing or not a string")
    head = version.split(".", 1)[0]
    if not head.isdigit():
        raise SchemaVersionError(f"{kind}: malformed schema_version {version!r}")
    major = int(head)
    if major != supported_major:
        raise SchemaVersionError(
            f"{kind}: unsupported schema major {major} (this build reads {supported_major}.x)"
        )
    return version
