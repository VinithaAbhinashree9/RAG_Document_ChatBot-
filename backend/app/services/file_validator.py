"""Upload validation and filename sanitisation.

Defence in depth for untrusted uploads:
  * extension allow-list (only formats with a registered extractor)
  * size ceiling enforced on the real byte count
  * filename sanitisation that defeats path traversal and NUL injection
  * magic-byte inspection to catch a payload whose bytes contradict its extension
  * rejection of executable/script content masquerading as a document
"""

from __future__ import annotations

import logging
import re
import unicodedata
from pathlib import Path, PurePosixPath

from app.config import Settings, get_settings
from app.errors import (
    EmptyDocumentError,
    FileTooLargeError,
    UnsupportedFileTypeError,
    ValidationError,
)
from app.services.extraction import registry

logger = logging.getLogger(__name__)

#: Replace anything outside this set in stored filenames.
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._\- ]+")
_MULTI_DOT = re.compile(r"\.{2,}")
MAX_FILENAME_LENGTH = 120

#: Magic-byte signatures for formats that have them.
_SIGNATURES: dict[str, tuple[bytes, ...]] = {
    ".pdf": (b"%PDF",),
    ".docx": (b"PK\x03\x04",),
    ".xlsx": (b"PK\x03\x04",),
    ".xlsm": (b"PK\x03\x04",),
    ".pptx": (b"PK\x03\x04",),
    ".pptm": (b"PK\x03\x04",),
    ".odt": (b"PK\x03\x04",),
    ".ods": (b"PK\x03\x04",),
    ".odp": (b"PK\x03\x04",),
    ".epub": (b"PK\x03\x04",),
    ".rtf": (b"{\\rtf",),
    # Legacy OLE2 containers; a zip header is also tolerated (mislabelled modern file).
    ".doc": (b"\xd0\xcf\x11\xe0", b"PK\x03\x04", b"{\\rtf"),
    ".xls": (b"\xd0\xcf\x11\xe0", b"PK\x03\x04"),
    ".ppt": (b"\xd0\xcf\x11\xe0", b"PK\x03\x04"),
}

#: Byte prefixes that indicate an executable, never a document.
_EXECUTABLE_SIGNATURES: tuple[tuple[bytes, str], ...] = (
    (b"\x7fELF", "Linux executable"),
    (b"MZ", "Windows executable"),
    (b"\xca\xfe\xba\xbe", "Java class file"),
    (b"\xfe\xed\xfa", "macOS executable"),
    (b"#!/", "shell script"),
    (b"\x1f\x8b", "gzip archive"),
    (b"Rar!", "RAR archive"),
    (b"7z\xbc\xaf", "7-Zip archive"),
)


def sanitize_filename(filename: str) -> str:
    """Return a safe basename, stripped of any path components."""
    if not filename or not filename.strip():
        raise ValidationError("A filename is required.")

    # Reject NUL bytes outright — they can truncate paths in lower layers.
    if "\x00" in filename:
        raise ValidationError("The filename contains invalid characters.")

    # Strip directory components from both POSIX and Windows style paths.
    candidate = filename.replace("\\", "/")
    candidate = PurePosixPath(candidate).name

    candidate = unicodedata.normalize("NFKD", candidate)
    candidate = _UNSAFE_CHARS.sub("_", candidate)
    candidate = _MULTI_DOT.sub(".", candidate).strip(" .")

    if not candidate:
        raise ValidationError("The filename is not valid after sanitisation.")

    stem = Path(candidate).stem[: MAX_FILENAME_LENGTH]
    suffix = Path(candidate).suffix[:20]
    safe = f"{stem}{suffix}"

    if safe in {".", "..", ""}:
        raise ValidationError("The filename is not valid.")
    return safe


def get_extension(filename: str) -> str:
    return Path(filename).suffix.lower()


def validate_extension(filename: str) -> str:
    """Confirm an extractor exists for this extension."""
    extension = get_extension(filename)
    if not extension:
        raise UnsupportedFileTypeError(
            "The file has no extension, so its format cannot be determined. "
            f"Supported formats: {', '.join(registry.supported_extensions)}"
        )
    if registry.get(extension) is None:
        raise UnsupportedFileTypeError(
            f"'{extension}' files are not supported. Supported formats: "
            f"{', '.join(registry.supported_extensions)}"
        )
    return extension


def validate_size(size_bytes: int, settings: Settings | None = None) -> None:
    settings = settings or get_settings()
    if size_bytes <= 0:
        raise EmptyDocumentError("The uploaded file is empty.")
    if size_bytes > settings.max_upload_bytes:
        raise FileTooLargeError(
            f"The file is {size_bytes / 1_048_576:.1f} MB, which exceeds the "
            f"{settings.max_upload_mb} MB limit."
        )


def validate_content(head: bytes, extension: str) -> None:
    """Inspect leading bytes for signature mismatches and executables."""
    if not head:
        raise EmptyDocumentError("The uploaded file is empty.")

    for signature, label in _EXECUTABLE_SIGNATURES:
        if head.startswith(signature):
            # `.doc`/`.xls`/`.ppt` legitimately start with an OLE2 header, and
            # some text formats can begin with `#!`; only block real mismatches.
            if extension in {".doc", ".xls", ".ppt"} and head.startswith(b"\xd0\xcf\x11\xe0"):
                continue
            if signature == b"#!/" and extension in {".txt", ".sh", ".py", ".md", ".log"}:
                continue
            raise UnsupportedFileTypeError(
                f"The file appears to be a {label}, not a document. Upload rejected."
            )

    expected = _SIGNATURES.get(extension)
    if expected and not any(head.startswith(signature) for signature in expected):
        raise UnsupportedFileTypeError(
            f"The file contents do not match the '{extension}' format. The file may "
            "be corrupted or have the wrong extension."
        )


def validate_upload(
    filename: str, size_bytes: int, head: bytes, settings: Settings | None = None
) -> tuple[str, str]:
    """Run every check. Returns `(safe_filename, extension)`."""
    safe_name = sanitize_filename(filename)
    extension = validate_extension(safe_name)
    validate_size(size_bytes, settings)
    validate_content(head, extension)
    return safe_name, extension
