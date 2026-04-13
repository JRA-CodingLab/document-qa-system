"""Document loaders for PDF, Markdown, and plain-text files."""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document


SUPPORTED_EXTENSIONS = {".pdf", ".md", ".markdown", ".txt"}


class DocumentLoaderError(Exception):
    """Raised when a document cannot be loaded."""


# ── Individual loaders ────────────────────────────────────────────────


def load_pdf(file_path: Path) -> List[Document]:
    """Load a PDF file, returning one ``Document`` per page."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise DocumentLoaderError(f"File not found: {file_path}")

    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(str(file_path))
        pages = loader.load()
        for page in pages:
            page.metadata["source"] = str(file_path)
            page.metadata["file_type"] = "pdf"
        return pages
    except Exception as exc:
        if isinstance(exc, DocumentLoaderError):
            raise
        raise DocumentLoaderError(f"Failed to load PDF {file_path}: {exc}") from exc


def load_markdown(file_path: Path) -> List[Document]:
    """Load a Markdown file as a single ``Document``."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise DocumentLoaderError(f"File not found: {file_path}")

    try:
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_type"] = "markdown"
        return docs
    except Exception as exc:
        if isinstance(exc, DocumentLoaderError):
            raise
        raise DocumentLoaderError(
            f"Failed to load Markdown {file_path}: {exc}"
        ) from exc


def load_text(file_path: Path) -> List[Document]:
    """Load a plain-text file as a single ``Document``."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise DocumentLoaderError(f"File not found: {file_path}")

    try:
        from langchain_community.document_loaders import TextLoader

        loader = TextLoader(str(file_path), encoding="utf-8")
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = str(file_path)
            doc.metadata["file_type"] = "text"
        return docs
    except Exception as exc:
        if isinstance(exc, DocumentLoaderError):
            raise
        raise DocumentLoaderError(
            f"Failed to load text file {file_path}: {exc}"
        ) from exc


# ── Dispatcher ───────────────────────────────────────────────────────


def load_document(file_path: Path) -> List[Document]:
    """Route to the appropriate loader based on file extension."""
    file_path = Path(file_path)
    ext = file_path.suffix.lower()

    dispatch = {
        ".pdf": load_pdf,
        ".md": load_markdown,
        ".markdown": load_markdown,
        ".txt": load_text,
    }

    loader_fn = dispatch.get(ext)
    if loader_fn is None:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise DocumentLoaderError(
            f"Unsupported file type '{ext}'. Supported types: {supported}"
        )
    return loader_fn(file_path)


# ── Directory loader ─────────────────────────────────────────────────


def load_directory(
    dir_path: Path,
    extensions: Optional[List[str]] = None,
) -> List[Document]:
    """Load all supported files from a directory.

    Parameters
    ----------
    dir_path:
        Directory to scan.
    extensions:
        Optional whitelist of extensions (e.g. ``[".txt", ".md"]``).
        Normalised and intersected with *SUPPORTED_EXTENSIONS*.
    """
    dir_path = Path(dir_path)
    if not dir_path.exists():
        raise DocumentLoaderError(f"Directory not found: {dir_path}")
    if not dir_path.is_dir():
        raise DocumentLoaderError(f"Path is not a directory: {dir_path}")

    if extensions is not None:
        normalised = set()
        for ext in extensions:
            ext = ext.lower()
            if not ext.startswith("."):
                ext = f".{ext}"
            normalised.add(ext)
        target_exts = normalised & SUPPORTED_EXTENSIONS
    else:
        target_exts = SUPPORTED_EXTENSIONS

    all_docs: List[Document] = []
    for ext in sorted(target_exts):
        for fp in sorted(dir_path.glob(f"*{ext}")):
            try:
                all_docs.extend(load_document(fp))
            except DocumentLoaderError as exc:
                warnings.warn(f"Skipping {fp}: {exc}")

    return all_docs
