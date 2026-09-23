"""
Document upload/extraction utilities for the admin console's "train the bot"
feature. Supports .docx (Word) and .pdf files.
"""

import io
import re


def extract_text_from_file(file_bytes, filename):
    """Extracts raw text from an uploaded .docx or .pdf file."""
    if filename.endswith(".docx"):
        return _extract_from_docx(file_bytes)
    elif filename.endswith(".pdf"):
        return _extract_from_pdf(file_bytes)
    else:
        raise ValueError("Unsupported file type. Please upload a .docx or .pdf file.")


def _extract_from_docx(file_bytes):
    from docx import Document  # python-docx
    doc = Document(io.BytesIO(file_bytes))
    paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    return "\n".join(paragraphs)


def _extract_from_pdf(file_bytes):
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(file_bytes))
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages)


def chunk_text_into_qa_pairs(text, sentences_per_chunk=2):
    """
    Splits raw extracted text into chunks and stores each as a knowledge-base
    entry. Since uploaded documents aren't already in Q&A format, we use the
    chunk itself as both the searchable "question" text and the "answer" --
    the retrieval logic just needs something to match against and return.
    """
    sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

    entries = []
    for i in range(0, len(sentences), sentences_per_chunk):
        chunk = " ".join(sentences[i:i + sentences_per_chunk])
        if len(chunk) > 20:  # skip tiny/empty fragments
            entries.append({"question": chunk, "answer": chunk})

    return entries
