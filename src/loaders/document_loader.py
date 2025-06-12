import re
import unicodedata
from typing import List

from ftfy import fix_text
from langchain.schema import Document
from langchain.document_loaders import TextLoader, UnstructuredFileLoader

import spacy
from spacy.cli import download as spacy_download


MODEL = "en_core_web_sm"
try:
    nlp = spacy.load(MODEL)
except OSError:
    spacy_download(MODEL)
    nlp = spacy.load(MODEL)

if not nlp.has_pipe("sentencizer"):
    nlp.add_pipe("sentencizer", first=True)


def clean_text(text: str) -> str:
    text = fix_text(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\\n", " ")
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)

    text = (
        text.replace("“", '"')
            .replace("”", '"')
            .replace("‘", "'")
            .replace("’", "'")
            .replace("—", " - ")
            .replace("–", " - ")
            .replace("…", "...")
    )

    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n\n", text)
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(lines).strip()


def split_sentences(text: str):
    doc = nlp(text)
    return [sent.text.strip() for sent in doc.sents if sent.text.strip()]


def load_documents_and_chunk_sentences(
    docs_path: str,
    window_size: int = 3,
    overlap: int = 1
) -> List[Document]:
    """
    Splits each raw document into sentence chunks, then gives back a list of Documents
    whose page_content is a sliding window of `window_size` sentences, 
    stepping by (window_size - overlap)
    """
    raw_docs = []
    for file in docs_path.rglob("*"):
        suffix = file.suffix.lower()
        if suffix in {".txt", ".md"}:
            loader = TextLoader(str(file), encoding="utf-8")
        elif suffix in {".pdf", ".docx"}:
            loader = UnstructuredFileLoader(str(file))
        else:
            continue
        try:
            raw_docs.extend(loader.load())
        except Exception as e:
            print(f"Warning: could not load {file.name}: {e}")

    sentence_chunks: List[Document] = []
    step = window_size - overlap
    for doc in raw_docs:
        cleaned = clean_text(doc.page_content)
        sentences = split_sentences(cleaned)

        for i in range(0, len(sentences), step):
            chunk_sents = sentences[i : i + window_size]
            if not chunk_sents:
                continue

            chunk_text = " ".join(chunk_sents)
            print("sentence_chunk added to list: ", chunk_text)
            sentence_chunks.append(
                Document(page_content=chunk_text, metadata=doc.metadata)
            )

    return sentence_chunks