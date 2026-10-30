"""
index_status.py — Deteksi index yang sudah usang terhadap dokumen sumbernya.

Masalah yang diselesaikan:
    Index dibangun dari dokumen (PDF/Markdown) pada satu titik waktu. Bila dokumen diperbarui,
    index lama tetap terpakai tanpa keluhan apa pun.
    Modul ini membaca catatan hash SHA-256 di `index_info.json` dan membandingkannya
    dengan berkas yang ada di folder data saat ini.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from app.config import DATA_DIR, INDEX_INFO_PATH, INGESTION_PIPELINE_VERSION


def file_sha256(path: str | Path) -> str:
    """Hash SHA-256 sebuah berkas, dibaca bertahap agar hemat memori."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass
class DocumentStatus:
    """Status satu dokumen sumber terhadap index."""

    name: str
    indexed_sha256: str = ""
    current_sha256: str | None = None
    exists: bool = True

    @property
    def changed(self) -> bool:
        """True bila berkasnya ada tetapi isinya sudah berbeda."""
        return (
            self.exists
            and self.current_sha256 is not None
            and bool(self.indexed_sha256)
            and self.current_sha256 != self.indexed_sha256
        )


@dataclass
class IndexFreshness:
    """Ringkasan kesegaran index terhadap seluruh dokumen sumber."""

    has_info: bool = False
    documents: list[DocumentStatus] = field(default_factory=list)
    untracked_documents: list[str] = field(default_factory=list)

    indexed_pipeline_version: int = 0
    current_pipeline_version: int = INGESTION_PIPELINE_VERSION

    @property
    def pipeline_outdated(self) -> bool:
        """True bila index dibangun oleh logika pipeline yang lebih lama."""
        if not self.has_info:
            return False
        return self.indexed_pipeline_version < self.current_pipeline_version

    @property
    def changed(self) -> list[DocumentStatus]:
        return [d for d in self.documents if d.changed]

    @property
    def missing(self) -> list[DocumentStatus]:
        return [d for d in self.documents if not d.exists]

    @property
    def is_stale(self) -> bool:
        """True bila index perlu dibangun ulang."""
        return bool(self.changed or self.untracked_documents or self.pipeline_outdated)

    def messages(self) -> list[str]:
        """Pesan siap tampil, satu per masalah yang ditemukan."""
        pesan: list[str] = []

        if not self.has_info:
            pesan.append(
                "Index ini dibangun sebelum pencatatan asal-usul ada, sehingga "
                "kesegarannya tidak dapat diperiksa. Bangun ulang untuk "
                "mengaktifkan deteksi otomatis."
            )
            return pesan

        if self.pipeline_outdated:
            pesan.append(
                f"Pipeline ingestion sudah diperbarui "
                f"(versi {self.indexed_pipeline_version} → "
                f"{self.current_pipeline_version}). Dokumennya tidak berubah, "
                "tetapi aturan ekstraksi atau chunking sudah diperbaiki, "
                "sehingga index sekarang menghasilkan chunk yang berbeda."
            )

        for doc in self.changed:
            pesan.append(
                f"Dokumen **{doc.name}** sudah berubah sejak index dibangun. "
                "Jawaban chatbot masih memakai versi lama."
            )
        for doc in self.missing:
            pesan.append(
                f"Berkas sumber **{doc.name}** tidak ditemukan lagi, jadi "
                "kesegarannya tidak bisa diperiksa."
            )
        for name in self.untracked_documents:
            pesan.append(
                f"Dokumen **{name}** ada di folder data tetapi belum terindeks."
            )
        return pesan


def load_index_info(path: str | Path = INDEX_INFO_PATH) -> dict:
    """Baca `index_info.json`. Kembalikan dict kosong bila tidak ada/rusak."""
    path = Path(path)
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def indexed_documents(index_info: dict) -> list[dict]:
    """Ambil daftar dokumen sumber dari catatan build."""
    documents = index_info.get("source_documents")
    if isinstance(documents, list) and documents:
        return documents

    single = index_info.get("source_pdf")
    if isinstance(single, dict) and single.get("name"):
        return [single]

    return []


def check_index_freshness(
    index_info: dict | None = None,
    data_dir: str | Path = DATA_DIR,
) -> IndexFreshness:
    """
    Bandingkan hash dokumen yang tercatat di index dengan berkas saat ini.
    Mendukung berkas PDF dan Markdown.
    """
    if index_info is None:
        index_info = load_index_info()

    freshness = IndexFreshness(has_info=bool(index_info))
    if not index_info:
        return freshness

    freshness.indexed_pipeline_version = int(index_info.get("pipeline_version", 0) or 0)

    data_dir = Path(data_dir)
    tercatat: set[str] = set()

    for entry in indexed_documents(index_info):
        name = entry.get("name", "")
        if not name:
            continue
        tercatat.add(name)

        path = data_dir / name
        if not path.exists():
            matches = list(data_dir.rglob(name))
            if matches:
                path = matches[0]
        status = DocumentStatus(
            name=name,
            indexed_sha256=entry.get("sha256", ""),
            exists=path.exists(),
        )
        if status.exists:
            try:
                status.current_sha256 = file_sha256(path)
            except OSError:
                status.exists = False
        freshness.documents.append(status)

    if data_dir.exists():
        freshness.untracked_documents = sorted(
            p.name
            for p in data_dir.rglob("*")
            if p.is_file()
            and p.suffix.lower() in (".pdf", ".md", ".markdown")
            and p.name.lower() != "readme.md"
            and p.name not in tercatat
        )

    return freshness
