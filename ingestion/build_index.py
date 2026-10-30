"""
build_index.py — Pipeline ingestion: PDF & Markdown → teks bersih → chunk → embedding →
FAISS index.

Usage:
    # Lihat hasil chunking tanpa memanggil API sama sekali (gratis)
    python ingestion/build_index.py --dry-run

    # Bangun index dari seluruh dokumen (PDF dan Markdown) di folder data/
    python ingestion/build_index.py --yes

    # Dokumen tertentu saja (boleh diulang, PDF maupun Markdown)
    python ingestion/build_index.py --doc data/pedoman.pdf --doc data/catatan.md --yes
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Tambahkan root project ke path agar modul lain bisa diimpor saat script
# dijalankan langsung.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.cli_utils import enable_utf8_stdout
from app.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    DATA_DIR,
    EMBEDDING_MODEL,
    INDEX_DIR,
    INGESTION_PIPELINE_VERSION,
    MIN_CHUNK_SIZE,
)
from app.index_status import file_sha256
from ingestion.chunking import MAX_PAGES_PER_CHUNK, Chunk, chunk_pages, print_chunk_stats
from ingestion.extract_markdown import extract_markdown_with_report
from ingestion.extract_text import ExtractionReport, extract_text_with_report


# ──────────────────────────────────────────────
# Penemuan dokumen
# ──────────────────────────────────────────────

def discover_documents(paths: list[str] | None, data_dir: Path = DATA_DIR) -> list[Path]:
    """
    Tentukan daftar dokumen (PDF dan Markdown) yang akan diproses.

    Args:
        paths: Path eksplisit dari CLI. Boleh berupa berkas maupun folder.
            Bila ``None``, seluruh dokumen di ``data_dir`` dipakai.
        data_dir: Folder dokumen bawaan.

    Returns:
        Daftar path dokumen, terurut dan tanpa duplikat.
    """
    if not paths:
        found: list[Path] = []
        for ext in ("*.pdf", "*.md", "*.markdown"):
            for p in data_dir.rglob(ext):
                if p.name.lower() == "readme.md":
                    continue
                found.append(p)
        return sorted(found)

    hasil: list[Path] = []
    for raw in paths:
        path = Path(raw)
        if path.is_dir():
            for ext in ("*.pdf", "*.md", "*.markdown"):
                for p in path.rglob(ext):
                    if p.name.lower() == "readme.md":
                        continue
                    hasil.append(p)
        else:
            hasil.append(path)

    # Buang duplikat tetapi jaga urutannya.
    unik: list[Path] = []
    terlihat: set[Path] = set()
    for path in hasil:
        resolved = path.resolve()
        if resolved not in terlihat:
            terlihat.add(resolved)
            unik.append(path)
    return unik


discover_pdfs = discover_documents


# ──────────────────────────────────────────────
# Index & metadata
# ──────────────────────────────────────────────

def build_faiss_index(embeddings: list[list[float]]):
    """
    Bangun FAISS index dari embedding vector.

    Memakai `IndexFlatIP` dengan vector yang dinormalisasi L2, sehingga inner
    product setara cosine similarity. Index flat juga memungkinkan
    `reconstruct()`, yang dipakai retriever untuk menghitung skor dense
    kandidat yang hanya ditemukan BM25.
    """
    import faiss

    vectors = np.array(embeddings, dtype=np.float32)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    print(f"✅ FAISS index built: {index.ntotal} vectors, dimension {vectors.shape[1]}")
    return index


def chunks_to_metadata(chunks: list[Chunk]) -> list[dict]:
    """
    Susun metadata yang disimpan berdampingan dengan index.
    """
    return [
        {
            "chunk_index": chunk.chunk_index,
            "source_document": chunk.source_document,
            "text": chunk.text,
            "page_numbers": chunk.page_numbers,
            "page_start": chunk.page_start,
            "page_end": chunk.page_end,
            "section_title": chunk.section_title,
            "heading_path": chunk.heading_path,
            "char_count": chunk.char_count,
            "estimated_tokens": chunk.estimated_tokens,
        }
        for chunk in chunks
    ]


def build_index_info(
    documents: list[dict],
    chunks: list[Chunk],
    embedding_dimension: int,
    task_type: str,
) -> dict:
    """Catatan asal-usul index, disimpan sebagai `index_info.json`."""
    spans = [len(chunk.page_numbers) for chunk in chunks]
    pages_per_doc: dict[str, set[int]] = {}
    for chunk in chunks:
        pages_per_doc.setdefault(chunk.source_document, set()).update(chunk.page_numbers)

    total_pages = sum(len(pages) for pages in pages_per_doc.values())

    return {
        "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_documents": documents,
        "embedding": {
            "model": EMBEDDING_MODEL,
            "task_type": task_type,
            "dimension": embedding_dimension,
            "includes_heading_path": True,
        },
        "embedding_task_type": task_type,
        "pipeline_version": INGESTION_PIPELINE_VERSION,
        "chunking": {
            "chunk_size": CHUNK_SIZE,
            "chunk_overlap": CHUNK_OVERLAP,
            "min_chunk_size": MIN_CHUNK_SIZE,
            "max_pages_per_chunk": MAX_PAGES_PER_CHUNK,
        },
        "stats": {
            "total_documents": len(documents),
            "total_chunks": len(chunks),
            "total_pages": total_pages,
            "max_pages_per_chunk": max(spans) if spans else 0,
            "avg_pages_per_chunk": round(sum(spans) / len(spans), 2) if spans else 0,
            "single_page_chunks": sum(1 for s in spans if s == 1),
            "avg_chars_per_chunk": (
                round(sum(c.char_count for c in chunks) / len(chunks)) if chunks else 0
            ),
        },
    }


def save_index(index, metadata: list[dict], info: dict, index_dir: Path) -> None:
    """Tulis FAISS index, metadata, dan catatan build ke disk."""
    import faiss

    index_dir.mkdir(parents=True, exist_ok=True)

    index_path = index_dir / "faiss.index"
    faiss.write_index(index, str(index_path))
    print(f"✅ FAISS index saved: {index_path}")

    metadata_path = index_dir / "metadata.json"
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)
    print(f"✅ Metadata saved: {metadata_path} ({len(metadata)} entries)")

    info_path = index_dir / "index_info.json"
    with open(info_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"✅ Build log saved: {info_path}")


# ──────────────────────────────────────────────
# Pipeline
# ──────────────────────────────────────────────

def prepare_chunks(
    doc_paths: list[Path],
) -> tuple[list[Chunk], list[dict]]:
    """
    Jalankan tahap yang tidak memerlukan API: ekstraksi teks dan chunking.
    Mendukung PDF maupun berkas Markdown.
    """
    semua_chunk: list[Chunk] = []
    catatan: list[dict] = []

    for nomor, doc_path in enumerate(doc_paths, start=1):
        print(f"\n{'─' * 62}")
        print(f"📖 Document {nomor}/{len(doc_paths)}: {doc_path.name}")
        print(f"{'─' * 62}")

        if doc_path.suffix.lower() in (".md", ".markdown"):
            chunks, report = extract_markdown_with_report(doc_path)
            if not chunks:
                print(f"⚠️  Skipped: no chunks generated.")
                continue
            pages_with_text = report.pages_with_text
            empty_pages = []
        else:
            pages, report = extract_text_with_report(doc_path)
            if not pages:
                print(f"⚠️  Skipped: no text could be extracted.")
                continue
            chunks = chunk_pages(pages)
            if not chunks:
                print(f"⚠️  Skipped: no chunks generated.")
                continue
            pages_with_text = len(pages)
            empty_pages = report.pages_empty_after_cleaning

        for chunk in chunks:
            chunk.source_document = doc_path.name

        semua_chunk.extend(chunks)
        catatan.append({
            "name": doc_path.name,
            "sha256": file_sha256(doc_path),
            "size_bytes": doc_path.stat().st_size,
            "pages_with_text": pages_with_text,
            "chunks": len(chunks),
            "extraction": {
                "lines_raw": report.lines_raw,
                "lines_kept": report.lines_kept,
                "dropped": dict(report.dropped),
                "pages_empty_after_cleaning": empty_pages,
            },
        })

    # Nomor ulang secara global supaya sejajar dengan urutan vector FAISS.
    for i, chunk in enumerate(semua_chunk):
        chunk.chunk_index = i

    return semua_chunk, catatan


def run_ingestion(
    doc_paths: list[Path],
    api_key: str | None = None,
    dry_run: bool = False,
    index_dir: Path = INDEX_DIR,
) -> None:
    """
    Jalankan pipeline ingestion lengkap.

    Args:
        doc_paths: Daftar berkas PDF / Markdown sumber.
        api_key: Override API key Gemini.
        dry_run: Berhenti setelah chunking, tanpa memanggil API.
        index_dir: Folder tujuan penyimpanan index.
    """
    print("=" * 62)
    print("🚀 INGESTION PIPELINE — Academic RAG Chatbot")
    print(f"   Dokumen: {len(doc_paths)}")
    if dry_run:
        print("   MODE: dry-run (tanpa panggilan API, tanpa menulis index)")
    print("=" * 62)

    chunks, documents = prepare_chunks(doc_paths)
    if not chunks:
        print("\n❌ Tidak ada chunk yang dihasilkan dari dokumen mana pun!")
        sys.exit(1)

    if len(documents) > 1:
        print(f"\n{'─' * 62}")
        print("📚 Gabungan seluruh dokumen")
        print(f"{'─' * 62}")
        for doc in documents:
            print(f"   {doc['name']}: {doc['chunks']} chunk dari "
                  f"{doc['pages_with_text']} halaman/seksi")
        print_chunk_stats(chunks)

    if dry_run:
        print("\n🔍 Contoh chunk hasil pemotongan")
        for chunk in chunks[:3]:
            heading = " > ".join(chunk.heading_path) or "-"
            print(f"\n   --- chunk {chunk.chunk_index} | {chunk.source_document} "
                  f"| halaman {chunk.page_label} | {chunk.char_count} karakter ---")
            print(f"   heading : {heading}")
            print(f"   teks    : {chunk.text[:200].replace(chr(10), ' ⏎ ')}")

        print("\n" + "=" * 62)
        print("✅ DRY-RUN SELESAI — tidak ada biaya API dan index tidak diubah.")
        print(f"   Jalankan tanpa --dry-run untuk meng-embed {len(chunks)} chunk.")
        print("=" * 62)
        return

    # Embedded text includes heading path for better semantic retrieval
    print(f"\n🧮 Embedding {len(chunks)} chunks via Gemini Embedding 2...")
    import hashlib
    import pickle
    import time
    from app.config import EMBED_BATCH_DELAY, EMBED_BATCH_SIZE
    from app.llm_client import TASK_TYPE_DOCUMENT, embed_texts

    cache_path = index_dir / "embedding_cache.pkl"
    cache: dict[str, list[float]] = {}
    if cache_path.exists():
        try:
            with open(cache_path, "rb") as f:
                cache = pickle.load(f)
            print(f"📦 Loaded {len(cache)} cached embeddings from {cache_path.name}")
        except Exception:
            cache = {}

    hashes = [hashlib.sha256(chunk.embed_text.encode("utf-8")).hexdigest() for chunk in chunks]
    missing_indices = [i for i, h in enumerate(hashes) if h not in cache]

    if missing_indices:
        print(f"📦 {len(chunks) - len(missing_indices)} chunks already cached on disk.")
        print(f"🚀 Embedding remaining {len(missing_indices)} chunks in batches of {EMBED_BATCH_SIZE}...")
        missing_texts = [chunks[i].embed_text for i in missing_indices]
        total_batches = (len(missing_texts) - 1) // EMBED_BATCH_SIZE + 1

        for b_idx in range(0, len(missing_texts), EMBED_BATCH_SIZE):
            batch_texts = missing_texts[b_idx:b_idx + EMBED_BATCH_SIZE]
            batch_hashes = [hashes[missing_indices[b_idx + j]] for j in range(len(batch_texts))]
            batch_num = b_idx // EMBED_BATCH_SIZE + 1
            print(f"\n   --- Batch {batch_num}/{total_batches} ({len(batch_texts)} chunks) ---", flush=True)

            batch_vectors = embed_texts(
                batch_texts,
                task_type=TASK_TYPE_DOCUMENT,
                api_key=api_key,
                batch_size=len(batch_texts),
                batch_delay=0.0,
                progress=True,
            )
            for h, vec in zip(batch_hashes, batch_vectors):
                cache[h] = vec

            # Save checkpoint immediately to disk
            index_dir.mkdir(parents=True, exist_ok=True)
            temp_cache = cache_path.with_suffix(".tmp")
            with open(temp_cache, "wb") as f:
                pickle.dump(cache, f)
            temp_cache.replace(cache_path)

            if batch_num < total_batches and EMBED_BATCH_DELAY > 0:
                print(f"   ⏳ Waiting {EMBED_BATCH_DELAY:.0f}s before next batch to respect rate limits...", flush=True)
                time.sleep(EMBED_BATCH_DELAY)
    else:
        print(f"⚡ All {len(chunks)} chunks retrieved instantly from cache!")

    embeddings = [cache[h] for h in hashes]

    print("\n💾 Membangun dan menyimpan index...")
    index = build_faiss_index(embeddings)
    info = build_index_info(
        documents=documents,
        chunks=chunks,
        embedding_dimension=len(embeddings[0]),
        task_type=TASK_TYPE_DOCUMENT,
    )
    save_index(index, chunks_to_metadata(chunks), info, index_dir)

    stats = info["stats"]
    print("\n" + "=" * 62)
    print("✅ INGESTION COMPLETED")
    print(f"   📚 Documents      : {stats['total_documents']}")
    print(f"   📝 Pages/Sections : {stats['total_pages']}")
    print(f"   ✂️  Chunks         : {stats['total_chunks']}")
    print(f"   🎯 Pages/chunk    : max {stats['max_pages_per_chunk']}, "
          f"avg {stats['avg_pages_per_chunk']}")
    print(f"   🧮 Vector Dim     : {info['embedding']['dimension']}")
    print(f"   💾 Index Directory: {index_dir}")


def _confirm_overwrite(index_dir: Path, assume_yes: bool = False) -> None:
    """Minta konfirmasi bila index lama akan ditimpa."""
    if assume_yes:
        return

    existing = [p.name for p in index_dir.glob("*") if p.is_file()]
    if not existing:
        return

    print("⚠️  Index yang sudah ada akan DITIMPA:")
    for path in existing:
        print(f"      {path}")
    print("\n   Proses ini juga memakai kuota embedding Gemini API.")
    print("   Tambahkan flag --yes untuk melanjutkan, atau --dry-run untuk")
    print("   melihat hasil chunking tanpa biaya dan tanpa mengubah apa pun.")
    sys.exit(1)


def main() -> None:
    enable_utf8_stdout()

    parser = argparse.ArgumentParser(
        description="Bangun FAISS index dari satu atau beberapa dokumen PDF / Markdown kampus",
    )
    parser.add_argument(
        "--doc",
        "--pdf",
        dest="docs",
        action="append",
        default=None,
        metavar="PATH",
        help="Berkas PDF/Markdown atau folder. Boleh diulang. "
             f"Default: seluruh PDF dan Markdown di {DATA_DIR.name}/",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Google Gemini API key (default: env GEMINI_API_KEY atau secrets.toml)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Hanya ekstraksi + chunking, tanpa panggilan API dan tanpa menulis index",
    )
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Setujui penimpaan index yang sudah ada",
    )
    args = parser.parse_args()

    doc_paths = discover_documents(args.docs)
    if not doc_paths:
        lokasi = ", ".join(args.docs) if args.docs else str(DATA_DIR)
        print(f"❌ Tidak ada berkas PDF / Markdown yang ditemukan di: {lokasi}")
        sys.exit(1)

    hilang = [p for p in doc_paths if not p.exists()]
    if hilang:
        for path in hilang:
            print(f"❌ File tidak ditemukan: {path}")
        sys.exit(1)

    if not args.dry_run:
        _confirm_overwrite(INDEX_DIR, assume_yes=args.yes)

        from app.llm_client import LLMConfigError, resolve_api_key

        try:
            resolve_api_key(args.api_key)
        except LLMConfigError as e:
            print(f"❌ {e}")
            sys.exit(1)

    run_ingestion(doc_paths, api_key=args.api_key, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
