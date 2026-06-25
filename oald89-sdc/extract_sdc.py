#!/usr/bin/env python3
"""
Extract chunks/resources from SlovoEd-style *.sdc files with an SLD2 header.

This is based on the chunk table layout visible:
  offset 0x00: b"SLD2"
  offset 0x04: table offset / header size, usually 0x80
  offset 0x10: file size
  offset 0x14: database id/name bytes, zero-padded
  offset 0x18: chunk count
  offset 0x1c: chunk entry size, usually 0x10

Each chunk table entry is:
  0x00: 4-byte ASCII tag, e.g. IMGA, STRW, HEAD
  0x04: uint32 chunk index
  0x08: uint32 chunk size
  0x0c: uint32 chunk file offset

The script writes:
  - manifest.json with every chunk's offset/size/type
  - chunks/<tag>_<index>_<offset>.bin for all raw chunks
  - images/<tag>_<index>.<jpg|png|...> for chunks whose payload is an image
  - text/<tag>_<index>.txt for UTF-16-ish string chunks (STRW/STRL)
"""

from __future__ import annotations

import argparse
import json
import re
import struct
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence


@dataclass
class Chunk:
    """A single entry from the SDC chunk table.

    Attributes:
        ordinal: Position of the entry in the chunk table (0-based).
        tag: 4-byte ASCII type tag, e.g. ``IMGA``, ``STRW``, ``HEAD``.
        index: Per-tag chunk index from the table entry.
        size: Payload size in bytes.
        offset: Absolute byte offset of the payload within the file.
        end: Absolute byte offset of the payload's end (``offset + size``).
        extension: Detected file extension for the payload, or ``None``.
    """

    ordinal: int
    tag: str
    index: int
    size: int
    offset: int
    end: int
    extension: str | None = None


class SdcError(Exception):
    """Raised when a file is not a valid/parseable SLD2 SDC container."""


def u32le(buf: bytes, off: int) -> int:
    """Read a little-endian unsigned 32-bit integer from ``buf`` at ``off``."""
    return struct.unpack_from("<I", buf, off)[0]


def safe_name(text: str) -> str:
    """Sanitize ``text`` into a filesystem-safe name.

    Replaces runs of disallowed characters with ``_`` and trims leading/trailing
    dots and underscores. Falls back to ``"unnamed"`` if nothing usable remains.
    """
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text)
    return text.strip("._") or "unnamed"


def detect_ext(data: bytes) -> str | None:
    """Return a practical extension for common embedded payloads."""
    if data.startswith(b"\xff\xd8\xff"):
        return "jpg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if data.startswith(b"BM"):
        return "bmp"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "webp"
    if data.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return "wav"
    if data.startswith(b"OggS"):
        return "ogg"
    if data.startswith(b"ID3"):
        return "mp3"
    return None


def parse_sdc(path: Path) -> tuple[dict, list[Chunk]]:
    """Parse the SLD2 header and chunk table of an SDC file.

    Args:
        path: Path to the ``.sdc`` file to read.

    Returns:
        A ``(header, chunks)`` tuple where ``header`` is a dict describing the
        SLD2 header fields and ``chunks`` is the list of parsed :class:`Chunk`
        entries.

    Raises:
        SdcError: If the file is too small, lacks the ``SLD2`` magic, or the
            chunk table / any chunk points past the end of the file.
    """
    buf = path.read_bytes()
    if len(buf) < 0x20:
        raise SdcError(f"{path}: too small to be an SLD2 SDC file")
    if buf[:4] != b"SLD2":
        raise SdcError(f"{path}: expected SLD2 magic, got {buf[:4]!r}")

    table_offset = u32le(buf, 0x04)
    declared_file_size = u32le(buf, 0x10)
    chunk_count = u32le(buf, 0x18)
    entry_size = u32le(buf, 0x1C)

    if entry_size < 16:
        raise SdcError(f"{path}: invalid chunk entry size {entry_size}")
    if table_offset + chunk_count * entry_size > len(buf):
        raise SdcError(f"{path}: chunk table points past end of file")

    raw_id = buf[0x14:0x18]
    db_id = raw_id.split(b"\0", 1)[0].decode("ascii", "replace")

    header = {
        "magic": "SLD2",
        "table_offset": table_offset,
        "declared_file_size": declared_file_size,
        "actual_file_size": len(buf),
        "database_id_guess": db_id,
        "chunk_count": chunk_count,
        "entry_size": entry_size,
    }

    chunks: list[Chunk] = []
    for ordinal in range(chunk_count):
        entry_off = table_offset + ordinal * entry_size
        tag_bytes = buf[entry_off : entry_off + 4]
        tag = tag_bytes.decode("ascii", "replace")
        index = u32le(buf, entry_off + 4)
        size = u32le(buf, entry_off + 8)
        offset = u32le(buf, entry_off + 12)
        end = offset + size
        if end > len(buf):
            raise SdcError(
                f"{path}: chunk {ordinal} {tag}/{index} points past EOF "
                f"(offset=0x{offset:x}, size=0x{size:x}, file=0x{len(buf):x})"
            )
        ext = detect_ext(buf[offset:end])
        chunks.append(Chunk(ordinal, tag, index, size, offset, end, ext))

    return header, chunks


def decode_utf16_string_chunk(data: bytes) -> list[str]:
    """
    SDC string chunks in this sample look like:
      uint32 size, char[4] language/id, then UTF-16LE NUL-separated strings.
    Return non-empty printable strings.
    """
    if len(data) <= 8:
        return []
    payload = data[8:]
    if len(payload) % 2:
        payload = payload[:-1]
    try:
        text = payload.decode("utf-16le", errors="ignore")
    except UnicodeDecodeError:
        return []
    strings = []
    for part in text.split("\x00"):
        part = part.strip()
        if part and any(ch.isprintable() and not ch.isspace() for ch in part):
            strings.append(part)
    return strings


def extract_file(path: Path, out_root: Path, raw_chunks: bool = True) -> dict:
    """Extract one SDC file's chunks to disk and return its manifest.

    Writes outputs under ``out_root/<file stem>/``: ``manifest.json``, optional
    raw ``chunks/`` binaries, decoded ``images/``, and ``text/`` files for
    UTF-16 string chunks (``STRW``/``STRL``).

    Args:
        path: Path to the ``.sdc`` file to extract.
        out_root: Directory under which the per-file output folder is created.
        raw_chunks: If ``True``, also write every chunk's raw payload as ``.bin``.

    Returns:
        The manifest dict that was serialized to ``manifest.json``.
    """
    buf = path.read_bytes()
    header, chunks = parse_sdc(path)

    base_out = out_root / safe_name(path.stem)
    chunks_dir = base_out / "chunks"
    images_dir = base_out / "images"
    text_dir = base_out / "text"
    base_out.mkdir(parents=True, exist_ok=True)

    exported_images = 0
    exported_text = 0
    if raw_chunks:
        chunks_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        data = buf[chunk.offset : chunk.end]
        tag = safe_name(chunk.tag)
        stem = f"{chunk.ordinal:03d}_{tag}_{chunk.index:04d}_0x{chunk.offset:08x}"

        if raw_chunks:
            (chunks_dir / f"{stem}.bin").write_bytes(data)

        if chunk.extension:
            # Use tag/index in the visible asset filename; ordinal/offset still in manifest.
            asset_name = f"{tag}_{chunk.index:04d}.{chunk.extension}"
            # Avoid overwrite if a weird file has duplicate tag/index.
            dest = images_dir / asset_name
            if dest.exists():
                dest = images_dir / f"{stem}.{chunk.extension}"
            dest.write_bytes(data)
            exported_images += 1

        if chunk.tag in {"STRW", "STRL"}:
            strings = decode_utf16_string_chunk(data)
            if strings:
                lang_or_id = (
                    data[4:8].decode("ascii", "replace") if len(data) >= 8 else ""
                )
                text = [f"# {chunk.tag} index={chunk.index} lang/id={lang_or_id!r}", ""]
                text.extend(strings)
                (text_dir / f"{tag}_{chunk.index:04d}.txt").write_text(
                    "\n".join(text), encoding="utf-8"
                )
                exported_text += 1

    manifest = {
        "source": str(path),
        "output_directory": str(base_out),
        "header": header,
        "chunks": [asdict(c) for c in chunks],
        "summary": {
            "chunks": len(chunks),
            "images": exported_images,
            "text_chunks": exported_text,
            "raw_chunks_written": raw_chunks,
        },
    }
    (base_out / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: extract one or more ``.sdc`` files.

    Args:
        argv: Optional argument list (defaults to ``sys.argv`` when ``None``).

    Returns:
        Process exit code (``0`` on success).
    """
    parser = argparse.ArgumentParser(
        description="Extract SLD2/Sdc chunk contents from *.sdc files."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="One or more .sdc files")
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("sdc_extracted"), help="Output directory"
    )
    parser.add_argument(
        "--no-raw", action="store_true", help="Do not write every raw chunk .bin file"
    )
    args = parser.parse_args(argv)

    for input_path in args.inputs:
        manifest = extract_file(input_path, args.out, raw_chunks=not args.no_raw)
        summary = manifest["summary"]
        print(
            f"{input_path}: {summary['chunks']} chunks, "
            f"{summary['images']} images, {summary['text_chunks']} text chunks -> "
            f"{manifest['output_directory']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
