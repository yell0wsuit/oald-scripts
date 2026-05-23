#!/usr/bin/env python3
"""Extract OALD9 Lingea media containers and decode dictionary records."""

from __future__ import annotations

import argparse
import html
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

MAGICS = (
    (b"\xff\xd8\xff", ".jpg"),
    (b"\x89PNG\r\n\x1a\n", ".png"),
    (b"GIF8", ".gif"),
    (b"OggS", ".ogg"),
    (b"ID3", ".mp3"),
    (b"\xff\xfb", ".mp3"),
    (b"RIFF", ".wav"),
    (b"FWS", ".swf"),
    (b"CWS", ".swf"),
)

LINGEA_SHIFT_CHARS = "`'^~:-.\"+,_@gcdh   splau"
LINGEA_PUNCT_CHARS = bytes.fromhex(
    "7f 3a 3b 2b 2a 2f 3d 60 27 22 30 31 32 33 34 35 "
    "36 37 38 39 21 3f 23 24 25 26 40 28 29 3c 3e 5b "
    "5d 7b 7d 5c 7c 5e 5f 7e 09 0a 00"
)
LINGEA_LOW_CHARS = bytes.fromhex(
    "7f 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f "
    "70 71 72 73 74 75 76 77 78 79 7a 7f 7f 7f 7f 7f "
    "20 2e 2d 2c 02 05 00 00"
)
LINGEA_LETTER_CHARS = bytes.fromhex(
    "7f 61 62 63 64 65 66 67 68 69 6a 6b 6c 6d 6e 6f "
    "70 71 72 73 74 75 76 77 78 79 7a 30 31 32 33 34 "
    "41 42 43 44 45 46 47 48 49 4a 4b 4c 4d 4e 4f 50 "
    "51 52 53 54 55 56 57 58 59 5a 35 36 37 38 39 21 "
    "3f 23 24 25 26 40 28 29 3c 3e 5b 5d 7b 7d 5c 7c "
    "5e 5f 7e 09 0a 00"
)


@dataclass(frozen=True)
class DecodedChunk:
    """One decoded string chunk inside a Lingea article record."""

    offset: int
    consumed: int
    text: str


def u32(buf: bytes, offset: int) -> int:
    """Read an unsigned little-endian 32-bit integer from a byte buffer."""
    return struct.unpack_from("<I", buf, offset)[0]


def i32(buf: bytes, offset: int) -> int:
    """Read a signed little-endian 32-bit integer from a byte buffer."""
    return struct.unpack_from("<i", buf, offset)[0]


def strip_known_prefix(blob: bytes, max_scan: int = 64) -> bytes:
    """Trim small Lingea wrapper bytes before a known media payload magic."""
    best = None
    scan = blob[:max_scan]
    for magic, _ in MAGICS:
        pos = scan.find(magic)
        if pos >= 0 and (best is None or pos < best):
            best = pos
    return blob[best:] if best is not None else blob


def extension_for(blob: bytes) -> str:
    """Return the output extension implied by a media blob's magic bytes."""
    payload = strip_known_prefix(blob)
    for magic, ext in MAGICS:
        if payload.startswith(magic):
            return ext
    return ".bin"


def safe_name(name: str) -> str:
    """Return a filesystem-safe name for a decoded dictionary identifier."""
    name = name.strip().replace("\\", "_").replace("/", "_")
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    return name.strip(" .") or "entry"


def _lingea_6bit_value(
    buf: bytes, byte_offset: int, value_index: int
) -> tuple[int, int]:
    """Read one packed 6-bit Lingea text value and return the next byte offset."""
    if value_index % 4 == 0:
        return buf[byte_offset] >> 2, byte_offset
    if value_index % 4 == 1:
        return (
            ((buf[byte_offset] & 0x03) << 4) | (buf[byte_offset + 1] >> 4),
            byte_offset + 1,
        )
    if value_index % 4 == 2:
        return (
            ((buf[byte_offset] & 0x0F) << 2) | (buf[byte_offset + 1] >> 6),
            byte_offset + 1,
        )
    return buf[byte_offset] & 0x3F, byte_offset + 1


def _lingea_emit_shifted(shift_index: int, code: int, extra: int = 0) -> str:
    """Convert a Lingea text-code pair into Unicode text."""
    shift = (
        LINGEA_SHIFT_CHARS[shift_index]
        if 0 <= shift_index < len(LINGEA_SHIFT_CHARS)
        else ""
    )
    if shift == "a":
        return chr(LINGEA_PUNCT_CHARS[code]) if code < len(LINGEA_PUNCT_CHARS) else ""
    if shift == "l":
        return chr(LINGEA_LETTER_CHARS[code]) if code < len(LINGEA_LETTER_CHARS) else ""
    if shift == "s":
        symbols = {
            3: "...",
            7: "<",
            13: "-",
            14: "-",
            15: "TM",
            16: ">",
            24: "(c)",
            26: "<<",
            38: ">>",
            39: "1/4",
            40: "1/2",
            41: "3/4",
            43: "x",
        }
        return symbols.get(code, "*")
    if code < len(LINGEA_LETTER_CHARS):
        base = chr(LINGEA_LETTER_CHARS[code])
        if extra and extra < len(LINGEA_LETTER_CHARS):
            return base + chr(LINGEA_LETTER_CHARS[extra])
        return base
    return "*"


def _decode_lingea_entity(
    buf: bytes, byte_offset: int, value_index: int
) -> tuple[str, int, int, int]:
    """Decode a four-value numeric entity from a Lingea text chunk."""
    number = 0
    last_value = 0
    for _ in range(4):
        code, byte_offset = _lingea_6bit_value(buf, byte_offset, value_index)
        value_index += 1
        last_value = code
        number = (code - 1) | (16 * number)
    return f"&#{number};", byte_offset, value_index, last_value


def _decode_lingea_low_value(value: int, current_shift: int) -> str:
    """Decode a Lingea value below 40 using the current shift state."""
    if value < 32:
        return _lingea_emit_shifted(current_shift - 40, value) if value else ""
    return chr(LINGEA_LOW_CHARS[value]) if value < len(LINGEA_LOW_CHARS) else ""


def _decode_lingea_shifted_value(
    buf: bytes, byte_offset: int, value_index: int, value: int
) -> tuple[str, int | None, int, int]:
    """Decode a shifted Lingea value and optionally return a new shift state."""
    code, byte_offset = _lingea_6bit_value(buf, byte_offset, value_index)
    value_index += 1
    extra = 0
    if value == 60:
        extra, byte_offset = _lingea_6bit_value(buf, byte_offset, value_index)
        value_index += 1
    if code == 63:
        return "", value, byte_offset, value_index
    return _lingea_emit_shifted(value - 40, code, extra), None, byte_offset, value_index


def decode_lingea_text_chunk(
    buf: bytes, offset: int, max_values: int = 50000
) -> tuple[str, int]:
    """Decode one null-terminated packed Lingea text chunk from a record."""
    out: list[str] = []
    byte_offset = offset
    value_index = 0
    values_read = 0
    current_shift = 61
    last_value = None
    try:
        while byte_offset < len(buf) and values_read < max_values:
            value, byte_offset = _lingea_6bit_value(buf, byte_offset, value_index)
            value_index += 1
            last_value = value
            if value == 63:
                text, byte_offset, value_index, last_value = _decode_lingea_entity(
                    buf, byte_offset, value_index
                )
                out.append(text)
            elif value < 40:
                out.append(_decode_lingea_low_value(value, current_shift))
            else:
                text, new_shift, byte_offset, value_index = (
                    _decode_lingea_shifted_value(buf, byte_offset, value_index, value)
                )
                current_shift = new_shift if new_shift is not None else current_shift
                out.append(text)
            values_read = value_index
            if last_value == 0:
                break
    except IndexError:
        pass
    if value_index % 4:
        byte_offset += 1
    text = "".join(out).replace("\x02", "<").replace("\x05", ">").replace("\x7f", "")
    return text, max(0, byte_offset - offset)


def _decoded_chunk_score(chunk: DecodedChunk) -> int:
    """Rank candidate decoded chunks so real article XML wins over noise."""
    text = chunk.text
    score = len(text)
    if text.startswith("<c.entry>"):
        score += 4000
    if text.startswith("<c.") or text.startswith("<listcell"):
        score += 1500
    if "<c.def" in text:
        score += 1000
    if "dictionary" in text:
        score += 500
    score -= 50 * text.count("*")
    return score


def decoded_article_chunks(blob: bytes) -> list[DecodedChunk]:
    """Decode likely article text chunks from a binary Lingea dictionary record."""
    selected: list[DecodedChunk] = []
    offset = 0
    while offset < len(blob):
        candidates: list[DecodedChunk] = []
        window_end = min(offset + 8, len(blob))
        for candidate_offset in range(offset, window_end):
            text, consumed = decode_lingea_text_chunk(
                blob, candidate_offset, max_values=2500
            )
            if consumed < 4 or len(text) < 8:
                continue
            bad_ratio = (text.count("*") + text.count("\x00")) / max(len(text), 1)
            has_article_xml = "<c." in text or "</" in text or "<listcell" in text
            has_words = sum(ch.isalpha() for ch in text) / max(len(text), 1) > 0.65
            if bad_ratio <= 0.15 and (
                has_article_xml or "dictionary" in text or has_words
            ):
                candidates.append(DecodedChunk(candidate_offset, consumed, text))
        if not candidates:
            offset += 1
            continue
        best = max(candidates, key=_decoded_chunk_score)
        selected.append(best)
        offset = best.offset + max(best.consumed, 1)
    return selected


def decoded_article_chunks_exhaustive(blob: bytes) -> list[DecodedChunk]:
    """Decode article chunks by scanning every byte offset for diagnostics."""
    candidates: list[DecodedChunk] = []
    for offset in range(len(blob)):
        text, consumed = decode_lingea_text_chunk(blob, offset, max_values=2500)
        if consumed < 4 or len(text) < 8:
            continue
        bad_ratio = (text.count("*") + text.count("\x00")) / max(len(text), 1)
        has_article_xml = "<c." in text or "</" in text or "<listcell" in text
        has_words = sum(ch.isalpha() for ch in text) / max(len(text), 1) > 0.65
        if bad_ratio <= 0.15 and (has_article_xml or "dictionary" in text or has_words):
            candidates.append(DecodedChunk(offset, consumed, text))

    selected: list[DecodedChunk] = []
    for candidate in sorted(candidates, key=lambda chunk: chunk.offset):
        if not selected:
            selected.append(candidate)
            continue
        previous = selected[-1]
        previous_end = previous.offset + max(previous.consumed, 1)
        if candidate.offset < previous_end:
            if _decoded_chunk_score(candidate) > _decoded_chunk_score(previous):
                selected[-1] = candidate
            continue
        selected.append(candidate)
    return selected


@dataclass(frozen=True)
class LingeaBox:
    """Reader for Lingea MBX/WBX media containers."""

    path: Path
    count: int = field(init=False)
    offset_table: int = field(init=False)
    data_base: int = field(init=False)

    def __post_init__(self) -> None:
        """Load container metadata from the Lingea file header."""
        object.__setattr__(self, "path", Path(self.path))
        with self.path.open("rb") as f:
            header = f.read(0x100)
        if len(header) < 0xAC or header[0x54:0x56] != b"LD":
            raise ValueError(f"{self.path} is not a supported Lingea box")
        object.__setattr__(self, "count", u32(header, 0x90))
        object.__setattr__(self, "offset_table", u32(header, 0xA4))
        object.__setattr__(self, "data_base", u32(header, 0xA8))

    def offsets(self) -> list[int]:
        """Read the media entry offset table."""
        with self.path.open("rb") as f:
            f.seek(self.offset_table)
            raw = f.read(4 * (self.count + 1))
        if len(raw) != 4 * (self.count + 1):
            raise ValueError(f"{self.path} offset table is truncated")
        return list(struct.unpack("<" + "I" * (self.count + 1), raw))

    def read_entry(self, index: int) -> bytes:
        """Read one raw media entry by numeric index."""
        if index < 0 or index >= self.count:
            raise IndexError(index)
        offsets = self.offsets()
        start = self.data_base + offsets[index]
        end = self.data_base + offsets[index + 1]
        if end < start:
            raise ValueError(f"entry {index} has a negative length")
        with self.path.open("rb") as f:
            f.seek(start)
            return f.read(end - start)

    def extract(
        self, out_dir: Path, limit: int | None = None, known_only: bool = True
    ) -> int:
        """Extract media entries into files and return the number written."""
        out_dir.mkdir(parents=True, exist_ok=True)
        offsets = self.offsets()
        total = self.count if limit is None else min(limit, self.count)
        written = 0
        with self.path.open("rb") as f:
            for index in range(total):
                start = self.data_base + offsets[index]
                end = self.data_base + offsets[index + 1]
                if end <= start:
                    continue
                f.seek(start)
                blob = strip_known_prefix(f.read(end - start))
                ext = extension_for(blob)
                if known_only and ext == ".bin":
                    continue
                (out_dir / f"{index:06d}{ext}").write_bytes(blob)
                written += 1
        return written


@dataclass(frozen=True)
class LingeaRecordStore:
    """Reader for Lingea LLD record stores."""

    path: Path
    count: int = field(init=False)
    offset_table: int = field(init=False)
    data_base: int = field(init=False)

    def __post_init__(self) -> None:
        """Load record-store metadata from the Lingea file header."""
        object.__setattr__(self, "path", Path(self.path))
        with self.path.open("rb") as f:
            header = f.read(0x100)
        if len(header) < 0xAC or header[0x54:0x56] != b"LD":
            raise ValueError(f"{self.path} is not a supported Lingea record store")
        object.__setattr__(self, "count", u32(header, 0x90))
        object.__setattr__(self, "offset_table", u32(header, 0xA4))
        object.__setattr__(self, "data_base", u32(header, 0xA8))

    def offsets(self) -> list[int]:
        """Read the dictionary record offset table."""
        with self.path.open("rb") as f:
            f.seek(self.offset_table)
            raw = f.read(4 * (self.count + 1))
        if len(raw) != 4 * (self.count + 1):
            raise ValueError(f"{self.path} offset table is truncated")
        return list(struct.unpack("<" + "i" * (self.count + 1), raw))

    def read_record(self, index: int) -> bytes:
        """Read one binary dictionary record by numeric index."""
        if index < 0 or index >= self.count:
            raise IndexError(index)
        offsets = self.offsets()
        start_offset = offsets[index] & 0x7FFFFFFF
        end_offset = offsets[index + 1] & 0x7FFFFFFF
        if end_offset < start_offset:
            raise ValueError(f"record {index} has a negative length")
        with self.path.open("rb") as f:
            f.seek(self.data_base + start_offset)
            return f.read(end_offset - start_offset)

    def extract_raw(self, out_dir: Path, limit: int | None = None) -> int:
        """Dump raw binary records for reverse-engineering diagnostics."""
        out_dir.mkdir(parents=True, exist_ok=True)
        offsets = self.offsets()
        total = self.count if limit is None else min(limit, self.count)
        written = 0
        with self.path.open("rb") as f:
            for index in range(total):
                start_offset = offsets[index] & 0x7FFFFFFF
                end_offset = offsets[index + 1] & 0x7FFFFFFF
                if end_offset <= start_offset:
                    continue
                f.seek(self.data_base + start_offset)
                (out_dir / f"{index:06d}.bin").write_bytes(
                    f.read(end_offset - start_offset)
                )
                written += 1
        return written


def xmlish_text(blob: bytes) -> str:
    """Return a rough printable/XML-like diagnostic view of a binary record."""
    text = blob.decode("latin1", errors="replace")
    text = re.sub(r"[^\x09\x0a\x0d\x20-\x7e]+", "\n", text)
    lines = [line.strip() for line in text.splitlines()]
    interesting = [
        line for line in lines if "<" in line or ">" in line or "SP1L2" in line
    ]
    return "\n".join(interesting)


def record_to_readable_xml(index: int, blob: bytes) -> str:
    """Wrap printable and binary runs from a record in readable XML."""
    lines = [f'<record index="{index}" size="{len(blob)}">']
    offset = 0
    while offset < len(blob):
        byte = blob[offset]
        if 32 <= byte <= 126:
            start = offset
            while offset < len(blob) and 32 <= blob[offset] <= 126:
                offset += 1
            text = blob[start:offset].decode("ascii", errors="replace")
            lines.append(f"  <text>{html.escape(text)}</text>")
            continue
        start = offset
        while offset < len(blob) and not 32 <= blob[offset] <= 126:
            offset += 1
        raw = blob[start:offset]
        lines.append(
            f'  <data offset="{start}" length="{len(raw)}" hex="{raw.hex(" ")}" />'
        )
    lines.append("</record>")
    return "\n".join(lines) + "\n"


def record_to_strings_xml(index: int, blob: bytes, min_length: int = 4) -> str:
    """Render only printable string runs from a binary record as XML."""
    lines = [f'<strings index="{index}" size="{len(blob)}">']
    offset = 0
    while offset < len(blob):
        byte = blob[offset]
        if 32 <= byte <= 126:
            start = offset
            while offset < len(blob) and 32 <= blob[offset] <= 126:
                offset += 1
            if offset - start >= min_length:
                text = blob[start:offset].decode("ascii", errors="replace")
                lines.append(f'  <s offset="{start}">{html.escape(text)}</s>')
            continue
        offset += 1
    lines.append("</strings>")
    return "\n".join(lines) + "\n"


def dump_readable_records(
    store: LingeaRecordStore, out_dir: Path, limit: int | None = None
) -> int:
    """Write readable binary-run XML diagnostics for record-store entries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = store.count if limit is None else min(limit, store.count)
    written = 0
    for index in range(total):
        blob = store.read_record(index)
        if not blob:
            continue
        (out_dir / f"{index:06d}.xml").write_text(
            record_to_readable_xml(index, blob), encoding="utf-8"
        )
        written += 1
    return written


def dump_string_records(
    store: LingeaRecordStore, out_dir: Path, limit: int | None = None
) -> int:
    """Write printable-string XML diagnostics for record-store entries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = store.count if limit is None else min(limit, store.count)
    written = 0
    for index in range(total):
        blob = store.read_record(index)
        if not blob:
            continue
        (out_dir / f"{index:06d}.xml").write_text(
            record_to_strings_xml(index, blob), encoding="utf-8"
        )
        written += 1
    return written


def dump_decoded_records(
    store: LingeaRecordStore, out_dir: Path, limit: int | None = None
) -> int:
    """Write decoded Lingea article XML chunks for record-store entries."""
    out_dir.mkdir(parents=True, exist_ok=True)
    total = store.count if limit is None else min(limit, store.count)
    written = 0
    for index in range(total):
        blob = store.read_record(index)
        chunks = decoded_article_chunks(blob)
        if not chunks:
            continue
        lines = [f'<decoded-record index="{index}" chunks="{len(chunks)}">']
        for chunk in chunks:
            lines.append(
                f'  <chunk offset="{chunk.offset}" '
                f'consumed="{chunk.consumed}"><![CDATA['
            )
            lines.append(chunk.text.replace("]]>", "]]]]><![CDATA[>"))
            lines.append("  ]]></chunk>")
        lines.append("</decoded-record>")
        (out_dir / f"{index:06d}.xml").write_text(
            "\n".join(lines) + "\n", encoding="utf-8"
        )
        written += 1
    return written


def dump_xmlish(
    store: LingeaRecordStore, out_file: Path, limit: int | None = None
) -> int:
    """Write one combined rough XML-like diagnostic text file."""
    out_file.parent.mkdir(parents=True, exist_ok=True)
    total = store.count if limit is None else min(limit, store.count)
    count = 0
    with out_file.open("w", encoding="utf-8", newline="\n") as out:
        for index in range(total):
            text = xmlish_text(store.read_record(index))
            if not text:
                continue
            out.write(f"\n<!-- record {index} -->\n")
            out.write(text)
            out.write("\n")
            count += 1
    return count


def extract_media(
    data_dir: Path, out_dir: Path, limit: int | None, images: bool, audio: bool
) -> None:
    """Extract image and audio container files from the data directory."""
    targets = [
        ("oup_en-dic.mbx", "images"),
        ("oup_en-dic.wbx", "audio"),
    ]
    for filename, subdir in targets:
        if subdir == "images" and not images:
            continue
        if subdir == "audio" and not audio:
            continue
        box = LingeaBox(data_dir / filename)
        written = box.extract(out_dir / subdir, limit=limit)
        print(f"{filename}: wrote {written} files to {out_dir / subdir}")


def extract_dictionary(
    data_dir: Path, out_dir: Path, limit: int | None, diagnostics: bool
) -> None:
    """Decode dictionary LLD records and optionally write diagnostic views."""
    store = LingeaRecordStore(data_dir / "oup_en-dic.lld")
    decoded_count = dump_decoded_records(
        store, out_dir / "dic_records_decoded", limit=limit
    )
    print(
        "oup_en-dic.lld: wrote "
        f"{decoded_count} decoded XML records to {out_dir / 'dic_records_decoded'}"
    )
    if diagnostics:
        readable_count = dump_readable_records(
            store, out_dir / "dic_records_readable", limit=limit
        )
        strings_count = dump_string_records(
            store, out_dir / "dic_records_strings", limit=limit
        )
        xmlish_count = dump_xmlish(
            store, out_dir / "dic_records_xmlish.txt", limit=limit
        )
        print(
            "oup_en-dic.lld: wrote "
            f"{readable_count} readable XML records to {out_dir / 'dic_records_readable'}"
        )
        print(
            "oup_en-dic.lld: wrote "
            f"{strings_count} string XML records to {out_dir / 'dic_records_strings'}"
        )
        print(
            "oup_en-dic.lld: wrote "
            f"xmlish view for {xmlish_count} records to {out_dir / 'dic_records_xmlish.txt'}"
        )


def main() -> int:
    """Run the command-line extractor."""
    parser = argparse.ArgumentParser(
        description="Extract media and dictionary records from OALD9 Lingea data files."
    )
    parser.add_argument("--data-dir", default="data", type=Path)
    parser.add_argument("--out-dir", default="extracted", type=Path)
    parser.add_argument(
        "--media", action="store_true", help="extract MBX/WBX images and audio"
    )
    parser.add_argument("--images", action="store_true", help="extract only MBX images")
    parser.add_argument("--audio", action="store_true", help="extract only WBX audio")
    parser.add_argument(
        "--dictionary", action="store_true", help="dump dictionary LLD records"
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="also write raw-ish decoder diagnostic views",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="limit entries per container for testing",
    )
    args = parser.parse_args()

    if not args.media and not args.images and not args.audio and not args.dictionary:
        args.media = True
        args.dictionary = True

    if args.media or args.images or args.audio:
        images = args.media or args.images
        audio = args.media or args.audio
        extract_media(
            args.data_dir,
            args.out_dir / "media",
            args.limit,
            images=images,
            audio=audio,
        )
    if args.dictionary:
        extract_dictionary(
            args.data_dir,
            args.out_dir / "dictionary",
            args.limit,
            diagnostics=args.diagnostics,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
