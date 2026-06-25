#!/usr/bin/env python3
"""
Decode SlovoEd ``xTST`` quick-access headword index chunks from an SDC file.

Each search list in the dictionary has a per-letter ``xTST`` chunk (``ATST``,
``BTST``, ... ``OTST``). It is a binary-search "quick access" index: a balanced
tree of representative headwords laid out in array order (127 = 2^7 - 1 nodes),
used to jump near a word without scanning the whole list. It is NOT the complete
headword list -- it only samples representative words at the tree's node
positions.

Chunk layout (after LZ4 transport, see decode_artd.py):

  0x00: uint32 record count
  then, per record:
    uint32 rank   - the word's article position in the full sorted list
    uint16 index  - tree node index (0 for leaf-level samples)
    uint16 value  - tree node weight/depth marker
    UTF-16LE string, terminated by a 0x0000 code unit

Headwords may contain a homograph digit (e.g. "davis4") and an occasional
U+2744 marker glyph; both are preserved verbatim.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from decode_artd import decompress_chunk, iter_chunks


def parse_tst(data: bytes) -> list[dict]:
    """Parse a decompressed ``xTST`` chunk into a list of record dicts."""
    count = struct.unpack_from("<I", data, 0)[0]
    off = 4
    records: list[dict] = []
    for _ in range(count):
        if off + 8 > len(data):
            break
        rank, index, value = struct.unpack_from("<IHH", data, off)
        off += 8
        start = off
        while off + 1 < len(data) and not (data[off] == 0 and data[off + 1] == 0):
            off += 2
        word = data[start:off].decode("utf-16le", "replace")
        off += 2  # skip the 0x0000 terminator
        records.append({"rank": rank, "index": index, "value": value, "word": word})
    return records


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: dump ``xTST`` headword indexes from an SDC file."""
    parser = argparse.ArgumentParser(
        description="Decode xTST quick-access headword index chunks from an SDC file."
    )
    parser.add_argument("sdc", type=Path, help="Path to the .sdc file")
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("tst.json"), help="Output JSON file"
    )
    args = parser.parse_args(argv)

    buf = args.sdc.read_bytes()
    if buf[:4] != b"SLD2":
        raise SystemExit(f"{args.sdc}: not an SLD2 SDC file")

    result: dict[str, list[dict]] = {}
    for tag, index, size, offset, compressed in iter_chunks(buf):
        if not (len(tag) == 4 and tag.endswith("TST")):
            continue
        payload = buf[offset : offset + size]
        data = decompress_chunk(payload) if compressed else payload
        result[f"{tag}_{index}"] = parse_tst(data)

    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), "utf-8")
    total = sum(len(v) for v in result.values())
    print(f"{args.sdc}: decoded {len(result)} xTST chunk(s), {total} words -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
