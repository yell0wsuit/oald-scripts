#!/usr/bin/env python3
"""
Decompress the LZ4 transport layer of SlovoEd SDC chunks (e.g. ARTD article data).

In an SLD2/SDC container, a chunk whose size field has the high bit
set is LZ4-compressed. Its payload is:

  0x00: uint16 version (observed: 1)
  0x02: uint16 padding
  0x04: uint32 uncompressed size
  0x08: LZ4 block (raw block format, no frame header)

This reproduces the SlovoEd engine's decompressor (standard LZ4 block
format) to recover the raw, uncompressed chunk bytes.
"""

from __future__ import annotations

import argparse
import struct
from pathlib import Path


def lz4_block_decompress(src: bytes, expected_size: int | None = None) -> bytes:
    """Decompress a raw LZ4 block.

    Args:
        src: The LZ4 block bytes (no frame header).
        expected_size: Optional decompressed size to validate against.

    Returns:
        The decompressed bytes.

    Raises:
        ValueError: On a malformed stream (bad match offset or size mismatch).
    """
    out = bytearray()
    i = 0
    n = len(src)
    while i < n:
        token = src[i]
        i += 1

        # Literal run length (high nibble), extended by 0xFF bytes.
        lit = token >> 4
        if lit == 15:
            while True:
                b = src[i]
                i += 1
                lit += b
                if b != 255:
                    break
        out += src[i : i + lit]
        i += lit
        if i >= n:
            break

        # Match: 2-byte little-endian back offset, then length (low nibble + 4).
        offset = src[i] | (src[i + 1] << 8)
        i += 2
        match_len = token & 0xF
        if match_len == 15:
            while True:
                b = src[i]
                i += 1
                match_len += b
                if b != 255:
                    break
        match_len += 4

        start = len(out) - offset
        if start < 0:
            raise ValueError(f"bad match offset {offset} at output {len(out)}")
        # Byte-by-byte copy (offset may be < match_len, i.e. overlapping).
        for j in range(match_len):
            out.append(out[start + j])

    if expected_size is not None and len(out) != expected_size:
        raise ValueError(f"decompressed size {len(out)} != expected {expected_size}")
    return bytes(out)


def decompress_chunk(payload: bytes) -> bytes:
    """Decompress a compressed SDC chunk payload (8-byte header + LZ4 block)."""
    if len(payload) < 8:
        raise ValueError("chunk payload too small for compression header")
    version = struct.unpack_from("<H", payload, 0)[0]
    uncompressed_size = struct.unpack_from("<I", payload, 4)[0]
    if version != 1:
        raise ValueError(f"unexpected compression version {version}")
    return lz4_block_decompress(payload[8:], uncompressed_size)


def iter_chunks(buf: bytes):
    """Yield ``(tag, index, size, offset, compressed)`` for each chunk table entry."""
    table_offset = struct.unpack_from("<I", buf, 0x04)[0]
    chunk_count = struct.unpack_from("<I", buf, 0x18)[0]
    entry_size = struct.unpack_from("<I", buf, 0x1C)[0]
    for ordinal in range(chunk_count):
        eo = table_offset + ordinal * entry_size
        tag = buf[eo : eo + 4].decode("ascii", "replace")
        index, raw_size, offset = struct.unpack_from("<III", buf, eo + 4)
        compressed = bool(raw_size & 0x80000000)
        size = raw_size & 0x7FFFFFFF
        yield tag, index, size, offset, compressed


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: decompress matching chunks from an SDC file to disk."""
    parser = argparse.ArgumentParser(
        description="Decompress LZ4 (e.g. ARTD) chunks from an SDC file."
    )
    parser.add_argument("sdc", type=Path, help="Path to the .sdc file")
    parser.add_argument(
        "-t", "--tag", default="ARTD", help="Chunk tag to extract (default: ARTD)"
    )
    parser.add_argument(
        "-o", "--out", type=Path, default=Path("artd_raw"), help="Output directory"
    )
    args = parser.parse_args(argv)

    buf = args.sdc.read_bytes()
    if buf[:4] != b"SLD2":
        raise SystemExit(f"{args.sdc}: not an SLD2 SDC file")

    args.out.mkdir(parents=True, exist_ok=True)
    count = 0
    for tag, index, size, offset, compressed in iter_chunks(buf):
        if tag != args.tag:
            continue
        payload = buf[offset : offset + size]
        data = decompress_chunk(payload) if compressed else payload
        (args.out / f"{tag}_{index:04d}.bin").write_bytes(data)
        count += 1
    print(f"{args.sdc}: wrote {count} {args.tag} chunk(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
