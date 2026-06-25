#!/usr/bin/env python3
"""
Decode SlovoEd article translations (ARTD) from an SDC dictionary.

This reverses the full article pipeline, reproduced from the SlovoEd CE engine
source (CSldBitInput / CSldInputBase / sld2::decoders::CharChain):

  1. LZ4 transport (see decode_artd.py) gives the raw ARTD byte stream.
  2. The stream is a bit stream (LSB-first over little-endian 32-bit words).
  3. Each article is a sequence of styled text blocks. The article is decoded by:
       a. read a block-style sequence with tree 0 (SLD_DECORER_TYPE_STYLES),
          terminated by a NUL "char";
       b. for each style index `s`, decode that block's text with tree `s`.
     Every tree is a "CharChain" table: read `CodeSize` bits -> code ->
     (Shift, Len) -> copy `Len` UTF-16 chars from a shared char pool.
  4. Article start positions come from the ARTQ quick-access table; the stored
     bit offsets are scrambled with SLD_QA_SHIFT_RESTORE(shift, HASH), where the
     real HASH = header.HASH ^ DictID ^ NumberOfArticles (the compiler stores a
     masked HASH; see Compress.cpp / SldDefines.h).

Output blocks are (style_index, text). Text blocks carry the visible content;
other styles carry metadata (links, sound refs, CSS switches). Rendering them
into clean HTML needs the style table (ARTS) and is out of scope here.
"""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

from decode_artd import decompress_chunk, iter_chunks


def _fourcc(value: int) -> str:
    return struct.pack("<I", value).decode("latin1")


class _BitInput:
    """LSB-first bit reader over a byte stream (matches CSldBitInput)."""

    __slots__ = ("data", "pos")

    def __init__(self, data: bytes, pos: int = 0):
        self.data = data
        self.pos = pos

    def read(self, n: int) -> int:
        value = 0
        pos = self.pos
        data = self.data
        for k in range(n):
            value |= ((data[pos >> 3] >> (pos & 7)) & 1) << k
            pos += 1
        self.pos = pos
        return value


def _charchain_decode(bits: _BitInput, tree, max_len: int = 1 << 16) -> str:
    """Decode one NUL-terminated string with a CharChain tree."""
    code_size, num_chars, code_table, chars = tree
    out: list[int] = []
    while len(out) < max_len:
        code = bits.read(code_size)
        if code >= num_chars:
            break
        shift, length = code_table[code]
        terminated = False
        for i in range(length):
            c = chars[shift + i]
            if c == 0:
                terminated = True
                break
            out.append(c)
        if terminated:
            break
    return "".join(map(chr, out))


def _qa_restore(shift: int, hash_: int) -> int:
    x = shift ^ hash_
    return (
        ((x & 0o22222222222) >> 1)
        | ((x & 0o11111111111) << 1)
        | (shift & 0o44444444444)
    )


class ArticleDecoder:
    """Decodes article translations from a parsed SDC dictionary."""

    def __init__(self, sdc_path: Path):
        buf = sdc_path.read_bytes()
        if buf[:4] != b"SLD2":
            raise ValueError(f"{sdc_path}: not an SLD2 SDC file")

        # Group chunks by tag, keeping (index -> (size, offset, compressed)).
        by_tag: dict[str, dict[int, tuple[int, int, bool]]] = {}
        for tag, index, size, offset, compressed in iter_chunks(buf):
            by_tag.setdefault(tag, {})[index] = (size, offset, compressed)
        self._buf = buf
        self._by_tag = by_tag

        head = self._chunk("HEAD", 0)
        fields = struct.unpack_from("<18I", head, 0)
        # real HASH = header.HASH ^ DictID ^ NumberOfArticles (see Compress.cpp)
        self.hash = fields[4] ^ fields[5] ^ fields[12]
        data_tag = _fourcc(fields[8])
        qa_tag = _fourcc(fields[9])
        tree_tag = _fourcc(fields[10])
        self.num_articles = fields[12]
        if fields[13] != 2:
            raise ValueError(
                f"unsupported ArticlesCompressionMethod {fields[13]} (expected 2/CharChain)"
            )

        self._trees = self._load_trees(tree_tag)
        self._stream = self._build_stream(data_tag)
        self._qa = self._load_qa(qa_tag)

    def _chunk(self, tag: str, index: int) -> bytes:
        size, offset, compressed = self._by_tag[tag][index]
        payload = self._buf[offset : offset + size]
        return decompress_chunk(payload) if compressed else payload

    def _load_trees(self, tag: str) -> dict[int, tuple]:
        trees = {}
        for index in self._by_tag[tag]:
            t = self._chunk(tag, index)
            _struct_size, code_size, num_chars, _ = struct.unpack_from("<4I", t, 0)
            code_table = [
                struct.unpack_from("<HH", t, 16 + 4 * k) for k in range(num_chars)
            ]
            chars = memoryview(t[16 + 4 * num_chars :]).cast("H")
            trees[index] = (code_size, num_chars, code_table, chars)
        return trees

    def _build_stream(self, tag: str) -> bytes:
        stream = bytearray()
        for index in sorted(self._by_tag[tag]):
            stream += self._chunk(tag, index)
        return bytes(stream)

    def _load_qa(self, tag: str) -> list[tuple[int, int]]:
        q = self._chunk(tag, 0)
        header_size, entry_size, _ver, _type, count = struct.unpack_from("<5I", q, 0)
        entries = []
        for k in range(count):
            index, shift = struct.unpack_from("<II", q, header_size + k * entry_size)
            entries.append((index, _qa_restore(shift, self.hash)))
        return entries

    def _read_article(self, bits: _BitInput) -> list[tuple[int, str]]:
        styles = _charchain_decode(bits, self._trees[0])
        blocks = []
        for ch in styles:
            style = ord(ch)
            text = (
                _charchain_decode(bits, self._trees[style])
                if style in self._trees
                else ""
            )
            blocks.append((style, text))
        return blocks

    def decode(self, article_index: int) -> list[tuple[int, str]]:
        """Decode the article at the given (0-based) global article index."""
        if not 0 <= article_index < self.num_articles:
            raise IndexError(article_index)
        # Largest quick-access point at or before the requested article.
        start_index, start_pos = 0, self._qa[0][1]
        for index, pos in self._qa:
            if index <= article_index:
                start_index, start_pos = index, pos
            else:
                break
        bits = _BitInput(self._stream, start_pos)
        blocks: list[tuple[int, str]] = []
        for _ in range(start_index, article_index + 1):
            blocks = self._read_article(bits)
        return blocks


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: decode article(s) to JSON blocks."""
    parser = argparse.ArgumentParser(
        description="Decode SlovoEd ARTD articles from an SDC file."
    )
    parser.add_argument("sdc", type=Path, help="Path to the .sdc file")
    parser.add_argument("index", type=int, help="Article index to decode")
    parser.add_argument(
        "-n", "--count", type=int, default=1, help="Number of consecutive articles"
    )
    parser.add_argument(
        "-o", "--out", type=Path, help="Write JSON here instead of stdout"
    )
    args = parser.parse_args(argv)

    dec = ArticleDecoder(args.sdc)
    result = {
        i: [{"style": s, "text": t} for s, t in dec.decode(i)]
        for i in range(args.index, args.index + args.count)
    }
    text = json.dumps(result, ensure_ascii=False, indent=2)
    if args.out:
        args.out.write_text(text, "utf-8")
        print(f"wrote {len(result)} article(s) -> {args.out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
