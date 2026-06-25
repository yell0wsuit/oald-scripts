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

Output blocks are (style_index, text). ARTS style records are decoded as well,
so entries can be rendered as conservative style-annotated HTML or reduced to
headword/definition JSON fields.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from html import escape
import json
import struct
from pathlib import Path
from typing import Iterator

from decode_artd import decompress_chunk, iter_chunks

META_NAMES = [
    "eMetaText",
    "eMetaPhonetics",
    "eMetaImage",
    "eMetaSound",
    "eMetaTable",
    "eMetaTableRow",
    "eMetaTableCol",
    "eMetaParagraph",
    "eMetaLabel",
    "eMetaLink",
    "eMetaHide",
    "eMetaHideControl",
    "eMetaTest",
    "eMetaTestInput",
    "eMetaTestToken",
    "eMetaPopupImage",
    "eMetaUrl",
    "eMetaUiElement",
    "eMetaPopupArticle",
    "eMetaNoBrText",
    "eMetaInfoBlock",
    "eMetaBackgroundImage",
    "eMetaFlashCardsLink",
    "eMetaVideo",
    "eMetaScene",
    "eMetaImageArea",
    "eMetaSlideShow",
    "eMetaVideoSource",
    "eMetaMediaContainer",
    "eMetaTestSpear",
    "eMetaTestTarget",
    "eMetaTestControl",
    "eMetaSwitch",
    "eMetaSwitchControl",
    "eMetaSwitchState",
    "eMetaManagedSwitch",
    "eMetaDiv",
    "eMetaMap",
    "eMetaMapElement",
    "eMetaCaption",
    "eMetaTestResult",
    "eMetaTestResultElement",
    "eMetaTextControl",
    "eMetaTaskBlockEntry",
    "eMetaSidenote",
    "eMetaConstructionSet",
    "eMetaDrawingBlock",
    "eMetaArticleEventHandler",
    "eMetaDemoLink",
    "eMeta_UnusedBroken",
    "eMetaTestContainer",
    "eMetaLegendItem",
    "eMetaAtomicObject",
    "eMetaCrossword",
    "eMetaExternArticle",
    "eMetaList",
    "eMetaLi",
    "eMetaInteractiveObject",
    "eMeta_Unused0",
    "eMetaTimeLine",
    "eMetaTimeLineItem",
    "eMetaAbstractResource",
    "eMetaFormula",
    "eMetaCrosswordHint",
    "eMetaFootnoteBrief",
    "eMetaFootnoteTotal",
]

FONT_FAMILIES = {
    0: "sans-serif",
    1: "serif",
    2: "fantasy",
    3: "monospace",
}


def _fourcc(value: int) -> str:
    return struct.pack("<I", value).decode("latin1")


def _read_u16z(buf: bytes) -> str:
    end = 0
    while end + 1 < len(buf) and buf[end : end + 2] != b"\0\0":
        end += 2
    return buf[:end].decode("utf-16le", "replace")


def _fixup_block_text(text: str) -> str:
    if text[-4:].lower() == r"\%0a":
        return text[:-4] + "\n"
    return text


def _html_attr(text: str) -> str:
    safe = "".join(
        ch if ord(ch) >= 32 or ch in "\t\n\r" else f"\\x{ord(ch):02x}" for ch in text
    )
    return escape(safe, quote=True)


def _clean_entry_text(text: str) -> str:
    return " ".join(text.replace("\u200b", "").split())


def _normalize_headword(text: str) -> str:
    return (
        _clean_entry_text(text)
        .replace("·", "")
        .replace("-", "")
        .replace(" ", "")
        .replace("❄", "")
        .lower()
    )


@dataclass(frozen=True)
class StyleInfo:
    """Decoded ARTS style record, using the default variant."""

    index: int
    tag: str
    usages: tuple[int, ...]
    variant_type: int
    visible: bool
    meta_type: int
    level: int
    color: tuple[int, int, int, int]
    background: tuple[int, int, int, int]
    bold: int
    italic: bool
    underline: int
    strikethrough: bool
    text_size: int
    line_height: int
    font_family: int
    font_name: int
    prefix: str
    postfix: str
    overline: bool
    unclickable: bool

    @property
    def meta_name(self) -> str:
        if 0 <= self.meta_type < len(META_NAMES):
            return META_NAMES[self.meta_type]
        if self.meta_type == 0xFFFF:
            return "eMetaUnknown"
        return f"eMeta_{self.meta_type}"

    def css(self) -> str:
        props: list[str] = []
        r, g, b, a = self.color
        if a and (r, g, b) != (0, 0, 0):
            props.append(f"color: #{r:02x}{g:02x}{b:02x}")
        br, bg, bb, ba = self.background
        if ba and (br, bg, bb) != (0, 0, 0):
            props.append(f"background-color: #{br:02x}{bg:02x}{bb:02x}")
        if self.bold:
            props.append("font-weight: bold")
        if self.italic:
            props.append("font-style: italic")
        decoration = []
        if self.underline:
            decoration.append("underline")
        if self.strikethrough:
            decoration.append("line-through")
        if self.overline:
            decoration.append("overline")
        if decoration:
            props.append(f"text-decoration: {' '.join(decoration)}")
        if self.font_family in FONT_FAMILIES:
            props.append(f"font-family: {FONT_FAMILIES[self.font_family]}")
        if self.text_size > 5 and self.text_size != 0xFFFFFFFF:
            props.append(f"font-size: {self.text_size}pt")
        if self.line_height > 5 and self.line_height != 0xFFFFFFFF:
            props.append(f"line-height: {self.line_height}pt")
        return "; ".join(props)


@dataclass(frozen=True)
class SearchWordList:
    """Decoded accessors for one SlovoEd search word list."""

    prefix: str
    word_count: int
    article_index_bits: int
    word_offsets: tuple[int, ...]
    word_data: bytes
    index_data: bytes
    tree: tuple

    def word_at(self, index: int) -> str:
        return _charchain_decode(
            _BitInput(self.word_data, self.word_offsets[index]), self.tree
        )

    def article_index_at(self, index: int) -> int:
        return _BitInput(self.index_data, index * self.article_index_bits).read(
            self.article_index_bits
        )

    def find_word_indices(self, word: str) -> Iterator[int]:
        needle = _normalize_headword(word)
        lo = 0
        hi = min(self.word_count, len(self.word_offsets))
        while lo < hi:
            mid = (lo + hi) // 2
            if _normalize_headword(self.word_at(mid)) < needle:
                lo = mid + 1
            else:
                hi = mid

        index = lo
        limit = min(self.word_count, len(self.word_offsets))
        while index < limit:
            current = _normalize_headword(self.word_at(index))
            if current != needle:
                break
            yield index
            index += 1


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
        style_tag = _fourcc(fields[11])
        self.num_articles = fields[12]
        if fields[13] != 2:
            raise ValueError(
                f"unsupported ArticlesCompressionMethod {fields[13]} (expected 2/CharChain)"
            )

        self._trees = self._load_trees(tree_tag)
        self.styles = self._load_styles(style_tag)
        self._stream = self._build_stream(data_tag)
        self._qa = self._load_qa(qa_tag)
        self._search_lists: dict[str, SearchWordList | None] = {}
        self._search_list_prefixes = self._discover_search_list_prefixes()

    def _chunk(self, tag: str, index: int) -> bytes:
        size, offset, compressed = self._by_tag[tag][index]
        payload = self._buf[offset : offset + size]
        return decompress_chunk(payload) if compressed else payload

    def _discover_search_list_prefixes(self) -> tuple[str, ...]:
        prefixes = {
            tag[0]
            for tag in self._by_tag
            if len(tag) == 4 and tag.endswith("INH") and tag[0].isalpha()
        }
        compatible = [
            prefix
            for prefix in prefixes
            if all(
                f"{prefix}{suffix}" in self._by_tag
                for suffix in ("INH", "IND", "DAT", "SDT", "TRE")
            )
        ]
        return tuple(sorted(compatible, key=lambda prefix: (prefix != "A", prefix)))

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

    def _load_search_word_list(self, prefix: str = "A") -> SearchWordList | None:
        if prefix in self._search_lists:
            return self._search_lists[prefix]

        required = [
            f"{prefix}INH",
            f"{prefix}IND",
            f"{prefix}DAT",
            f"{prefix}SDT",
            f"{prefix}TRE",
        ]
        if any(tag not in self._by_tag for tag in required):
            self._search_lists[prefix] = None
            return None

        header = self._chunk(f"{prefix}INH", 0)
        if len(header) < 28:
            self._search_lists[prefix] = None
            return None
        fields = struct.unpack_from("<7I", header, 0)
        word_count = fields[2]
        article_index_bits = fields[6]
        if article_index_bits <= 0:
            self._search_lists[prefix] = None
            return None

        offset_data = self._build_stream(f"{prefix}SDT")
        word_offsets = tuple(
            struct.unpack_from("<I", offset_data, i * 4)[0]
            for i in range(len(offset_data) // 4)
        )
        trees = self._load_trees(f"{prefix}TRE")
        if 1 not in trees:
            self._search_lists[prefix] = None
            return None

        word_list = SearchWordList(
            prefix=prefix,
            word_count=word_count,
            article_index_bits=article_index_bits,
            word_offsets=word_offsets,
            word_data=self._build_stream(f"{prefix}DAT"),
            index_data=self._build_stream(f"{prefix}IND"),
            tree=trees[1],
        )
        self._search_lists[prefix] = word_list
        return word_list

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

    def _load_styles(self, tag: str) -> list[StyleInfo]:
        data = bytearray()
        for index in sorted(self._by_tag.get(tag, {})):
            data += self._chunk(tag, index)

        styles: list[StyleInfo] = []
        off = 0
        index = 0
        while off < len(data):
            struct_size, total_size, _language, variants_count = struct.unpack_from(
                "<4I", data, off
            )
            if struct_size == 0 or total_size == 0:
                break
            if struct_size < 128 or off + total_size > len(data):
                raise ValueError(
                    f"bad ARTS style {index}: struct=0x{struct_size:x} total=0x{total_size:x}"
                )
            (
                _struct_size,
                _total_size,
                _language,
                _variants_count,
                variant_size,
                default_variant,
                usage_count,
                usage_size,
            ) = struct.unpack_from("<8I", data, off)
            tag_name = _read_u16z(data[off + 32 : off + 96])

            cursor = off + struct_size
            usages = tuple(
                struct.unpack_from("<I", data, cursor + i * usage_size)[0]
                for i in range(usage_count)
            )
            cursor += usage_count * usage_size

            if variants_count == 0:
                styles.append(
                    StyleInfo(
                        index=index,
                        tag=tag_name,
                        usages=usages,
                        variant_type=0,
                        visible=False,
                        meta_type=0xFFFF,
                        level=0,
                        color=(0, 0, 0, 0),
                        background=(0, 0, 0, 0),
                        bold=0,
                        italic=False,
                        underline=0,
                        strikethrough=False,
                        text_size=0,
                        line_height=0,
                        font_family=0xFFFF,
                        font_name=0xFFFF,
                        prefix="",
                        postfix="",
                        overline=False,
                        unclickable=False,
                    )
                )
            else:
                variant_index = min(default_variant, variants_count - 1)
                v = cursor + variant_index * variant_size
                fields = struct.unpack_from("<22I", data, v)
                styles.append(
                    StyleInfo(
                        index=index,
                        tag=tag_name,
                        usages=usages,
                        variant_type=fields[1],
                        visible=bool(fields[2]),
                        meta_type=fields[3],
                        level=fields[4],
                        color=(fields[5], fields[6], fields[7], fields[8]),
                        background=(fields[9], fields[10], fields[11], fields[12]),
                        bold=fields[13],
                        italic=bool(fields[14]),
                        underline=fields[15],
                        strikethrough=bool(fields[16]),
                        text_size=fields[17],
                        line_height=fields[18],
                        font_family=fields[19],
                        font_name=fields[20],
                        prefix=_read_u16z(data[v + 84 : v + 118]),
                        postfix=_read_u16z(data[v + 118 : v + 152]),
                        overline=bool(struct.unpack_from("<I", data, v + 152)[0]),
                        unclickable=bool(struct.unpack_from("<I", data, v + 164)[0]),
                    )
                )

            off += total_size
            index += 1
        return styles

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

    def _html_from_blocks(
        self, article_index: int, blocks: list[tuple[int, str]]
    ) -> str:
        parts = [f'<article data-index="{article_index}">']
        for style_index, raw_text in blocks:
            style = self.styles[style_index] if style_index < len(self.styles) else None
            text = raw_text
            if style and style.meta_type in (0, 1):
                text = style.prefix + _fixup_block_text(text) + style.postfix
            if not text:
                continue

            if style and style.meta_type not in (0, 1):
                parts.append(
                    "<span hidden"
                    f' data-style="{style_index}"'
                    f' data-meta="{escape(style.meta_name)}"'
                    f' data-raw="{_html_attr(text)}"></span>'
                )
                continue

            meta = escape(style.meta_name if style else "eMetaUnknown")
            cls = f"sld-style s{style_index}"
            if style and style.meta_type == 1:
                cls += " phonetics"
            css = style.css() if style else ""
            attrs = [
                f'class="{cls}"',
                f'data-style="{style_index}"',
                f'data-meta="{meta}"',
            ]
            if style and style.tag:
                attrs.append(f'data-tag="{_html_attr(style.tag)}"')
            if css:
                attrs.append(f'style="{_html_attr(css)}"')
            parts.append(f"<span {' '.join(attrs)}>{escape(text)}</span>")
        parts.append("</article>")
        return "\n".join(parts)

    def decode_html(self, article_index: int) -> str:
        """Decode one article into conservative, style-annotated HTML."""
        return self._html_from_blocks(article_index, self.decode(article_index))

    def _entry_from_blocks(
        self, article_index: int, blocks: list[tuple[int, str]]
    ) -> dict:
        headword_parts: list[str] = []
        part_of_speech: list[str] = []
        phonetics: list[str] = []
        definitions: list[str] = []
        word_origin: list[str] = []

        in_word_origin = False
        for style_index, raw_text in blocks:
            if style_index >= len(self.styles):
                continue
            style = self.styles[style_index]
            if style.meta_type not in (0, 1):
                continue
            text = _fixup_block_text(raw_text)
            tag = style.tag
            if not text:
                continue
            if tag in {"h", "h_dot", "h_sup", "hm-g"}:
                headword_parts.append(text)
            elif tag in {"pos", "pos-g"}:
                cleaned = _clean_entry_text(text)
                if cleaned:
                    part_of_speech.append(cleaned)
            elif tag == "phon":
                cleaned = _clean_entry_text(text)
                if cleaned:
                    phonetics.append(cleaned)
            elif tag == "def":
                cleaned = _clean_entry_text(text)
                if cleaned:
                    definitions.append(cleaned)
            elif "Word Origin" in text:
                in_word_origin = True
            elif in_word_origin and tag in {
                "body",
                "ff",
                "lang",
                "etym_i",
                "qt",
                "pnc",
                "tr_e",
            }:
                cleaned = _clean_entry_text(text)
                if cleaned:
                    word_origin.append(cleaned)

        return {
            "index": article_index,
            "headword": _clean_entry_text("".join(headword_parts)),
            "part_of_speech": " ".join(dict.fromkeys(part_of_speech)),
            "phonetics": list(dict.fromkeys(phonetics)),
            "definitions": definitions,
            "word_origin": _clean_entry_text(" ".join(word_origin)),
        }

    def decode_entry(self, article_index: int) -> dict:
        """Decode one main dictionary article into headword/definition fields."""
        return self._entry_from_blocks(article_index, self.decode(article_index))

    def iter_articles(
        self, start: int = 0, limit: int | None = None
    ) -> "Iterator[tuple[int, list[tuple[int, str]]]]":
        """Yield decoded articles sequentially from the ARTD stream."""
        if start < 0 or start >= self.num_articles:
            raise IndexError(start)
        start_index, start_pos = 0, self._qa[0][1]
        for index, pos in self._qa:
            if index <= start:
                start_index, start_pos = index, pos
            else:
                break

        bits = _BitInput(self._stream, start_pos)
        emitted = 0
        for article_index in range(start_index, self.num_articles):
            blocks = self._read_article(bits)
            if article_index < start:
                continue
            yield article_index, blocks
            emitted += 1
            if limit is not None and emitted >= limit:
                break

    def iter_entries(
        self, start: int = 0, limit: int | None = None, include_html: bool = False
    ) -> "Iterator[dict]":
        """Yield main dictionary entries with definitions."""
        emitted = 0
        for article_index, blocks in self.iter_articles(start):
            entry = self._entry_from_blocks(article_index, blocks)
            if not entry["headword"] or not entry["definitions"]:
                continue
            if include_html:
                entry = dict(entry)
                entry["html"] = self._html_from_blocks(article_index, blocks)
            yield entry
            emitted += 1
            if limit is not None and emitted >= limit:
                break

    def _indexed_article_indices(self, word: str) -> Iterator[int]:
        seen: set[int] = set()
        for prefix in self._search_list_prefixes:
            word_list = self._load_search_word_list(prefix)
            if not word_list:
                continue
            for word_index in word_list.find_word_indices(word):
                article_index = word_list.article_index_at(word_index)
                if article_index in seen or not 0 <= article_index < self.num_articles:
                    continue
                seen.add(article_index)
                yield article_index

    def find_entries(
        self, word: str, limit: int | None = None, include_html: bool = False
    ) -> "Iterator[dict]":
        """Find main dictionary entries whose normalized headword matches word."""
        needle = _normalize_headword(word)
        emitted = 0
        seen: set[int] = set()
        for article_index in self._indexed_article_indices(word):
            entry = self.decode_entry(article_index)
            if _normalize_headword(entry["headword"]) != needle:
                continue
            if not entry["definitions"]:
                continue
            if include_html:
                entry = dict(entry)
                entry["html"] = self.decode_html(article_index)
            seen.add(article_index)
            yield entry
            emitted += 1
            if limit is not None and emitted >= limit:
                return

        for entry in self.iter_entries(include_html=include_html):
            if entry["index"] in seen:
                continue
            if _normalize_headword(entry["headword"]) != needle:
                continue
            yield entry
            emitted += 1
            if limit is not None and emitted >= limit:
                break


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: decode article(s) to JSON blocks."""
    parser = argparse.ArgumentParser(
        description="Decode SlovoEd ARTD articles from an SDC file."
    )
    parser.add_argument("sdc", type=Path, help="Path to the .sdc file")
    parser.add_argument("index", type=int, nargs="?", help="Article index to decode")
    parser.add_argument(
        "-n", "--count", type=int, default=1, help="Number of consecutive articles"
    )
    parser.add_argument(
        "-o", "--out", type=Path, help="Write JSON here instead of stdout"
    )
    parser.add_argument(
        "--json-out", type=Path, help="Write JSON output here instead of stdout"
    )
    parser.add_argument("--html-out", type=Path, help="Write rendered HTML output here")
    parser.add_argument(
        "--with-html",
        action="store_true",
        help="Include rendered HTML in JSON entry output",
    )
    parser.add_argument(
        "--html", action="store_true", help="Render best-effort HTML instead of JSON"
    )
    parser.add_argument(
        "--entry",
        action="store_true",
        help="Render dictionary entry fields instead of raw blocks",
    )
    parser.add_argument("--word", help="Find dictionary entries by headword")
    parser.add_argument(
        "--dump-entries",
        action="store_true",
        help="Stream all main dictionary entries as JSON Lines",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit --word matches or --dump-entries output",
    )
    args = parser.parse_args(argv)

    dec = ArticleDecoder(args.sdc)
    json_out = args.json_out or args.out

    if args.word:
        include_html = args.with_html or bool(args.html_out)
        matches = list(
            dec.find_entries(args.word, args.limit, include_html=include_html)
        )
        text = json.dumps(matches, ensure_ascii=False, indent=2)
        if args.html_out:
            args.html_out.write_text(
                "\n".join(match["html"] for match in matches), "utf-8"
            )
            print(f"wrote HTML -> {args.html_out}")
            if not args.with_html:
                matches = [
                    {k: v for k, v in match.items() if k != "html"} for match in matches
                ]
                text = json.dumps(matches, ensure_ascii=False, indent=2)
    elif args.dump_entries:
        include_html = args.with_html or bool(args.html_out)
        entries = dec.iter_entries(limit=args.limit, include_html=include_html)
        if json_out:
            html_file = (
                args.html_out.open("w", encoding="utf-8") if args.html_out else None
            )
            with json_out.open("w", encoding="utf-8") as f:
                count = 0
                try:
                    for entry in entries:
                        if html_file:
                            html_file.write(entry["html"] + "\n")
                        json_entry = (
                            entry
                            if args.with_html
                            else {k: v for k, v in entry.items() if k != "html"}
                        )
                        f.write(json.dumps(json_entry, ensure_ascii=False) + "\n")
                        count += 1
                finally:
                    if html_file:
                        html_file.close()
            print(f"wrote {count} entries -> {json_out}")
            if args.html_out:
                print(f"wrote HTML -> {args.html_out}")
            return 0
        rows = []
        html_parts = []
        for entry in entries:
            if args.html_out:
                html_parts.append(entry["html"])
            json_entry = (
                entry
                if args.with_html
                else {k: v for k, v in entry.items() if k != "html"}
            )
            rows.append(json.dumps(json_entry, ensure_ascii=False))
        if args.html_out:
            args.html_out.write_text("\n".join(html_parts), "utf-8")
            print(f"wrote HTML -> {args.html_out}")
        text = "\n".join(rows)
    else:
        if args.index is None:
            parser.error("index is required unless --word or --dump-entries is used")
        if args.entry:
            result = {
                i: dec.decode_entry(i)
                for i in range(args.index, args.index + args.count)
            }
            if args.with_html:
                for i, entry in result.items():
                    entry["html"] = dec.decode_html(i)
            text = json.dumps(result, ensure_ascii=False, indent=2)
        elif args.html:
            articles = [
                dec.decode_html(i) for i in range(args.index, args.index + args.count)
            ]
            text = "\n".join(articles)
            if args.html_out:
                args.html_out.write_text(text, "utf-8")
                print(f"wrote HTML -> {args.html_out}")
        else:
            result = {
                i: [{"style": s, "text": t} for s, t in dec.decode(i)]
                for i in range(args.index, args.index + args.count)
            }
            text = json.dumps(result, ensure_ascii=False, indent=2)

    if json_out:
        json_out.write_text(text, "utf-8")
        print(f"wrote JSON -> {json_out}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
