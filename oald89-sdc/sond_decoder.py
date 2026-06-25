#!/usr/bin/env python3
"""
SOND -> WAV extractor for Paragon SlovoEd SDC dictionaries.

Input: raw SOND resource blob(s), already pulled from the .sdc and inflated if
they were stored compressed.
Output: WAV/container files in a chosen output folder.

Codec-1 (Speex) is decoded by wrapping the raw frames into Ogg-Speex and calling
ffmpeg. ffmpeg is required for Speex.

Subcommands:
  single  Extract one SOND blob.
  multi   Extract multiple manually listed SOND blobs.
  folder  Extract all matching SOND blobs from a folder.

Examples:
  python sond.py single input.sond -o out
  python sond.py multi a.sond b.sond c.sond -o out
  python sond.py folder ./sond_blobs -o out --recursive
"""

import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import sys
import wave
from wave import Wave_write

MODE_FRAME_BYTES = [
    15,
    20,
    25,
    33,
    43,
    52,
    60,
    70,
    86,
    106,
]  # dword_1004818F8
CODEC_SPEEX = 1
CODEC_PASSTHROUGH = (2, 4, 5)
SPEEX_WB_RATE = 16000  # wideband native rate for codec 1


class SondError(Exception):
    pass


def _u32(b, o):
    return struct.unpack_from("<I", b, o)[0]


def _read(path: os.PathLike | str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _write(path: os.PathLike | str, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)


def parse_header(blob: bytes) -> dict:
    if len(blob) < 32:
        raise SondError(f"blob too small ({len(blob)} bytes)")
    if _u32(blob, 0) != 32:
        raise SondError("bad marker (expected 32) -- not a decompressed SOND blob")
    codec = _u32(blob, 8)
    payload_len = _u32(blob, 12)
    rate_label = _u32(blob, 16)
    info = {"codec": codec, "rate_label": rate_label, "payload_len": payload_len}

    if codec == CODEC_SPEEX:
        hlen = 48
        info["bits_per_sample"] = _u32(blob, 32)  # ==16
        mode_flag = _u32(blob, 36)  # ==1/100 -> table, else byte@40
        if mode_flag in (1, 100):
            idx = blob[40]
            if not (1 <= idx <= len(MODE_FRAME_BYTES)):
                raise SondError(f"mode index {idx} out of range")
            frame_bytes = MODE_FRAME_BYTES[idx - 1]
        else:
            frame_bytes = blob[40]
        info.update(
            header_len=hlen,
            frame_bytes=frame_bytes,
            channels=1,
            speex_band="wideband",
            native_rate=SPEEX_WB_RATE,
        )
        if payload_len % frame_bytes:
            raise SondError(
                f"payload {payload_len} not divisible by frame {frame_bytes}"
            )
        info["frame_count"] = payload_len // frame_bytes
    elif codec in CODEC_PASSTHROUGH:
        info.update(header_len=32, channels=1, bits_per_sample=16)
    else:
        raise SondError(f"unknown codec id {codec} (dispatch handles 1/2/4/5)")

    end = info["header_len"] + payload_len
    if len(blob) < end:
        raise SondError(
            f"blob truncated: need {end} bytes, only have {len(blob)} bytes"
        )

    info["payload"] = blob[info["header_len"] : end]
    return info


def sniff_container(p: bytes) -> str:
    if p[:4] == b"RIFF" and p[8:12] == b"WAVE":
        return "wav"
    if p[:4] == b"caff":
        return "caf"
    if p[4:8] == b"ftyp":
        return "m4a"
    if p[:4] == b"OggS":
        return "ogg"
    if p[:3] == b"ID3":
        return "mp3"
    return "raw?"


def write_wav(path, pcm, rate, channels=1):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf: Wave_write = wf
        wf.setnchannels(channels)  # pylint: disable=no-member
        wf.setsampwidth(2)  # pylint: disable=no-member
        wf.setframerate(rate)  # pylint: disable=no-member
        wf.writeframes(pcm)  # pylint: disable=no-member


def extract(blob: bytes, out_base: os.PathLike | str) -> dict:
    info = parse_header(blob)
    payload = info.pop("payload")
    codec = info["codec"]
    out_base = str(out_base)

    if codec in CODEC_PASSTHROUGH:
        # 2=WAV, 4=MP3, 5=OGG. Payload after the 32-byte
        # TSoundFileHeader is the container/codec data itself.
        fmt = {2: "wav", 4: "mp3", 5: "ogg"}[codec]
        info["sound_format"] = fmt
        kind = sniff_container(payload)
        info["container_sniff"] = kind
        if fmt == "wav" and kind != "wav":
            out = out_base + ".wav"
            write_wav(out, payload, info["rate_label"])  # raw PCM16 mono -> WAV
        else:
            out = f"{out_base}.{fmt}"
            _write(out, payload)  # already a WAV/MP3/OGG file; write as-is
        info["output"] = out
    else:
        out = out_base + ".wav"
        method = _decode_speex(payload, info["frame_bytes"], info["rate_label"], out)
        info["output"] = out
        info["decoder"] = method
        with wave.open(out, "rb") as w:
            info["duration_s"] = round(w.getnframes() / w.getframerate(), 3)

    return info


# --- inlined pure-python Ogg-Speex muxer (lets a stock ffmpeg decode raw frames) ---
def _ogg_crc_table():
    t = []
    for i in range(256):
        r = i << 24
        for _ in range(8):
            r = (
                ((r << 1) ^ 0x04C11DB7) & 0xFFFFFFFF
                if (r & 0x80000000)
                else (r << 1) & 0xFFFFFFFF
            )
        t.append(r)
    return t


_OGG_CRC = _ogg_crc_table()


def _ogg_crc(b: bytes) -> int:
    c = 0
    for x in b:
        c = ((c << 8) & 0xFFFFFFFF) ^ _OGG_CRC[((c >> 24) & 0xFF) ^ x]
    return c


def _ogg_page(serial, seq, hdr_type, granule, packets):
    seg = bytearray()
    body = bytearray()
    for p in packets:
        n = len(p)
        while n >= 255:
            seg.append(255)
            n -= 255
        seg.append(n)
        body += p
    page = bytearray(b"OggS") + bytes([0, hdr_type])
    page += (
        struct.pack("<q", granule) + struct.pack("<I", serial) + struct.pack("<I", seq)
    )
    page += b"\x00\x00\x00\x00" + bytes([len(seg)]) + seg + body
    page[22:26] = struct.pack("<I", _ogg_crc(page))
    return bytes(page)


def ogg_speex_wrap(frames, rate=16000, mode=1, frame_size=320, channels=1, serial=1):
    """Wrap raw header-less Speex frames into Ogg-Speex (.spx) bytes."""
    hdr = b"Speex   " + b"1.2.0".ljust(20, b"\0")
    hdr += struct.pack(
        "<13i", 1, 80, rate, mode, 4, channels, -1, frame_size, 0, 1, 0, 0, 0
    )
    comment = struct.pack("<I", 7) + b"spxwrap" + struct.pack("<I", 0)
    out = bytearray()
    out += _ogg_page(serial, 0, 0x02, 0, [hdr])  # BOS: identification header
    out += _ogg_page(serial, 1, 0x00, 0, [comment])  # comment header
    seq, i, per = 2, 0, 200
    while i < len(frames):
        chunk = frames[i : i + per]
        i += per
        last = i >= len(frames)
        out += _ogg_page(serial, seq, 0x04 if last else 0x00, i * frame_size, chunk)
        seq += 1
    return bytes(out)


def _decode_speex(payload, frame_bytes, rate, out_wav):
    """Decode raw Speex frames -> WAV using ffmpeg only."""
    if shutil.which("ffmpeg") is None:
        raise SondError("ffmpeg not found in PATH; Speex SOND decoding requires ffmpeg")

    frames = [payload[i : i + frame_bytes] for i in range(0, len(payload), frame_bytes)]
    spx = out_wav + ".spx"
    _write(spx, ogg_speex_wrap(frames, rate=rate))
    try:
        subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", spx, out_wav],
            check=True,
        )
    finally:
        try:
            os.remove(spx)
        except FileNotFoundError:
            pass

    return "ffmpeg (ogg-speex)"


def output_base_for(input_path: Path, output_dir: Path, used: set[str]) -> Path:
    """Return a collision-safe output base path inside output_dir."""
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = input_path.stem
    candidate = output_dir / stem
    index = 2
    while str(candidate) in used or any(
        (candidate.with_suffix(ext)).exists()
        for ext in (".wav", ".caf", ".m4a", ".ogg", ".mp3")
    ):
        candidate = output_dir / f"{stem}_{index}"
        index += 1

    used.add(str(candidate))
    return candidate


def extract_path(input_path: Path, output_dir: Path, used: set[str]) -> dict:
    out_base = output_base_for(input_path, output_dir, used)
    info = extract(_read(input_path), out_base)
    info["input"] = str(input_path)
    return info


def print_info(info: dict) -> None:
    print(f"\n== {info.get('input', '<memory>')} ==")
    for k, v in info.items():
        if k != "input":
            print(f"{k}: {v}")


def run_many(paths: list[Path], output_dir: Path, keep_going: bool) -> int:
    used: set[str] = set()
    ok = 0
    failed = 0

    for path in paths:
        try:
            info = extract_path(path, output_dir, used)
            print_info(info)
            ok += 1
        except (OSError, SondError, subprocess.CalledProcessError, wave.Error) as exc:
            failed += 1
            print(f"\n!! {path} failed: {exc}", file=sys.stderr)
            if not keep_going:
                break

    print(f"\nDone: {ok} extracted, {failed} failed")
    return 1 if failed else 0


def collect_folder(folder: Path, pattern: str, recursive: bool) -> list[Path]:
    iterator = folder.rglob(pattern) if recursive else folder.glob(pattern)
    return sorted(p for p in iterator if p.is_file())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="sond.py",
        description="Extract raw SOND resource blobs to WAV/container files.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    single = sub.add_parser("single", help="extract one SOND blob")
    single.add_argument("input", type=Path)
    single.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    single.add_argument(
        "--keep-going",
        action="store_true",
        help="accepted for consistency; no effect for a single file",
    )

    multi = sub.add_parser("multi", help="extract manually listed SOND blobs")
    multi.add_argument("inputs", type=Path, nargs="+")
    multi.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    multi.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="continue after a failed file (default: true)",
    )

    folder = sub.add_parser("folder", help="extract matching files from a folder")
    folder.add_argument("folder", type=Path)
    folder.add_argument("-o", "--output-dir", type=Path, default=Path("."))
    folder.add_argument(
        "-p",
        "--pattern",
        default="*.sond",
        help='glob pattern to match (default: "*.sond")',
    )
    folder.add_argument("-r", "--recursive", action="store_true")
    folder.add_argument(
        "--keep-going",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="continue after a failed file (default: true)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "single":
        return run_many([args.input], args.output_dir, keep_going=False)

    if args.command == "multi":
        return run_many(args.inputs, args.output_dir, keep_going=args.keep_going)

    if args.command == "folder":
        if not args.folder.is_dir():
            parser.error(f"folder does not exist or is not a directory: {args.folder}")
        paths = collect_folder(args.folder, args.pattern, args.recursive)
        if not paths:
            parser.error(
                f"no files matched {args.pattern!r} in {args.folder}"
                + (" recursively" if args.recursive else "")
            )
        return run_many(paths, args.output_dir, keep_going=args.keep_going)

    parser.error("unknown command")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
