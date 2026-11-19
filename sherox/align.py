"""Word-level forced alignment — entry point.

Aligns an existing transcript to a recording and writes word start/end
timestamps (seconds) as JSON.  No ASR is run.

Usage:
    sherox.align --wav audio.wav --transcript transcript.txt \\
                 --language en --output alignment.json

    # Print JSON to stdout instead of a file:
    sherox.align --wav audio.wav --transcript transcript.txt

    # Use your own CTC ONNX model (any language) instead of the default:
    sherox.align --wav audio.wav --transcript t.txt --language fr \\
                 --model-dir models/my-ctc-model

Requires the optional ``align`` extra (``pip install 'sherox[align]'``).
The JSON schema is documented in ALIGN_MODEL.md.  Logs go to stderr so that
stdout carries only JSON when ``--output`` is omitted.
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from rich.console import Console

from . import AlignmentError, AudioError, ConfigError, SherpaError
from .alignment import AlignmentResult, available_backends, get_backend, split_words
from .audio import read_wav
from .utils import run_cli as _run_cli

_SAMPLE_RATE = 16000
_err_console = Console(stderr=True)


def _info(msg: str) -> None:
    _err_console.print(f"[bold green]\\[info][/bold green] {msg}")


def _warn(msg: str) -> None:
    _err_console.print(f"[bold yellow]\\[warn][/bold yellow] {msg}")


def _error(msg: str, exc_type: type[SherpaError] = ConfigError) -> None:
    """Raise ``exc_type``; ``_main_impl`` reports it once and sets the exit code."""
    raise exc_type(msg)


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Word-level forced alignment of a transcript to audio",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--wav", required=True, metavar="PATH", help="Audio file (mono)")
    parser.add_argument(
        "--transcript", required=True, metavar="PATH",
        help="UTF-8 text file with the transcript of the audio",
    )
    parser.add_argument(
        "--language", default="en", metavar="CODE",
        help="Transcript language code (default model supports: en)",
    )
    parser.add_argument(
        "--output", default="", metavar="PATH",
        help="Write JSON here (default: print to stdout)",
    )
    parser.add_argument(
        "--backend", default="ctc", choices=available_backends(),
        help="Alignment backend",
    )
    parser.add_argument(
        "--model-dir", default="", metavar="DIR",
        help="Backend model directory (default: auto-downloaded model)",
    )
    parser.add_argument(
        "--device", default="cpu", choices=["cpu", "cuda"],
        help="Inference device",
    )
    parser.add_argument("--threads", type=int, default=4, help="CPU threads for inference")
    return parser.parse_args(argv)


def load_audio(path: str) -> np.ndarray:
    """Load ``path`` as mono float32 at 16 kHz."""
    if not Path(path).is_file():
        _error(f"Audio file not found: {path}", AudioError)
    try:
        chunks = list(read_wav(path, target_sr=_SAMPLE_RATE, chunk_size=10.0))
    except Exception as exc:  # noqa: BLE001 - soundfile/format/mono errors
        _error(f"Could not read audio '{path}': {exc}", AudioError)
    if not chunks:
        _error(f"Audio file is empty: {path}", AudioError)
    return np.concatenate(chunks).astype(np.float32)


def load_transcript(path: str) -> str:
    p = Path(path)
    if not p.is_file():
        _error(f"Transcript file not found: {path}")
    try:
        text = p.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        _error(f"Could not read transcript '{path}': {exc}")
    text = " ".join(text.split())
    if not text:
        _error(f"Transcript is empty: {path}")
    return text


def run_alignment(args: argparse.Namespace) -> AlignmentResult:
    """Validate inputs, run the backend and package the result."""
    if args.threads <= 0:
        _error(f"--threads must be > 0, got {args.threads}")
    transcript = load_transcript(args.transcript)
    audio = load_audio(args.wav)
    words = split_words(transcript)

    backend = get_backend(
        args.backend,
        model_dir=args.model_dir or None,
        device=args.device,
        num_threads=args.threads,
        language=args.language,
    )
    _info(f"Aligning {len(words)} words ({audio.size / _SAMPLE_RATE:.1f}s) "
          f"with backend '{backend.name}' on {args.device}…")
    aligned = backend.align(audio, _SAMPLE_RATE, words, args.language)
    if len(aligned) != len(words):  # backend contract violation
        raise AlignmentError(
            f"Backend returned {len(aligned)} words for {len(words)} input words"
        )
    return AlignmentResult(
        language=args.language,
        backend=backend.name,
        model=backend.model_name,
        audio_path=str(args.wav),
        duration=audio.size / _SAMPLE_RATE,
        sample_rate=_SAMPLE_RATE,
        transcript=transcript,
        words=aligned,
    )


def _main_impl() -> None:
    args = parse_args()
    try:
        result = run_alignment(args)
    except SherpaError as exc:
        _err_console.print(f"[bold red]\\[error][/bold red] {exc}")
        raise
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(payload + "\n", encoding="utf-8")
        _info(f"Wrote {out}")
    else:
        sys.stdout.write(payload + "\n")
    if result.num_unaligned:
        missed = [w.word for w in result.words if w.status != "aligned"]
        _warn(
            f"{result.num_unaligned} of {len(result.words)} words could not be "
            f"aligned (no vocabulary characters): {', '.join(missed[:10])}"
            + (" …" if len(missed) > 10 else "")
            + ". They are kept in the output with status 'unaligned'."
        )


def main() -> None:
    _run_cli(_main_impl)


if __name__ == "__main__":
    main()
