"""CTC forced-alignment backend running on ``onnxruntime`` (no PyTorch).

Works with any character-level CTC acoustic model exported to ONNX that takes
a raw 16 kHz waveform ``(batch, samples)`` and returns logits
``(batch, frames, vocab)`` — e.g. wav2vec2 / HuBERT / MMS CTC checkpoints.  The
model directory must contain an ``.onnx`` file and a ``vocab.json``
(``{token: id}``); ``preprocessor_config.json`` is optional (used for
``do_normalize``).

The default model is ``wav2vec2-base-960h`` (English), auto-downloaded to
``models/wav2vec2-base-960h-onnx``.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
import unicodedata
from pathlib import Path
from typing import Optional

import numpy as np

from .. import AlignmentError, ConfigError, ModelNotFoundError
from .. import model_cache
from ..utils import download_file as _download_file
from .base import (
    STATUS_ALIGNED,
    STATUS_UNALIGNED,
    AlignerBackend,
    WordAlignment,
)

DEFAULT_MODEL_NAME = "wav2vec2-base-960h-onnx"
_HF_BASE = "https://huggingface.co/Xenova/wav2vec2-base-960h/resolve/main"
_DEFAULT_MODEL_FILES = {
    "model.onnx": f"{_HF_BASE}/onnx/model_quantized.onnx",
    "vocab.json": f"{_HF_BASE}/vocab.json",
    "preprocessor_config.json": f"{_HF_BASE}/preprocessor_config.json",
}
DEFAULT_LANGUAGES = frozenset({"en"})
_LANGUAGE_ALIASES = {"eng": "en", "english": "en", "en-us": "en", "en-gb": "en"}

_BLANK_CANDIDATES = ("<pad>", "[PAD]", "<blank>", "<blk>", "[blank]")
_DELIMITER_CANDIDATES = ("|",)
_SPECIAL_RE = re.compile(r"^(<.*>|\[.*\])$")

_SAMPLE_RATE = 16000
_MIN_SAMPLES = 400            # wav2vec2 conv receptive field
_HOP = 320                    # wav2vec2 frame stride in samples (20 ms @ 16 kHz)
_CHUNK_SAMPLES = 1000 * _HOP  # 20 s of audio per forward pass on long inputs
_CONTEXT_SAMPLES = 100 * _HOP  # 2 s of extra context on each side of a chunk
_MAX_TRELLIS_CELLS = 400_000_000


def normalize_language(language: str) -> str:
    lang = language.strip().lower().replace("_", "-")
    return _LANGUAGE_ALIASES.get(lang, lang)


def _check_default_language(language: str) -> None:
    if normalize_language(language) not in DEFAULT_LANGUAGES:
        raise ConfigError(
            f"Language '{language}' is not supported by the default alignment "
            f"model (supported: {', '.join(sorted(DEFAULT_LANGUAGES))}). "
            "Pass --model-dir with a CTC model for that language."
        )


# ── Viterbi ──────────────────────────────────────────────────────────────────

def check_trellis_size(frames: int, states: int) -> None:
    """Fail early, with advice, when the Viterbi trellis would not fit."""
    if frames * states > _MAX_TRELLIS_CELLS:
        raise AlignmentError(
            f"Input too long to align in one pass ({frames} frames x {states} "
            "states). Split the audio and transcript into shorter pieces "
            "(e.g. with sherox.segment) and align each separately."
        )


def ctc_forced_align(
    log_probs: np.ndarray, labels: list[int], blank: int
) -> list[tuple[int, int, float]]:
    """Viterbi-align ``labels`` to CTC ``log_probs`` of shape ``(T, V)``.

    Uses the standard blank-interleaved CTC topology (2L+1 states, with the
    skip transition forbidden between identical consecutive labels), so
    repeated letters such as "LL" are handled correctly.

    Returns one ``(first_frame, last_frame_exclusive, mean_posterior)`` per
    label, in order.  Raises :class:`AlignmentError` if no valid alignment
    exists (audio too short for the label sequence) or the trellis would be
    too large to hold in memory.
    """
    T = log_probs.shape[0]
    L = len(labels)
    if L == 0:
        return []
    repeats = sum(1 for a, b in zip(labels, labels[1:]) if a == b)
    if T < L + repeats:
        raise AlignmentError(
            f"Audio is too short for the transcript: {T} model frames available "
            f"but at least {L + repeats} are needed. Check that the transcript "
            "matches the audio."
        )
    S = 2 * L + 1
    check_trellis_size(T, S)

    ext = np.full(S, blank, dtype=np.int64)
    ext[1::2] = labels

    skip_ok = np.zeros(S, dtype=bool)
    skip_ok[3::2] = ext[3::2] != ext[1:-2:2]

    neg = np.float32(-np.inf)
    dp = np.full(S, neg, dtype=np.float32)
    e0 = log_probs[0, ext]
    dp[0] = e0[0]
    dp[1] = e0[1]
    back = np.zeros((T, S), dtype=np.int8)                # 0 stay, 1 +1, 2 +2
    arange_s = np.arange(S)
    cand = np.full((3, S), neg, dtype=np.float32)
    for t in range(1, T):
        cand[0] = dp
        cand[1, 0] = neg
        cand[1, 1:] = dp[:-1]
        cand[2, :2] = neg
        cand[2, 2:] = np.where(skip_ok[2:], dp[:-2], neg)
        choice = cand.argmax(axis=0)
        back[t] = choice
        dp = cand[choice, arange_s] + log_probs[t, ext].astype(np.float32)

    s = S - 1 if dp[S - 1] >= dp[S - 2] else S - 2
    if not np.isfinite(dp[s]):
        raise AlignmentError("No valid CTC alignment path found.")
    path = np.empty(T, dtype=np.int64)
    for t in range(T - 1, -1, -1):
        path[t] = s
        s -= int(back[t, s])

    first = [-1] * L
    last = [-1] * L
    post_sum = [0.0] * L
    count = [0] * L
    for t in range(T):
        st = int(path[t])
        if st % 2 == 1:
            j = (st - 1) // 2
            if first[j] < 0:
                first[j] = t
            last[j] = t
            post_sum[j] += float(np.exp(log_probs[t, ext[st]]))
            count[j] += 1
    return [(first[j], last[j] + 1, post_sum[j] / count[j]) for j in range(L)]


def _frame_to_seconds(n_samples: int, n_frames: int, sample_rate: int):
    """Return a function mapping a frame index to seconds from audio start.

    For the standard wav2vec2 geometry (``(N - 400) // 320 + 1`` frames) frame
    ``k`` covers samples ``[k*320 + 40, (k+1)*320 + 40)``: a 320-sample hop,
    offset by half the difference between the 400-sample receptive field and
    the hop.  Other models fall back to spreading frames evenly over the audio.
    """
    if n_frames == (n_samples - _MIN_SAMPLES) // _HOP + 1:
        offset = (_MIN_SAMPLES - _HOP) // 2
        return lambda k: (k * _HOP + offset) / sample_rate
    stride = n_samples / n_frames / sample_rate
    return lambda k: k * stride


# ── Model resolution ─────────────────────────────────────────────────────────

def _download_default_model(target: Path) -> None:
    """Download the default model into ``target`` atomically.

    Files go into a process-unique temp directory next to ``target`` that is
    renamed into place only once every file has downloaded completely, so a
    failed, interrupted or concurrent run can never leave (or clobber) a
    half-written model directory.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f".{target.name}.", suffix=".part", dir=target.parent))
    try:
        for fname, url in _DEFAULT_MODEL_FILES.items():
            # Progress goes to stderr: stdout may carry the JSON result.
            _download_file(url, tmp / fname, to_stderr=True)
        try:
            tmp.rename(target)
        except OSError:
            if not target.is_dir():      # not a lost race with another process
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def default_model_dir() -> Path:
    """User-writable location of the default model (shared sherox cache)."""
    return model_cache.cache_root() / DEFAULT_MODEL_NAME


def resolve_model_dir(model_dir: Optional[str]) -> tuple[Path, bool]:
    """Return ``(directory, is_default_model)``, downloading the default if needed.

    The default model lives in the shared cache (``SHEROX_CACHE_DIR`` or
    ``~/.cache/sherox/models``) rather than under the installed package, so it
    works from read-only installs.
    """
    if model_dir:
        p = Path(model_dir)
        if not p.is_dir():
            raise ModelNotFoundError(f"Alignment model directory not found: {p}")
        return p, False
    target = default_model_dir()
    if not all((target / f).is_file() for f in _DEFAULT_MODEL_FILES):
        if target.exists():          # incomplete leftovers from an older run
            shutil.rmtree(target, ignore_errors=True)
        _download_default_model(target)
    return target, True


def _find_onnx(model_dir: Path) -> Path:
    for name in ("model.onnx", "model_quantized.onnx", "model.int8.onnx"):
        if (model_dir / name).is_file():
            return model_dir / name
    onnx_files = sorted(model_dir.glob("*.onnx"))
    if len(onnx_files) == 1:
        return onnx_files[0]
    raise ModelNotFoundError(
        f"Expected exactly one .onnx file (or model.onnx) in {model_dir}, "
        f"found {len(onnx_files)}."
    )


def _require_onnxruntime():
    try:
        import onnxruntime  # noqa: PLC0415
    except ImportError as exc:
        raise ConfigError(
            "onnxruntime is required for the 'ctc' alignment backend. "
            "Install it with: pip install 'sherox[align]'"
        ) from exc
    return onnxruntime


# ── Backend ──────────────────────────────────────────────────────────────────

class CtcOnnxBackend(AlignerBackend):
    """Character-level CTC forced alignment with an ONNX acoustic model."""

    name = "ctc"

    def __init__(
        self,
        model_dir: Optional[str] = None,
        device: str = "cpu",
        num_threads: int = 4,
        language: Optional[str] = None,
    ) -> None:
        if device not in ("cpu", "cuda"):
            raise ConfigError(f"--device must be 'cpu' or 'cuda', got '{device}'")
        ort = _require_onnxruntime()
        # Reject an unsupported language *before* downloading the default model.
        if not model_dir and language is not None:
            _check_default_language(language)
        self.model_dir, self.is_default_model = resolve_model_dir(model_dir)
        self.model_name = (
            "wav2vec2-base-960h" if self.is_default_model else self.model_dir.name
        )
        self.languages: Optional[frozenset[str]] = (
            DEFAULT_LANGUAGES if self.is_default_model else None
        )
        self._load_vocab()
        self._load_preprocessor()

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            if "CUDAExecutionProvider" not in ort.get_available_providers():
                raise ConfigError(
                    "--device cuda requested but onnxruntime has no "
                    "CUDAExecutionProvider. Install onnxruntime-gpu or use --device cpu."
                )
            providers.insert(0, "CUDAExecutionProvider")
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = num_threads
        try:
            self._session = ort.InferenceSession(
                str(_find_onnx(self.model_dir)), sess_options=opts, providers=providers
            )
        except Exception as exc:  # noqa: BLE001
            raise ModelNotFoundError(f"Could not load alignment model: {exc}") from exc
        self._input_name = self._session.get_inputs()[0].name

    # -- setup helpers -------------------------------------------------------

    def _load_vocab(self) -> None:
        vocab_path = self.model_dir / "vocab.json"
        if not vocab_path.is_file():
            raise ModelNotFoundError(f"vocab.json not found in {self.model_dir}")
        try:
            vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
            vocab = {str(k): int(v) for k, v in vocab.items()}
        except (ValueError, TypeError, AttributeError) as exc:
            raise ModelNotFoundError(f"Invalid vocab.json: {exc}") from exc
        blank = next((vocab[t] for t in _BLANK_CANDIDATES if t in vocab), None)
        if blank is None:
            raise ModelNotFoundError(
                f"vocab.json has no CTC blank token (looked for {_BLANK_CANDIDATES})"
            )
        self.blank_id = blank
        self.delimiter_id = next(
            (vocab[t] for t in _DELIMITER_CANDIDATES if t in vocab), None
        )
        self._char_ids = {
            tok: i
            for tok, i in vocab.items()
            if len(tok) == 1 and tok not in _DELIMITER_CANDIDATES
        }

    def _load_preprocessor(self) -> None:
        self.do_normalize = True
        cfg = self.model_dir / "preprocessor_config.json"
        if cfg.is_file():
            try:
                self.do_normalize = bool(
                    json.loads(cfg.read_text(encoding="utf-8")).get("do_normalize", True)
                )
            except ValueError:
                pass

    # -- text → labels -------------------------------------------------------

    def tokenize_word(self, word: str) -> list[int]:
        """Map a word to vocabulary ids, dropping chars the model can't emit.

        Punctuation and out-of-vocabulary characters are skipped (the original
        word, punctuation included, is still reported in the output).
        """
        ids: list[int] = []
        for ch in unicodedata.normalize("NFKC", word):
            for variant in (ch, ch.upper(), ch.lower()):
                tok = self._char_ids.get(variant)
                if tok is not None:
                    ids.append(tok)
                    break
        return ids

    # -- inference -----------------------------------------------------------

    def _forward(self, x: np.ndarray) -> np.ndarray:
        try:
            return self._session.run(None, {self._input_name: x[None, :]})[0][0]
        except Exception as exc:  # noqa: BLE001 - onnxruntime raises many types
            raise AlignmentError(
                f"Alignment model inference failed: {exc}. If the audio is very "
                "long, split it into shorter pieces; if you passed --model-dir, "
                "check that the model maps a (batch, samples) waveform to "
                "(batch, frames, vocab) logits."
            ) from exc

    def _logits(self, x: np.ndarray) -> np.ndarray:
        """Frame logits for ``x``; long inputs run in overlapping 20 s windows.

        wav2vec2 self-attention is quadratic in length (a 5-minute file needs
        tens of GB in one pass), so long audio is processed in fixed windows
        with 2 s of context on each side, keeping only each window's core
        frames.  Windows start on multiples of the 320-sample frame stride, so
        window frames map exactly onto global frame indices.
        """
        n = x.size
        if n <= _CHUNK_SAMPLES + 2 * _CONTEXT_SAMPLES:
            return self._forward(x)
        total = (n - _MIN_SAMPLES) // _HOP + 1
        out = None
        for core_start in range(0, n, _CHUNK_SAMPLES):
            core_end = min(core_start + _CHUNK_SAMPLES, n)
            win_start = max(0, core_start - _CONTEXT_SAMPLES)
            win_end = min(n, core_end + _CONTEXT_SAMPLES)
            win = self._forward(x[win_start:win_end])
            expected = (win_end - win_start - _MIN_SAMPLES) // _HOP + 1
            if win.shape[0] != expected:
                raise AlignmentError(
                    "Chunked inference assumes a 20 ms (320-sample) model frame "
                    f"stride, but the model produced {win.shape[0]} frames for "
                    f"{expected} expected. Align shorter audio with this model."
                )
            if out is None:
                out = np.empty((total, win.shape[1]), dtype=win.dtype)
            lo = core_start // _HOP
            hi = total if core_end == n else core_end // _HOP
            off = win_start // _HOP
            out[lo:hi] = win[lo - off:hi - off]
        return out

    def _log_probs(self, audio: np.ndarray) -> np.ndarray:
        x = audio.astype(np.float32)
        if self.do_normalize:
            x = (x - x.mean()) / np.sqrt(x.var() + 1e-7)
        logits = self._logits(x).astype(np.float64)
        logits -= logits.max(axis=-1, keepdims=True)
        return logits - np.log(np.exp(logits).sum(axis=-1, keepdims=True))

    def align(
        self,
        audio: np.ndarray,
        sample_rate: int,
        words: list[str],
        language: str,
    ) -> list[WordAlignment]:
        if self.languages is not None and normalize_language(language) not in self.languages:
            _check_default_language(language)
        if sample_rate != _SAMPLE_RATE:
            raise AlignmentError(f"Expected {_SAMPLE_RATE} Hz audio, got {sample_rate} Hz")
        if audio.size < _MIN_SAMPLES:
            raise AlignmentError("Audio is too short to align (< 25 ms).")

        # Build the label sequence, remembering which word each label belongs to.
        labels: list[int] = []
        owner: list[int] = []       # word index per label, -1 for delimiters
        word_label_range: dict[int, tuple[int, int]] = {}
        for i, w in enumerate(words):
            ids = self.tokenize_word(w)
            if not ids:
                continue
            if labels and self.delimiter_id is not None:
                labels.append(self.delimiter_id)
                owner.append(-1)
            word_label_range[i] = (len(labels), len(labels) + len(ids))
            labels.extend(ids)
            owner.extend([i] * len(ids))

        results = [
            WordAlignment(index=i, word=w, status=STATUS_UNALIGNED)
            for i, w in enumerate(words)
        ]
        if not labels:
            return results      # nothing alignable: every word reported unaligned

        # Reject inputs whose trellis cannot fit *before* spending time on inference.
        check_trellis_size((audio.size - _MIN_SAMPLES) // _HOP + 1, 2 * len(labels) + 1)
        log_probs = self._log_probs(audio)
        spans = ctc_forced_align(log_probs, labels, self.blank_id)
        to_seconds = _frame_to_seconds(audio.size, log_probs.shape[0], sample_rate)
        duration = audio.size / sample_rate

        for i, (lo, hi) in word_label_range.items():
            word_spans = [spans[j] for j in range(lo, hi)]
            frames = sum(e - s for s, e, _ in word_spans)
            r = results[i]
            r.start = round(min(to_seconds(word_spans[0][0]), duration), 3)
            r.end = round(min(to_seconds(word_spans[-1][1]), duration), 3)
            r.score = round(
                sum(p * (e - s) for s, e, p in word_spans) / frames, 4
            )
            r.status = STATUS_ALIGNED
        return results
