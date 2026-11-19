"""Backend interface and result types for forced alignment.

A backend turns ``(audio, words)`` into one :class:`WordAlignment` per input
word.  The CLI (``sherox.align``) and the JSON output schema only depend on
this module, so engines can be added or replaced without touching either.

To add a backend, subclass :class:`AlignerBackend`, implement ``align`` and
register ``"<name>": "<module>:<Class>"`` in :data:`_BACKENDS`.  Backend
modules are imported lazily, so heavyweight dependencies stay optional.
"""
from __future__ import annotations

import importlib
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Optional

import numpy as np

from .. import ConfigError

SCHEMA_VERSION = 1

STATUS_ALIGNED = "aligned"
STATUS_UNALIGNED = "unaligned"


@dataclass
class WordAlignment:
    """Alignment of one transcript word.

    ``word`` is the original whitespace-delimited token, including any
    punctuation (``"Hello,"``).  ``start``/``end`` are seconds from the start
    of the audio, or ``None`` when ``status == "unaligned"``.  ``score`` is a
    backend-specific confidence in [0, 1] (``None`` if unavailable).
    """

    index: int
    word: str
    start: Optional[float] = None
    end: Optional[float] = None
    score: Optional[float] = None
    status: str = STATUS_ALIGNED

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AlignmentResult:
    """Complete alignment output; see ``ALIGN_MODEL.md`` for the JSON schema."""

    language: str
    backend: str
    model: str
    audio_path: str
    duration: float
    sample_rate: int
    transcript: str
    words: list[WordAlignment] = field(default_factory=list)

    @property
    def num_unaligned(self) -> int:
        return sum(1 for w in self.words if w.status != STATUS_ALIGNED)

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "language": self.language,
            "backend": self.backend,
            "model": self.model,
            "status": "ok" if self.num_unaligned == 0 else "partial",
            "audio": {
                "path": self.audio_path,
                "duration": round(self.duration, 3),
                "sample_rate": self.sample_rate,
            },
            "transcript": self.transcript,
            "num_words": len(self.words),
            "num_unaligned": self.num_unaligned,
            "words": [w.to_dict() for w in self.words],
        }


def split_words(transcript: str) -> list[str]:
    """Split a transcript into whitespace-delimited words (punctuation kept)."""
    return transcript.split()


class AlignerBackend(ABC):
    """Interface every alignment engine implements."""

    #: Registry name used by ``--backend``.
    name: str = ""
    #: Identifier of the underlying model, recorded in the output.
    model_name: str = ""

    @abstractmethod
    def align(
        self,
        audio: np.ndarray,
        sample_rate: int,
        words: list[str],
        language: str,
    ) -> list[WordAlignment]:
        """Align ``words`` to ``audio`` (mono float32 in [-1, 1]).

        Must return exactly ``len(words)`` entries in input order.  Words the
        engine cannot place must be returned with ``status="unaligned"`` and
        ``None`` timestamps rather than being dropped.  Raise
        :class:`sherox.AlignmentError` if the alignment as a whole fails.
        """


_BACKENDS: dict[str, str] = {
    "ctc": "sherox.alignment.ctc:CtcOnnxBackend",
}


def available_backends() -> list[str]:
    return sorted(_BACKENDS)


def get_backend(name: str, **kwargs) -> AlignerBackend:
    """Instantiate backend ``name`` (imported lazily) with ``kwargs``."""
    try:
        target = _BACKENDS[name]
    except KeyError:
        raise ConfigError(
            f"Unknown alignment backend '{name}'. "
            f"Available: {', '.join(available_backends())}"
        ) from None
    module_name, cls_name = target.split(":")
    cls = getattr(importlib.import_module(module_name), cls_name)
    return cls(**kwargs)
