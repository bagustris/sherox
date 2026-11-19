"""Tests for the CTC forced-alignment backend (no real model required)."""
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sherox import AlignmentError, ConfigError, ModelNotFoundError
from sherox.alignment import base, ctc
from sherox.alignment.ctc import CtcOnnxBackend, ctc_forced_align

# Tiny vocabulary: blank=0, delimiter '|'=1, letters a..e = 2..6
VOCAB = {"<pad>": 0, "|": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6}
BLANK = 0


def _emissions(path: list[int], V: int = 7, p: float = 0.97) -> np.ndarray:
    """Log-prob matrix whose argmax follows ``path`` frame by frame."""
    T = len(path)
    probs = np.full((T, V), (1 - p) / (V - 1))
    probs[np.arange(T), path] = p
    return np.log(probs)


class TestCtcForcedAlign:
    def test_known_alignment(self):
        # frames:  blank A A blank B blank C
        lp = _emissions([0, 2, 2, 0, 3, 0, 4])
        spans = ctc_forced_align(lp, [2, 3, 4], BLANK)
        assert [(s, e) for s, e, _ in spans] == [(1, 3), (4, 5), (6, 7)]
        assert all(0.9 < score <= 1.0 for _, _, score in spans)

    def test_repeated_letters_need_blank_between(self):
        # "A A" must be separated by a blank: A blank A
        lp = _emissions([2, 0, 2])
        spans = ctc_forced_align(lp, [2, 2], BLANK)
        assert [(s, e) for s, e, _ in spans] == [(0, 1), (2, 3)]

    def test_repeated_letters_infeasible_without_blank_frame(self):
        with pytest.raises(AlignmentError, match="too short"):
            ctc_forced_align(_emissions([2, 2]), [2, 2], BLANK)

    def test_audio_too_short(self):
        with pytest.raises(AlignmentError, match="too short"):
            ctc_forced_align(_emissions([2, 3]), [2, 3, 4], BLANK)

    def test_empty_labels(self):
        assert ctc_forced_align(_emissions([0, 0]), [], BLANK) == []

    def test_spans_monotonic_and_cover_every_label(self):
        rng = np.random.default_rng(0)
        lp = np.log(rng.dirichlet(np.ones(7), size=60))
        labels = [2, 3, 3, 4, 5, 6, 2]
        spans = ctc_forced_align(lp, labels, BLANK)
        assert len(spans) == len(labels)
        prev_end = 0
        for s, e, score in spans:
            assert prev_end <= s < e
            assert 0.0 <= score <= 1.0
            prev_end = e

    def test_trellis_size_limit(self):
        with patch.object(ctc, "_MAX_TRELLIS_CELLS", 10):
            with pytest.raises(AlignmentError, match="too long"):
                ctc_forced_align(_emissions([2] * 20), [2], BLANK)


# ── Backend with a fake onnxruntime session ──────────────────────────────────

def _make_model_dir(tmp_path: Path, vocab=VOCAB, onnx=True) -> Path:
    d = tmp_path / "fake-model"
    d.mkdir()
    (d / "vocab.json").write_text(json.dumps(vocab))
    if onnx:
        (d / "model.onnx").write_bytes(b"not a real model")
    return d


class _FakeSession:
    """Returns logits that spell out a fixed frame path."""

    def __init__(self, path):
        self._logits = _emissions(path)[None]

    def get_inputs(self):
        return [SimpleNamespace(name="input_values")]

    def run(self, _outputs, feed):
        assert feed["input_values"].ndim == 2
        return [self._logits]


def _fake_ort(path, providers=("CPUExecutionProvider",)):
    ort = MagicMock()
    ort.get_available_providers.return_value = list(providers)
    ort.InferenceSession.return_value = _FakeSession(path)
    return ort


def _backend(tmp_path, path, **kw):
    ort = _fake_ort(path, kw.pop("providers", ("CPUExecutionProvider",)))
    with patch.object(ctc, "_require_onnxruntime", return_value=ort):
        return CtcOnnxBackend(model_dir=str(_make_model_dir(tmp_path)), **kw)


class TestCtcOnnxBackend:
    # frames: A B | C D E  with blanks (10 frames → stride = 1600/10/16000 = 0.01 s)
    PATH = [0, 2, 3, 0, 1, 0, 4, 5, 6, 0]

    def test_alignment_words_and_times(self, tmp_path):
        be = _backend(tmp_path, self.PATH)
        audio = np.zeros(1600, dtype=np.float32)
        out = be.align(audio, 16000, ["Ab,", "CDE."], "custom")
        assert [w.word for w in out] == ["Ab,", "CDE."]        # punctuation preserved
        assert [w.status for w in out] == ["aligned", "aligned"]
        assert (out[0].start, out[0].end) == (0.01, 0.03)
        assert (out[1].start, out[1].end) == (0.06, 0.09)
        assert 0.9 < out[0].score <= 1.0

    def test_unalignable_word_is_kept_as_unaligned(self, tmp_path):
        be = _backend(tmp_path, self.PATH)
        audio = np.zeros(1600, dtype=np.float32)
        out = be.align(audio, 16000, ["AB", "42", "CDE"], "custom")
        assert [w.index for w in out] == [0, 1, 2]
        assert out[1].status == "unaligned"
        assert out[1].start is None and out[1].end is None and out[1].score is None
        assert out[0].status == out[2].status == "aligned"

    def test_all_words_unalignable(self, tmp_path):
        be = _backend(tmp_path, self.PATH)
        out = be.align(np.zeros(1600, np.float32), 16000, ["123", "!!"], "custom")
        assert [w.status for w in out] == ["unaligned", "unaligned"]

    def test_tokenize_is_case_insensitive_and_drops_unknown(self, tmp_path):
        be = _backend(tmp_path, self.PATH)
        assert be.tokenize_word("a-b!c") == [2, 3, 4]
        assert be.tokenize_word("zzz") == []

    def test_lowercase_vocab(self, tmp_path):
        vocab = {"[PAD]": 0, "a": 1, "b": 2}
        d = _make_model_dir(tmp_path, vocab)
        with patch.object(ctc, "_require_onnxruntime", return_value=_fake_ort([0])):
            be = CtcOnnxBackend(model_dir=str(d))
        assert be.tokenize_word("AB") == [1, 2]
        assert be.delimiter_id is None

    def test_default_model_rejects_other_languages(self, tmp_path):
        be = _backend(tmp_path, self.PATH)
        be.languages = frozenset({"en"})
        with pytest.raises(ConfigError, match="not supported"):
            be.align(np.zeros(1600, np.float32), 16000, ["ab"], "ja")
        # aliases are accepted
        be.align(np.zeros(1600, np.float32), 16000, ["ab"], "English")

    def test_rejects_wrong_sample_rate_and_tiny_audio(self, tmp_path):
        be = _backend(tmp_path, self.PATH)
        with pytest.raises(AlignmentError, match="Hz"):
            be.align(np.zeros(1600, np.float32), 8000, ["ab"], "en")
        with pytest.raises(AlignmentError, match="too short"):
            be.align(np.zeros(100, np.float32), 16000, ["ab"], "en")

    def test_transcript_longer_than_audio_fails_explicitly(self, tmp_path):
        be = _backend(tmp_path, self.PATH)
        with pytest.raises(AlignmentError, match="too short"):
            be.align(np.zeros(1600, np.float32), 16000, ["abcdeabcdeabcde"], "en")

    def test_cuda_unavailable_is_a_clear_error(self, tmp_path):
        with pytest.raises(ConfigError, match="CUDAExecutionProvider"):
            _backend(tmp_path, self.PATH, device="cuda")

    def test_cuda_used_when_available(self, tmp_path):
        ort = _fake_ort(self.PATH, ("CUDAExecutionProvider", "CPUExecutionProvider"))
        with patch.object(ctc, "_require_onnxruntime", return_value=ort):
            CtcOnnxBackend(model_dir=str(_make_model_dir(tmp_path)), device="cuda")
        providers = ort.InferenceSession.call_args.kwargs["providers"]
        assert providers[0] == "CUDAExecutionProvider"

    def test_invalid_device(self, tmp_path):
        with pytest.raises(ConfigError, match="--device"):
            _backend(tmp_path, self.PATH, device="tpu")


class TestModelResolution:
    def test_missing_onnxruntime_has_install_hint(self):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(ConfigError, match=r"sherox\[align\]"):
                ctc._require_onnxruntime()

    def test_missing_model_dir(self, tmp_path):
        with patch.object(ctc, "_require_onnxruntime", return_value=_fake_ort([0])):
            with pytest.raises(ModelNotFoundError, match="not found"):
                CtcOnnxBackend(model_dir=str(tmp_path / "nope"))

    def test_missing_vocab(self, tmp_path):
        d = _make_model_dir(tmp_path)
        (d / "vocab.json").unlink()
        with patch.object(ctc, "_require_onnxruntime", return_value=_fake_ort([0])):
            with pytest.raises(ModelNotFoundError, match="vocab.json"):
                CtcOnnxBackend(model_dir=str(d))

    def test_vocab_without_blank(self, tmp_path):
        d = _make_model_dir(tmp_path, {"a": 0, "b": 1})
        with patch.object(ctc, "_require_onnxruntime", return_value=_fake_ort([0])):
            with pytest.raises(ModelNotFoundError, match="blank"):
                CtcOnnxBackend(model_dir=str(d))

    def test_missing_onnx_file(self, tmp_path):
        d = _make_model_dir(tmp_path, onnx=False)
        with patch.object(ctc, "_require_onnxruntime", return_value=_fake_ort([0])):
            with pytest.raises(ModelNotFoundError, match=r"\.onnx"):
                CtcOnnxBackend(model_dir=str(d))

    def test_default_model_is_downloaded_once(self, tmp_path):
        calls = []

        def fake_download(url, dest):
            calls.append(Path(dest).name)
            Path(dest).write_text("{}")

        with patch.object(ctc, "_download_file", fake_download):
            d, is_default = ctc.resolve_model_dir(None, tmp_path)
            assert is_default and d.is_dir()
            assert sorted(calls) == sorted(ctc._DEFAULT_MODEL_FILES)
            assert not (d.parent / (d.name + ".part")).exists()
            ctc.resolve_model_dir(None, tmp_path)           # cached now
        assert len(calls) == len(ctc._DEFAULT_MODEL_FILES)


class TestRegistry:
    def test_get_backend_unknown(self):
        with pytest.raises(ConfigError, match="Unknown alignment backend"):
            base.get_backend("nope")

    def test_ctc_registered(self):
        assert "ctc" in base.available_backends()


class _RampSession:
    """Frame k's logits column 0 equals the waveform sample at k * 320."""

    def __init__(self):
        self.calls = 0

    def get_inputs(self):
        return [SimpleNamespace(name="input_values")]

    def run(self, _outputs, feed):
        x = feed["input_values"][0]
        self.calls += 1
        frames = (x.size - 400) // 320 + 1
        out = np.zeros((1, frames, 4), dtype=np.float32)
        out[0, :, 0] = x[np.arange(frames) * 320]
        return [out]


class TestChunkedInference:
    def _be(self, tmp_path):
        be = _backend(tmp_path, [0])
        be._session = _RampSession()
        return be

    def test_chunked_matches_single_pass(self, tmp_path):
        be = self._be(tmp_path)
        n = 320 * 95 + 123                      # not a multiple of the hop
        x = np.arange(n, dtype=np.float32)
        single = be._forward(x)
        with patch.object(ctc, "_CHUNK_SAMPLES", 320 * 10), \
                patch.object(ctc, "_CONTEXT_SAMPLES", 320 * 3):
            be._session.calls = 0
            chunked = be._logits(x)
        assert be._session.calls > 5            # really windowed
        assert chunked.shape == single.shape
        np.testing.assert_array_equal(chunked, single)

    def test_short_audio_is_a_single_pass(self, tmp_path):
        be = self._be(tmp_path)
        be._logits(np.zeros(16000, dtype=np.float32))
        assert be._session.calls == 1

    def test_unexpected_frame_stride_is_an_error(self, tmp_path):
        be = self._be(tmp_path)
        be._session.run = lambda *a, **k: [np.zeros((1, 3, 4), np.float32)]
        with patch.object(ctc, "_CHUNK_SAMPLES", 320 * 10), \
                patch.object(ctc, "_CONTEXT_SAMPLES", 320 * 3):
            with pytest.raises(AlignmentError, match="320-sample"):
                be._logits(np.zeros(320 * 95, dtype=np.float32))


class TestEarlyFailures:
    def test_inference_error_becomes_alignment_error(self, tmp_path):
        be = _backend(tmp_path, [0])
        be._session.run = MagicMock(side_effect=RuntimeError("out of memory"))
        with pytest.raises(AlignmentError, match="inference failed.*out of memory"):
            be.align(np.zeros(1600, np.float32), 16000, ["ab"], "en")

    def test_oversized_trellis_rejected_before_inference(self, tmp_path):
        be = _backend(tmp_path, [0])
        be._session.run = MagicMock()
        with patch.object(ctc, "_MAX_TRELLIS_CELLS", 10):
            with pytest.raises(AlignmentError, match="too long"):
                be.align(np.zeros(16000, np.float32), 16000, ["abc"], "en")
        be._session.run.assert_not_called()

    def test_unsupported_language_rejected_before_download(self):
        with patch.object(ctc, "_require_onnxruntime", return_value=_fake_ort([0])), \
                patch.object(ctc, "resolve_model_dir") as resolve, \
                patch.object(ctc, "_download_file") as dl:
            with pytest.raises(ConfigError, match="not supported"):
                CtcOnnxBackend(language="ja")
        resolve.assert_not_called()
        dl.assert_not_called()

    def test_custom_model_skips_language_check(self, tmp_path):
        ort = _fake_ort([0])
        with patch.object(ctc, "_require_onnxruntime", return_value=ort):
            CtcOnnxBackend(model_dir=str(_make_model_dir(tmp_path)), language="ja")
