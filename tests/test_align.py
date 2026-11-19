"""Tests for the sherox.align CLI."""
import json
import wave
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

import sherox.align as align_module
from sherox import AlignmentError, ConfigError
from sherox.alignment import base
from sherox.alignment.base import AlignerBackend, WordAlignment

DATA = Path(__file__).parent / "data"


def _write_wav(path: Path, seconds=1.0, sr=16000, channels=1):
    n = int(seconds * sr)
    pcm = (np.sin(np.arange(n) * 0.05) * 3000).astype("<i2")
    if channels == 2:
        pcm = np.repeat(pcm, 2)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


class FakeBackend(AlignerBackend):
    name = "fake"
    model_name = "fake-model"

    def __init__(self, **kwargs):
        FakeBackend.kwargs = kwargs

    def align(self, audio, sample_rate, words, language):
        FakeBackend.seen = (audio.size, sample_rate, list(words), language)
        out = []
        for i, w in enumerate(words):
            if any(c.isdigit() for c in w):
                out.append(WordAlignment(i, w, status="unaligned"))
            else:
                out.append(WordAlignment(i, w, i * 0.1, i * 0.1 + 0.05, 0.9))
        return out


@pytest.fixture
def files(tmp_path):
    wav = tmp_path / "a.wav"
    _write_wav(wav)
    txt = tmp_path / "t.txt"
    txt.write_text("Hello,   world\nagain 42\n", encoding="utf-8")
    return wav, txt


@pytest.fixture
def fake_backend():
    with patch.object(align_module, "get_backend", lambda name, **kw: FakeBackend(**kw)):
        yield FakeBackend


def _run(argv):
    with patch("sys.argv", ["sherox.align", *argv]):
        align_module.main()


class TestParseArgs:
    def test_required(self):
        with pytest.raises(SystemExit):
            align_module.parse_args(["--wav", "a.wav"])
        with pytest.raises(SystemExit):
            align_module.parse_args(["--transcript", "t.txt"])

    def test_defaults(self):
        a = align_module.parse_args(["--wav", "a.wav", "--transcript", "t.txt"])
        assert (a.language, a.output, a.backend, a.device, a.threads) == (
            "en", "", "ctc", "cpu", 4)
        assert a.model_dir == ""

    def test_invalid_backend_and_device(self):
        base_args = ["--wav", "a.wav", "--transcript", "t.txt"]
        with pytest.raises(SystemExit):
            align_module.parse_args(base_args + ["--backend", "nope"])
        with pytest.raises(SystemExit):
            align_module.parse_args(base_args + ["--device", "tpu"])


class TestRun:
    def test_writes_json_file(self, files, tmp_path, fake_backend):
        wav, txt = files
        out = tmp_path / "sub" / "out.json"
        _run(["--wav", str(wav), "--transcript", str(txt), "--output", str(out),
              "--language", "en", "--model-dir", "m", "--threads", "2"])
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["schema_version"] == 1
        assert set(data) == {"schema_version", "language", "backend", "model", "status",
                             "audio", "transcript", "num_words", "num_unaligned", "words"}
        assert data["language"] == "en" and data["backend"] == "fake"
        assert data["audio"]["duration"] == 1.0 and data["audio"]["sample_rate"] == 16000
        assert data["transcript"] == "Hello, world again 42"   # whitespace normalised
        assert [w["word"] for w in data["words"]] == ["Hello,", "world", "again", "42"]
        assert set(data["words"][0]) == {"index", "word", "start", "end", "score", "status"}
        assert fake_backend.kwargs["model_dir"] == "m"
        assert fake_backend.kwargs["num_threads"] == 2

    def test_unaligned_words_are_reported_not_dropped(self, files, tmp_path, fake_backend):
        wav, txt = files
        out = tmp_path / "o.json"
        _run(["--wav", str(wav), "--transcript", str(txt), "--output", str(out)])
        data = json.loads(out.read_text())
        assert data["status"] == "partial" and data["num_unaligned"] == 1
        last = data["words"][-1]
        assert last["word"] == "42" and last["status"] == "unaligned"
        assert last["start"] is None and last["end"] is None and last["score"] is None

    def test_output_oserror_is_a_clean_error(self, files, tmp_path, fake_backend, capsys):
        blocker = tmp_path / "file"
        blocker.write_text("x")
        target = blocker / "out.json"
        with pytest.raises(SystemExit) as e:
            _run(["--wav", str(files[0]), "--transcript", str(files[1]),
                  "--output", str(target)])                    # parent is a file
        assert e.value.code == 1
        err = capsys.readouterr().err.replace("\n", "")   # rich wraps long lines
        assert "[error]" in err and "Could not write output" in err
        assert "out.json" in err

    def test_failed_write_keeps_existing_output(self, files, tmp_path, fake_backend):
        out = tmp_path / "o.json"
        out.write_text("previous")
        with patch("os.replace", side_effect=OSError("disk full")):
            with pytest.raises(SystemExit) as e:
                _run(["--wav", str(files[0]), "--transcript", str(files[1]),
                      "--output", str(out)])
        assert e.value.code == 1
        assert out.read_text() == "previous"
        assert [p.name for p in tmp_path.iterdir() if p.name.startswith(".o.json")] == []

    def test_stdout_is_pure_json(self, files, fake_backend, capsys):
        wav, txt = files
        _run(["--wav", str(wav), "--transcript", str(txt)])
        captured = capsys.readouterr()
        assert json.loads(captured.out)["num_words"] == 4

    def test_audio_resampled_to_16k(self, files, tmp_path, fake_backend):
        wav = tmp_path / "8k.wav"
        _write_wav(wav, seconds=1.0, sr=8000)
        _run(["--wav", str(wav), "--transcript", str(files[1]),
              "--output", str(tmp_path / "o.json")])
        assert fake_backend.seen[:2] == (16000, 16000)

    def test_missing_audio_exits_1(self, files, fake_backend, tmp_path):
        with pytest.raises(SystemExit) as e:
            _run(["--wav", str(tmp_path / "no.wav"), "--transcript", str(files[1])])
        assert e.value.code == 1

    def test_stereo_audio_exits_1(self, files, tmp_path, fake_backend):
        st = tmp_path / "st.wav"
        _write_wav(st, channels=2)
        with pytest.raises(SystemExit) as e:
            _run(["--wav", str(st), "--transcript", str(files[1])])
        assert e.value.code == 1

    def test_not_audio_exits_1(self, files, tmp_path, fake_backend):
        bad = tmp_path / "bad.wav"
        bad.write_text("nope")
        with pytest.raises(SystemExit) as e:
            _run(["--wav", str(bad), "--transcript", str(files[1])])
        assert e.value.code == 1

    def test_missing_and_empty_transcript_exit_1(self, files, tmp_path, fake_backend):
        empty = tmp_path / "empty.txt"
        empty.write_text("  \n")
        for t in (tmp_path / "missing.txt", empty):
            with pytest.raises(SystemExit) as e:
                _run(["--wav", str(files[0]), "--transcript", str(t)])
            assert e.value.code == 1

    def test_invalid_threads(self, files, fake_backend):
        with pytest.raises(SystemExit) as e:
            _run(["--wav", str(files[0]), "--transcript", str(files[1]), "--threads", "0"])
        assert e.value.code == 1

    def test_backend_failure_exits_1_and_writes_nothing(self, files, tmp_path):
        class Boom(FakeBackend):
            def align(self, *a, **k):
                raise AlignmentError("cannot align")

        out = tmp_path / "o.json"
        with patch.object(align_module, "get_backend", lambda n, **kw: Boom(**kw)):
            with pytest.raises(SystemExit) as e:
                _run(["--wav", str(files[0]), "--transcript", str(files[1]),
                      "--output", str(out)])
        assert e.value.code == 1
        assert not out.exists()

    def test_backend_word_count_mismatch_is_an_error(self, files, tmp_path):
        class Short(FakeBackend):
            def align(self, audio, sr, words, language):
                return []

        with patch.object(align_module, "get_backend", lambda n, **kw: Short(**kw)):
            with pytest.raises(SystemExit):
                _run(["--wav", str(files[0]), "--transcript", str(files[1])])

    def test_missing_onnxruntime_exits_1(self, files):
        with patch.dict("sys.modules", {"onnxruntime": None}):
            with pytest.raises(SystemExit) as e:
                _run(["--wav", str(files[0]), "--transcript", str(files[1])])
        assert e.value.code == 1


class TestResultSchema:
    def test_ok_status_when_all_aligned(self):
        r = base.AlignmentResult("en", "fake", "m", "a.wav", 1.0, 16000, "hi",
                                 [WordAlignment(0, "hi", 0.0, 0.5, 0.9)])
        d = r.to_dict()
        assert d["status"] == "ok" and d["num_unaligned"] == 0


class TestCliWiring:
    def test_subcommand_listed(self):
        from sherox.main import _SUBCOMMANDS
        assert "align" in _SUBCOMMANDS

    def test_entry_point_declared(self):
        text = (Path(__file__).parent.parent / "pyproject.toml").read_text()
        assert '"sherox.align"' in text and "sherox.align:main" in text


# ── Integration: real model, skipped when it is not available ────────────────

def _real_model_dir():
    import os
    cand = os.environ.get("SHEROX_ALIGN_TEST_MODEL") or str(
        Path(__file__).parent.parent / "models" / "wav2vec2-base-960h-onnx")
    return cand if Path(cand).is_dir() else None


@pytest.mark.skipif(_real_model_dir() is None, reason="alignment model not downloaded")
def test_real_model_on_sample_recording(tmp_path):
    pytest.importorskip("onnxruntime")
    out = tmp_path / "o.json"
    _run(["--wav", str(DATA / "en_sample.wav"),
          "--transcript", str(DATA / "en_sample.txt"),
          "--model-dir", _real_model_dir(), "--output", str(out)])
    data = json.loads(out.read_text(encoding="utf-8"))
    words = data["words"]
    assert [w["word"] for w in words] == (DATA / "en_sample.txt").read_text().split()
    aligned = [w for w in words if w["status"] == "aligned"]
    assert [w["word"] for w in words if w["status"] != "aligned"] == ["5"]
    starts = [w["start"] for w in aligned]
    assert starts == sorted(starts)
    prev_end = 0.0
    for w in aligned:
        assert prev_end <= w["start"] < w["end"] <= data["audio"]["duration"]
        prev_end = w["end"]
    by_word = {w["word"]: w for w in words}
    assert by_word["Hello,"]["start"] < 0.5          # speech starts right away
    assert by_word["o'clock."]["end"] > 4.0          # and ends near the end
