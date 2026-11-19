# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
source .venv/bin/activate
uv pip install -e '.[dev]'        # pytest, pytest-cov, fastapi, httpx, editdistance
                                  # other extras: server, tts-ja, align, wake, denoise, benchmark

pytest                            # whole suite (CPU-only, no models/hardware needed)
pytest tests/test_asr.py          # one file
pytest tests/test_asr.py::TestParseArgs::test_mic_mode   # one test
pytest --cov=sherox --cov=benchmark

python -m sherox.asr --wav data/happy_50_gtts.wav        # run a CLI without the console script
python benchmark/benchmark.py --data-dir /path/to/LibriSpeech/dev-clean-2 --offline
```

No linter/formatter is configured. `audiokit` (shared resampling + safe tar extraction) is a git dependency pulled via `[tool.uv.sources]`; `sherox.audio` and `sherox.utils` re-export from it.

## Architecture

sherox is a collection of independent CLIs (one module + one `[project.scripts]` entry each: `asr`, `sid`, `lid`, `segment`, `align`, `tts`, `kws`, `wake`, `server`, `asr_server`, `tts_server`, `models`) built on sherpa-onnx. `sherox/main.py` is only a help/`list-models` dispatcher, not a router to the other CLIs.

**ASR data flow** (the core, spans several files): `asr.py` (arg parsing, validation, model auto-download, `resolve_language_model`) → fills a `Config` → `asr_engine.py` (builds sherpa-onnx `OnlineRecognizer`/`OfflineRecognizer`, VAD, diarization, embedding extractor) → `streaming.py` (decode loops fed by the generators in `audio.py`, rich output).
- Online mode: `build_recognizer()` → `run_streaming()`. Offline mode (`--offline`): `build_offline_recognizer()` + `build_vad()` → `run_offline_vad_streaming()`; audio accumulates until VAD silence, then is decoded.
- Diarization runs concurrently with ASR in a `ThreadPoolExecutor` in `streaming.py`.
- `config.py` dataclasses (`Config`, `SidConfig`, `SegmentConfig`, `TtsConfig`, `KwsConfig`, `WakeConfig`, …) are the only way CLI modules pass parameters to engine/streaming layers — add a field rather than a kwarg to `build_*()`.
- `cfg.vad_model` starts empty; `asr._validate_vad()` downloads/resolves it. Never call `build_vad()` before that.
- `asr_engine._find(dir, glob)` is the canonical way to locate ONNX files in a model dir (sorted, first match); follow the glob convention (`"encoder*.onnx"`) for new model types.

**Servers**: `server.py` is a single eagerly-loaded model with WebSocket streaming and a browser demo (`sherox/static/`). `asr_server.py` and `tts_server.py` are separate FastAPI apps that lazily build and cache one recognizer/TTS backend per language (`/transcribe`, `/synthesize`, `/health`). FastAPI is optional (`[server]` extra).

**Forced alignment** (`align.py` + `alignment/`): CLI and JSON schema depend only on `alignment/base.py` (`AlignerBackend`, `WordAlignment`, `AlignmentResult`). New backends subclass `AlignerBackend` and are registered in `_BACKENDS` as `"name": "module:Class"`, imported lazily. The `ctc` backend uses `onnxruntime` (extra `[align]`) + NumPy Viterbi, no PyTorch. Its default model is stored directly in the shared cache dir (not `models/`), downloaded atomically via a temp dir. Unalignable words are kept with `status: "unaligned"` and null timestamps. See `ALIGN_MODEL.md`.

**TTS**: English/Indonesian go through sherpa-onnx; Japanese uses the separate `piper-plus` backend (`[tts-ja]` extra).

**Model storage**: models live in `models/<name>/` (gitignored). `model_cache.py` moves downloaded dirs into a shared cache (`~/.cache/sherox/models`, override with `SHEROX_CACHE_DIR`) and leaves a symlink, so sibling projects share one copy. Cache keys include `model_type`. Per-feature model docs: `ASR_MODEL.md`, `TTS_MODEL.md`, `SID_MODEL.md`, `SEGMENT_MODEL.md`, `DIARIZATION_MODEL.md`, `ALIGN_MODEL.md`; `models.py` holds the catalog behind `sherox list-models`.

`benchmark/` is separate tooling (WER/CER, diarization, Japanese leaderboards) with its own README; tests import from it via a `sys.path` entry in `tests/conftest.py`.

## Conventions

- **Lazy heavy imports**: `sherpa_onnx`, `sounddevice`, `soundfile`, `onnxruntime` are imported inside `_require_*()` helpers at call time, so modules import (and tests run) without them. `asr_engine.py` uses `_MissingSherpaOnnxProxy` so `import sherpa_onnx` never fails at module level — reuse that pattern for new sherpa-onnx types.
- **Errors**: library code raises `SherpaError` subclasses (`ModelNotFoundError`, `AudioError`, `ConfigError`, `AlignmentError` in `sherox/__init__.py`), never `sys.exit()`. CLI `main()` wraps its impl in `utils.run_cli` (`SherpaError` → exit 1, Ctrl+C → 130).
- **Output**: use `rich`; status messages via `_info()` / `_error()` from `utils.py`. The console is `Console(highlight=False, markup=False)` so transcript text isn't parsed as markup.
- **Audio input**: mono 16-bit 16 kHz WAV; `audio.py` resamples from `capture_rate` to `sample_rate`. New audio sources should follow the generator protocol of `mic_stream()` / `read_wav()`.
- **Tests** (`tests/test_<feature>.py`): patch `sys.argv` for CLI tests, replace sherpa-onnx objects with `MagicMock`, patch `sounddevice.InputStream`; never download models. An autouse fixture in `conftest.py` redirects `SHEROX_CACHE_DIR` to a temp dir so tests can't touch the real cache.
- Keep patches targeted; match nearby import ordering. Commit subjects are short and imperative. Update `CHANGELOG.md` (`[Unreleased]`) for user-visible changes; the version must match across `pyproject.toml`, `sherox/__init__.py`, and the latest released heading in `CHANGELOG.md` (enforced by `tests/test_version.py`).
