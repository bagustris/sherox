# Forced Alignment (`sherox.align`)

`sherox.align` takes a recording **and its existing transcript** and returns
word-level start/end timestamps (seconds) as JSON. No ASR is run — the text is
only used to constrain the alignment.

```bash
pip install 'sherox[align]'        # adds onnxruntime (no PyTorch)

sherox.align --wav audio.wav --transcript transcript.txt \
             --language en --output alignment.json
```

Without `--output` the JSON is printed to stdout (logs go to stderr, so the
stream stays valid JSON). Exit code is `1` on any error; nothing is written.

## Backends

| `--backend` | Engine | Extra dependency | Status |
|-------------|--------|------------------|--------|
| `ctc` (default) | Character-level CTC Viterbi over an ONNX acoustic model, run with `onnxruntime` | `sherox[align]` | Implemented |

Alignment engines sit behind a small interface
(`sherox/alignment/base.py::AlignerBackend`). To add one (e.g. WhisperX or
Montreal Forced Aligner), subclass it, implement `align(audio, sample_rate,
words, language)` and register `"name": "module:Class"` in `_BACKENDS`.
Backends are imported lazily, so heavyweight dependencies (PyTorch, WhisperX,
MFA) never become requirements of a default install. The CLI and JSON schema do
not change when a backend is added.

> Sherpa-ONNX itself currently exposes no forced-alignment API (its decoders do
> not return CTC posteriors), which is why the default backend drives
> `onnxruntime` directly and does the Viterbi alignment in NumPy.

## Models and languages

| Model | Language | Size | Location |
|-------|----------|------|----------|
| `wav2vec2-base-960h` (int8 ONNX, [Xenova export](https://huggingface.co/Xenova/wav2vec2-base-960h)) | `en` | ~95 MB | `models/wav2vec2-base-960h-onnx` (auto-downloaded, shared cache) |

The default model supports **English only**; other `--language` values are
rejected with an error. To align another language, pass `--model-dir` pointing
at any character-level CTC model exported to ONNX:

- `*.onnx` (or `model.onnx`) taking a 16 kHz waveform `(batch, samples)` and
  returning logits `(batch, frames, vocab)` — wav2vec2 / HuBERT / MMS CTC
  checkpoints exported with e.g. `optimum-cli export onnx`;
- `vocab.json` (`{token: id}`) with a blank token (`<pad>`, `[PAD]`,
  `<blank>` or `<blk>`) and optionally `|` as the word delimiter;
- optional `preprocessor_config.json` (`do_normalize`, default `true`).

With `--model-dir` the `--language` value is just recorded in the output.

## Output JSON (`schema_version` 1)

```json
{
  "schema_version": 1,
  "language": "en",
  "backend": "ctc",
  "model": "wav2vec2-base-960h",
  "status": "partial",
  "audio": {"path": "audio.wav", "duration": 4.733, "sample_rate": 16000},
  "transcript": "Hello, world! ... at 5 o'clock.",
  "num_words": 12,
  "num_unaligned": 1,
  "words": [
    {"index": 0, "word": "Hello,", "start": 0.02, "end": 0.361, "score": 0.635, "status": "aligned"},
    {"index": 10, "word": "5", "start": null, "end": null, "score": null, "status": "unaligned"}
  ]
}
```

- Every key is always present; missing values are `null`.
- `words` has exactly one record per whitespace-separated transcript word, in
  transcript order. `word` is the original token **including punctuation and
  casing**. Whitespace in the transcript is normalised to single spaces.
- `start` / `end`: seconds from the start of the audio.
- `score`: mean model posterior (0–1) over the word's frames; higher is more
  confident. Backends that cannot provide a confidence emit `null`.
- `status`: `"aligned"` or `"unaligned"`. A word is `unaligned` when it has no
  characters the model can emit (digits, symbols, characters outside the
  vocabulary). It stays in the output with `null` timestamps — it is never
  dropped, and `sherox.align` prints a warning listing them.
- Top-level `status` is `"ok"` when every word is aligned, otherwise
  `"partial"`.
- **Failures** (missing/unreadable audio, empty transcript, transcript longer
  than the audio can support, unsupported language, missing model, missing
  `onnxruntime`) abort with exit code 1 and an `[error]` message on stderr.

## Limitations

- **Resolution** is one model frame, ~20 ms.
- **Long audio.** Inference runs in 20 s windows (2 s of context each side), so
  memory stays flat: on CPU a 5-minute file aligned in ~18 s using ~1.3 GB
  RSS, and results match a single pass to within one frame. The limit is the
  Viterbi trellis (`frames × 2·characters` ≤ 400 M cells, roughly 10–15 minutes
  of speech); beyond that `sherox.align` fails with a clear error — split the
  recording (e.g. with `sherox.segment`) and align the pieces separately.
  Windowed inference assumes the standard 20 ms wav2vec2 frame stride.
- **Transcript must match the audio.** Forced alignment always produces *a*
  result if one exists; a wrong or incomplete transcript yields low `score`s
  and misplaced words rather than an error. Use `score` to flag them.
- **No text normalisation.** Numbers, abbreviations and symbols are not
  expanded ("5" → "five"), so they come back `unaligned`; write them out as
  they are spoken for best results.
- **Word = whitespace-delimited token.** Languages written without spaces
  (Japanese, Chinese) need pre-tokenised transcripts, and need a suitable
  `--model-dir`; the default English model has no such vocabulary.
- **CPU by default**; `--device cuda` needs `onnxruntime-gpu` and fails with a
  clear message otherwise. The default int8 model is CPU-oriented.
